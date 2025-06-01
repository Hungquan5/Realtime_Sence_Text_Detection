from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import numpy as np
import base64
import io
import json
import asyncio
import uuid
import logging

try:
    from paddleocr import PaddleOCR
except ImportError as e:
    raise ImportError(
        "PaddleOCR not found. Install with: pip install paddleocr paddlepaddle-cpu"
    ) from e

from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Scene Text Detection API",
    description="Real-time scene text detection using PaddleOCR - receives frames from frontend",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for streaming
active_websockets: Dict[str, Dict] = {}

class TextDetectionResponse(BaseModel):
    success: bool
    message: str
    detections: List[Dict[str, Any]]
    processing_time: float
    image_info: Optional[Dict[str, Any]] = None

class StreamFrame(BaseModel):
    image: str  # base64 encoded image
    timestamp: float
    frame_id: Optional[str] = None

class StreamConfig(BaseModel):
    lang: str = "en"
    use_gpu: bool = False
    conf_threshold: float = 0.6

class OCRProcessor:
    """Wrapper class for PaddleOCR processing"""
    
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        self.ocr = PaddleOCR(lang=lang)
        self.lang = lang
        self.use_gpu = use_gpu
    
    def parse_ocr_results(self, ocr_result_list):
        """Parse OCR results from PaddleOCR"""
        if not ocr_result_list:
            return [], [], []
        
        ocr_data = ocr_result_list[0]
        if ocr_data is None:
            return [], [], []

        all_boxes, all_texts, all_confidences = [], [], []

        if isinstance(ocr_data, dict):
            texts = ocr_data.get('rec_texts', [])
            scores = ocr_data.get('rec_scores', [])
            boxes = ocr_data.get('rec_polys') or ocr_data.get('dt_polys', [])

            valid_indices = [
                i for i, (t, s, b) in enumerate(zip(texts, scores, boxes))
                if t and s is not None and b is not None and len(b) > 0
            ]

            all_texts = [texts[i] for i in valid_indices]
            all_confidences = [float(scores[i]) for i in valid_indices]
            all_boxes = [boxes[i] for i in valid_indices]
        
        elif isinstance(ocr_data, list):
            for item in ocr_data:
                if isinstance(item, list) and len(item) == 2:
                    box_coords, text_info = item[0], item[1]
                    if isinstance(text_info, (tuple, list)) and len(text_info) == 2:
                        text, confidence = text_info
                        if text and confidence is not None and box_coords:
                            all_boxes.append(box_coords)
                            all_texts.append(text)
                            all_confidences.append(float(confidence))
        
        return all_boxes, all_texts, all_confidences
    
    def process_image(self, image_array: np.ndarray, conf_threshold: float = 0.5):
        """Process image and return detection results"""
        start_time = time.time()
        
        try:
            ocr_result_list = self.ocr.ocr(image_array)
            all_boxes, all_texts, all_confidences = self.parse_ocr_results(ocr_result_list)
            
            detections = []
            for i in range(len(all_texts)):
                if all_confidences[i] >= conf_threshold:
                    detection = {
                        'text': all_texts[i],
                        'confidence': float(all_confidences[i]),
                        'box': [[int(point[0]), int(point[1])] for point in all_boxes[i]]
                    }
                    detections.append(detection)
            
            processing_time = time.time() - start_time
            return detections, processing_time
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

# Global OCR processor instance
ocr_processor = OCRProcessor()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Scene Text Detection API - Frontend streams frames to backend"}

@app.get("/health")
async def health_check():
    """Health check with more details"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "ocr_lang": ocr_processor.lang,
        "gpu_enabled": ocr_processor.use_gpu,
        "active_websockets": len(active_websockets),
        "description": "Backend processes frames received from frontend"
    }

@app.post("/detect/image", response_model=TextDetectionResponse)
async def detect_text_from_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    lang: str = "en"
):
    """Detect text from uploaded image"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to PIL Image then to OpenCV format
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Process with OCR
        detections, processing_time = ocr_processor.process_image(
            opencv_image, conf_threshold
        )
        
        # Get image info
        height, width = opencv_image.shape[:2]
        image_info = {
            "width": width,
            "height": height,
            "filename": file.filename,
            "size_bytes": len(contents)
        }
        
        return TextDetectionResponse(
            success=True,
            message=f"Successfully detected {len(detections)} text regions",
            detections=detections,
            processing_time=processing_time,
            image_info=image_info
        )
        
    except Exception as e:
        logger.error(f"Error in detect_text_from_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect/base64", response_model=TextDetectionResponse)
async def detect_text_from_base64(
    image_data: Dict[str, Any],
    conf_threshold: float = 0.5
):
    """Detect text from base64 encoded image"""
    
    try:
        if 'image' not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        base64_string = image_data['image']
        
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL Image then to OpenCV format
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Process with OCR
        detections, processing_time = ocr_processor.process_image(
            opencv_image, conf_threshold
        )
        
        # Get image info
        height, width = opencv_image.shape[:2]
        image_info = {
            "width": width,
            "height": height,
            "size_bytes": len(image_bytes)
        }
        
        return TextDetectionResponse(
            success=True,
            message=f"Successfully detected {len(detections)} text regions",
            detections=detections,
            processing_time=processing_time,
            image_info=image_info
        )
        
    except Exception as e:
        logger.error(f"Error in detect_text_from_base64: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.websocket("/ws/stream/{stream_id}")
async def websocket_stream_endpoint(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for real-time frame processing"""
    await websocket.accept()
    
    # Store connection info
    active_websockets[stream_id] = {
        "websocket": websocket,
        "config": StreamConfig(),
        "stats": {
            "frames_processed": 0,
            "total_detections": 0,
            "total_processing_time": 0,
            "start_time": time.time()
        },
        "created_at": time.time()
    }
    
    logger.info(f"WebSocket connected for stream {stream_id}")
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "stream_id": stream_id,
            "message": "Ready to receive frames for processing"
        })
        
        # Process incoming frames
        while True:
            try:
                # Receive frame data from frontend
                message = await websocket.receive_json()
                
                if message.get("type") == "frame":
                    # Process the frame
                    await process_stream_frame(stream_id, message)
                    
                elif message.get("type") == "config_update":
                    # Update stream configuration
                    config_data = message.get("config", {})
                    active_websockets[stream_id]["config"] = StreamConfig(**config_data)
                    
                    await websocket.send_json({
                        "type": "config_updated",
                        "message": "Configuration updated successfully"
                    })
                    
                elif message.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time()
                    })
                    
                elif message.get("type") == "stop_stream":
                    # Stop processing
                    break
                    
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message.get('type')}"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing message for stream {stream_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for stream {stream_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket stream {stream_id}: {e}")
    finally:
        # Clean up
        if stream_id in active_websockets:
            del active_websockets[stream_id]
        logger.info(f"Cleaned up stream {stream_id}")

async def process_stream_frame(stream_id: str, message: Dict):
    """Process a single frame from the stream"""
    try:
        if stream_id not in active_websockets:
            return
        
        websocket_info = active_websockets[stream_id]
        websocket = websocket_info["websocket"]
        config = websocket_info["config"]
        stats = websocket_info["stats"]
        
        # Extract frame data
        frame_data = message.get("frame", {})
        base64_image = frame_data.get("image")
        client_timestamp = frame_data.get("timestamp", time.time())
        frame_id = frame_data.get("frame_id")
        
        if not base64_image:
            await websocket.send_json({
                "type": "error",
                "message": "Missing image data in frame"
            })
            return
        
        # Process the frame
        start_time = time.time()
        
        # Remove data URL prefix if present
        if base64_image.startswith('data:image'):
            base64_image = base64_image.split(',')[1]
        
        # Decode and process image
        image_bytes = base64.b64decode(base64_image)
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Run OCR detection
        detections, processing_time = ocr_processor.process_image(
            opencv_image, config.conf_threshold
        )
        
        # Update stats
        stats["frames_processed"] += 1
        stats["total_detections"] += len(detections)
        stats["total_processing_time"] += processing_time
        
        # Send results back to frontend
        result = {
            "type": "detection_result",
            "stream_id": stream_id,
            "frame_id": frame_id,
            "client_timestamp": client_timestamp,
            "server_timestamp": time.time(),
            "processing_time": processing_time,
            "detections": detections,
            "frame_info": {
                "width": opencv_image.shape[1],
                "height": opencv_image.shape[0]
            },
            "stats": {
                "frames_processed": stats["frames_processed"],
                "total_detections": stats["total_detections"],
                "avg_processing_time": stats["total_processing_time"] / stats["frames_processed"],
                "stream_duration": time.time() - stats["start_time"]
            }
        }
        
        await websocket.send_json(result)
        
    except Exception as e:
        logger.error(f"Error processing frame for stream {stream_id}: {e}")
        if stream_id in active_websockets:
            websocket = active_websockets[stream_id]["websocket"]
            await websocket.send_json({
                "type": "error",
                "message": f"Frame processing error: {str(e)}"
            })

@app.get("/streams")
async def list_active_streams():
    """List all active WebSocket streams"""
    
    streams_info = []
    for stream_id, stream_data in active_websockets.items():
        stats = stream_data["stats"]
        streams_info.append({
            "stream_id": stream_id,
            "created_at": stream_data["created_at"],
            "config": stream_data["config"].dict(),
            "stats": {
                "frames_processed": stats["frames_processed"],
                "total_detections": stats["total_detections"],
                "avg_processing_time": stats["total_processing_time"] / max(stats["frames_processed"], 1),
                "stream_duration": time.time() - stats["start_time"]
            }
        })
    
    return {
        "active_streams": len(active_websockets),
        "streams": streams_info
    }

@app.delete("/stream/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop and remove a stream"""
    
    if stream_id not in active_websockets:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Send stop message and clean up
    websocket_info = active_websockets[stream_id]
    try:
        await websocket_info["websocket"].send_json({
            "type": "stream_stopped",
            "message": "Stream stopped by server"
        })
        await websocket_info["websocket"].close()
    except:
        pass
    
    del active_websockets[stream_id]
    
    return {
        "success": True,
        "message": f"Stream {stream_id} stopped and removed"
    }

@app.post("/ocr/config")
async def update_ocr_config(lang: str = "en", use_gpu: bool = False):
    """Update OCR configuration (reinitializes OCR processor)"""
    
    global ocr_processor
    
    try:
        ocr_processor = OCRProcessor(lang=lang, use_gpu=use_gpu)
        
        return {
            "success": True,
            "message": "OCR configuration updated successfully",
            "config": {
                "language": lang,
                "gpu_enabled": use_gpu
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating OCR config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

@app.get("/ocr/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    
    # Common PaddleOCR supported languages
    supported_languages = [
        {"code": "en", "name": "English"},
        {"code": "ch", "name": "Chinese"},
        {"code": "korean", "name": "Korean"},
        {"code": "japan", "name": "Japanese"},
        {"code": "ta", "name": "Tamil"},
        {"code": "te", "name": "Telugu"},
        {"code": "ka", "name": "Kannada"},
        {"code": "hi", "name": "Hindi"},
        {"code": "ar", "name": "Arabic"},
        {"code": "cyrillic", "name": "Cyrillic"},
        {"code": "devanagari", "name": "Devanagari"}
    ]
    
    return {
        "supported_languages": supported_languages,
        "current_language": ocr_processor.lang
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)