from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import time
import numpy as np
import base64
import io
import json
from pathlib import Path
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

try:
    from paddleocr import PaddleOCR
except ImportError as e:
    raise ImportError(
        "PaddleOCR not found. Install with: pip install paddleocr paddlepaddle-cpu"
    ) from e

from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Scene Text Detection API",
    description="Real-time scene text detection using PaddleOCR",
    version="1.0.0"
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
active_streams: Dict[str, Any] = {}
executor = ThreadPoolExecutor(max_workers=4)

class TextDetectionResponse(BaseModel):
    success: bool
    message: str
    detections: List[Dict[str, Any]]
    processing_time: float
    image_info: Optional[Dict[str, Any]] = None

class StreamConfig(BaseModel):
    camera_index: int = 0
    lang: str = "en"
    use_gpu: bool = False
    detection_interval: int = 5
    conf_threshold: float = 0.6
    cv_font_scale: float = 0.5

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
            ocr_result_list = self.ocr.predict(image_array)
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
    return {"message": "Scene Text Detection API is running"}

@app.get("/health")
async def health_check():
    """Health check with more details"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "ocr_lang": ocr_processor.lang,
        "gpu_enabled": ocr_processor.use_gpu
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

@app.post("/stream/start")
async def start_stream(config: StreamConfig):
    """Start a new video stream for text detection"""
    
    stream_id = str(uuid.uuid4())
    
    try:
        # Test if camera/video source is accessible
        cap = cv2.VideoCapture(config.camera_index)
        if not cap.isOpened():
            cap.release()
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot open camera/video source: {config.camera_index}"
            )
        
        # Get stream info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Store stream configuration
        active_streams[stream_id] = {
            "config": config,
            "status": "initialized",
            "stream_info": {
                "width": width,
                "height": height,
                "fps": fps
            },
            "created_at": time.time(),
            "last_detection": None
        }
        
        return {
            "success": True,
            "stream_id": stream_id,
            "message": "Stream initialized successfully",
            "stream_info": active_streams[stream_id]["stream_info"]
        }
        
    except Exception as e:
        logger.error(f"Error starting stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")

@app.get("/stream/{stream_id}/detect")
async def detect_from_stream(stream_id: str):
    """Get latest detection results from active stream"""
    
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    stream_data = active_streams[stream_id]
    config = stream_data["config"]
    
    try:
        cap = cv2.VideoCapture(config.camera_index)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot access video source")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=400, detail="Failed to capture frame")
        
        # Process frame with OCR
        detections, processing_time = ocr_processor.process_image(
            frame, config.conf_threshold
        )
        
        # Update stream data
        detection_result = {
            "detections": detections,
            "processing_time": processing_time,
            "timestamp": time.time(),
            "frame_info": {
                "height": frame.shape[0],
                "width": frame.shape[1]
            }
        }
        
        active_streams[stream_id]["last_detection"] = detection_result
        active_streams[stream_id]["status"] = "active"
        
        return {
            "success": True,
            "stream_id": stream_id,
            **detection_result
        }
        
    except Exception as e:
        logger.error(f"Error detecting from stream {stream_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.get("/stream/{stream_id}/status")
async def get_stream_status(stream_id: str):
    """Get status of a specific stream"""
    
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    stream_data = active_streams[stream_id]
    
    return {
        "stream_id": stream_id,
        "status": stream_data["status"],
        "config": stream_data["config"],
        "stream_info": stream_data["stream_info"],
        "created_at": stream_data["created_at"],
        "has_recent_detection": stream_data["last_detection"] is not None,
        "last_detection_time": (
            stream_data["last_detection"]["timestamp"] 
            if stream_data["last_detection"] else None
        )
    }

@app.delete("/stream/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop and remove a stream"""
    
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    del active_streams[stream_id]
    
    return {
        "success": True,
        "message": f"Stream {stream_id} stopped and removed"
    }

@app.get("/streams")
async def list_active_streams():
    """List all active streams"""
    
    streams_info = []
    for stream_id, stream_data in active_streams.items():
        streams_info.append({
            "stream_id": stream_id,
            "status": stream_data["status"],
            "created_at": stream_data["created_at"],
            "config": stream_data["config"],
            "has_recent_detection": stream_data["last_detection"] is not None
        })
    
    return {
        "active_streams": len(active_streams),
        "streams": streams_info
    }

@app.post("/ocr/config")
async def update_ocr_config(lang: str = "en", use_gpu: bool = False):
    """Update OCR configuration (reinitializes OCR processor)"""
    
    global ocr_processor
    
    try:
        ocr_processor = OCRProcessor(lang=lang)
        
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