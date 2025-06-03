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
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import queue
import multiprocessing as mp

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
    title="Optimized Scene Text Detection API",
    description="High-performance real-time scene text detection using PaddleOCR",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for optimization
MAX_WORKERS = min(4, mp.cpu_count())  # Limit concurrent processing
THREAD_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)
FRAME_QUEUE_SIZE = 10  # Limit queue to prevent memory buildup

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
    max_image_size: int = 1280  # Resize larger images for faster processing
    skip_detection_angle: bool = True  # Skip rotation detection for speed
    use_angle_cls: bool = False  # Disable angle classification
    det_db_thresh: float = 0.3  # Detection threshold
    det_db_box_thresh: float = 0.6  # Box threshold
# Add these imports at the top of your existing file
import os
from typing import Union
import google.generativeai as genai
from dataclasses import dataclass
import re
import json as json_module

# Add these new classes after your existing models
class LLMConfig(BaseModel):
    enabled: bool = True
    api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 1000
    correction_prompt_type: str = "general"  # general, structured, numbers, etc.

class EnhancedTextDetectionResponse(BaseModel):
    success: bool
    message: str
    detections: List[Dict[str, Any]]
    processing_time: float
    image_info: Optional[Dict[str, Any]] = None
    llm_corrections: Optional[Dict[str, Any]] = None
    total_processing_time: float

@dataclass
class CorrectionResult:
    original_text: str
    corrected_text: str
    confidence_score: float
    correction_applied: bool
    correction_reason: str

class GeminiLLMProcessor:
    """Enhanced LLM processor using Google's Gemini API for OCR post-processing"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key in config")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(config.model_name)
        
        # Predefined correction prompts for different contexts
        self.correction_prompts = {
            "general": """
You are an OCR correction expert. Analyze the following OCR-detected text and provide corrections for any obvious errors.

Rules:
1. Fix spelling mistakes and typos
2. Correct obvious character recognition errors (e.g., '0' vs 'O', '1' vs 'l', '5' vs 'S')
3. Improve capitalization and punctuation
4. Maintain the original meaning and context
5. Don't add new information not present in the original
6. If text appears to be a specific format (phone numbers, emails, addresses), correct accordingly

Original OCR Text: "{text}"

Respond in this exact JSON format:
{{
    "corrected_text": "corrected version here",
    "confidence_score": 0.95,
    "corrections_made": ["list of specific corrections"],
    "reasoning": "brief explanation of corrections"
}}
""",
            
            "structured": """
You are an OCR correction expert specializing in structured data (forms, tables, documents).

Analyze this OCR text and correct errors while preserving structure:

Rules:
1. Fix character recognition errors common in forms
2. Correct number/letter confusion (0/O, 1/l/I, 5/S, etc.)
3. Fix spacing issues in structured data
4. Correct obvious field labels and values
5. Maintain original formatting structure
6. Fix date formats and number formats

Original OCR Text: "{text}"

Respond in JSON format:
{{
    "corrected_text": "corrected version",
    "confidence_score": 0.90,
    "corrections_made": ["specific corrections"],
    "reasoning": "explanation",
    "detected_format": "form/table/document/other"
}}
""",
            
            "numbers": """
You are an OCR correction expert specializing in numerical data.

Correct this OCR text focusing on numbers, dates, and numerical patterns:

Rules:
1. Fix digit recognition errors (0/O, 1/l, 5/S, 6/G, 8/B, etc.)
2. Correct phone numbers, IDs, codes, prices
3. Fix date and time formats
4. Correct mathematical expressions or measurements
5. Fix decimal points and thousands separators

Original OCR Text: "{text}"

JSON Response:
{{
    "corrected_text": "corrected version",
    "confidence_score": 0.92,
    "corrections_made": ["corrections list"],
    "reasoning": "explanation",
    "number_corrections": ["specific number fixes"]
}}
""",
            
            "contextual": """
You are an OCR correction expert with contextual understanding.

Analyze this text considering its likely context and correct errors:

Rules:
1. Consider the context and likely content type
2. Fix errors based on common word patterns
3. Correct grammar and sentence structure
4. Fix punctuation and capitalization
5. Resolve ambiguous characters using context
6. Maintain original intent and meaning

Original OCR Text: "{text}"
Context Hint: Look for common document types, signs, labels, or text patterns.

JSON Response:
{{
    "corrected_text": "corrected version",
    "confidence_score": 0.88,
    "corrections_made": ["corrections"],
    "reasoning": "explanation",
    "detected_context": "likely content type"
}}
"""
        }
    
    async def correct_text_batch(self, detections: List[Dict], prompt_type: str = "general") -> Dict[str, Any]:
        """Process multiple text detections with LLM corrections"""
        if not self.config.enabled or not detections:
            return {
                "enabled": False,
                "total_corrections": 0,
                "corrected_detections": detections,
                "processing_time": 0,
                "model_used": None
            }
        
        start_time = time.time()
        corrected_detections = []
        total_corrections = 0
        correction_details = []
        
        try:
            # Process texts in batches for efficiency
            batch_size = 5  # Process 5 texts at once
            
            for i in range(0, len(detections), batch_size):
                batch = detections[i:i + batch_size]
                batch_results = await self._process_batch(batch, prompt_type)
                
                corrected_detections.extend(batch_results["detections"])
                total_corrections += batch_results["corrections_count"]
                correction_details.extend(batch_results["details"])
                
                # Small delay to respect API limits
                if i + batch_size < len(detections):
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"LLM correction failed: {str(e)}")
            # Return original detections if LLM fails
            corrected_detections = detections
            correction_details = [{"error": str(e)}]
        
        processing_time = time.time() - start_time
        
        return {
            "enabled": True,
            "total_corrections": total_corrections,
            "corrected_detections": corrected_detections,
            "processing_time": processing_time,
            "model_used": self.config.model_name,
            "correction_details": correction_details,
            "prompt_type": prompt_type
        }
    
    async def _process_batch(self, batch: List[Dict], prompt_type: str) -> Dict:
        """Process a batch of detections"""
        corrections_count = 0
        corrected_batch = []
        details = []
        
        # Combine texts for batch processing
        texts_to_process = [detection["text"] for detection in batch]
        
        # Create batch prompt
        if len(texts_to_process) == 1:
            prompt = self.correction_prompts[prompt_type].format(text=texts_to_process[0])
        else:
            # Multi-text batch prompt
            text_list = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts_to_process)])
            prompt = f"""
You are an OCR correction expert. Correct the following {len(texts_to_process)} OCR-detected texts:

{text_list}

Respond with a JSON array containing corrections for each text:
[
    {{
        "index": 1,
        "corrected_text": "corrected version",
        "confidence_score": 0.95,
        "corrections_made": ["corrections"],
        "reasoning": "explanation"
    }},
    ...
]
"""
        
        try:
            # Make API call to Gemini
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                THREAD_POOL,
                self._call_gemini_api,
                prompt
            )
            
            # Parse response
            if len(texts_to_process) == 1:
                corrections = [self._parse_single_response(response)]
            else:
                corrections = self._parse_batch_response(response, len(texts_to_process))
            
            # Apply corrections to detections
            for i, detection in enumerate(batch):
                original_text = detection["text"]
                
                if i < len(corrections) and corrections[i]:
                    correction = corrections[i]
                    corrected_text = correction.get("corrected_text", original_text)
                    
                    # Update detection with corrected text
                    corrected_detection = detection.copy()
                    corrected_detection["text"] = corrected_text
                    corrected_detection["original_text"] = original_text
                    corrected_detection["llm_correction"] = {
                        "applied": corrected_text != original_text,
                        "confidence": correction.get("confidence_score", 0.0),
                        "corrections_made": correction.get("corrections_made", []),
                        "reasoning": correction.get("reasoning", "")
                    }
                    
                    if corrected_text != original_text:
                        corrections_count += 1
                    
                    corrected_batch.append(corrected_detection)
                    details.append({
                        "original": original_text,
                        "corrected": corrected_text,
                        "changed": corrected_text != original_text
                    })
                else:
                    # No correction available
                    detection["llm_correction"] = {
                        "applied": False,
                        "confidence": 0.0,
                        "error": "Failed to process"
                    }
                    corrected_batch.append(detection)
                    details.append({
                        "original": original_text,
                        "corrected": original_text,
                        "changed": False,
                        "error": "Processing failed"
                    })
        
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            # Return original detections with error info
            for detection in batch:
                detection["llm_correction"] = {
                    "applied": False,
                    "error": str(e)
                }
                corrected_batch.append(detection)
                details.append({
                    "original": detection["text"],
                    "corrected": detection["text"],
                    "changed": False,
                    "error": str(e)
                })
        
        return {
            "detections": corrected_batch,
            "corrections_count": corrections_count,
            "details": details
        }
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API synchronously"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    def _parse_single_response(self, response: str) -> Dict:
        """Parse single correction response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json_module.loads(json_match.group())
            else:
                # Fallback parsing
                return {"corrected_text": response.strip(), "confidence_score": 0.5}
        except Exception as e:
            logger.error(f"Failed to parse single response: {str(e)}")
            return {}
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[Dict]:
        """Parse batch correction response"""
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                corrections = json_module.loads(json_match.group())
                return corrections[:expected_count]  # Ensure we don't have extra results
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to parse batch response: {str(e)}")
            return []

# Initialize LLM processor (add this after ocr_processor initialization)
llm_config = LLMConfig(
    enabled=bool(os.getenv('GOOGLE_API_KEY')),  # Only enable if API key is available
    api_key=os.getenv('GOOGLE_API_KEY'),
    model_name="gemini-1.5-flash",
    temperature=0.1
)

try:
    llm_processor = GeminiLLMProcessor(llm_config) if llm_config.enabled else None
except Exception as e:
    logger.warning(f"LLM processor initialization failed: {str(e)}")
    llm_processor = None

# Enhanced endpoint for image detection with LLM correction
@app.post("/detect/image/enhanced", response_model=EnhancedTextDetectionResponse)
async def detect_text_from_image_enhanced(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    lang: str = "en",
    enable_llm_correction: bool = True,
    correction_type: str = "general"  # general, structured, numbers, contextual
):
    """Enhanced text detection with LLM post-processing"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    total_start_time = time.time()
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Process with OCR
        loop = asyncio.get_event_loop()
        opencv_image = await loop.run_in_executor(
            THREAD_POOL,
            lambda: decode_and_convert_image(base64.b64encode(contents).decode())
        )
        
        # OCR Processing
        detections, ocr_processing_time = await process_image_async(
            opencv_image, conf_threshold
        )
        
        # LLM Post-processing
        llm_corrections = None
        if enable_llm_correction and llm_processor and detections:
            try:
                llm_corrections = await llm_processor.correct_text_batch(
                    detections, correction_type
                )
                # Use corrected detections if available
                if llm_corrections.get("corrected_detections"):
                    detections = llm_corrections["corrected_detections"]
            except Exception as e:
                logger.error(f"LLM correction failed: {str(e)}")
                llm_corrections = {
                    "enabled": False,
                    "error": str(e),
                    "processing_time": 0
                }
        
        # Get image info
        height, width = opencv_image.shape[:2]
        image_info = {
            "width": width,
            "height": height,
            "filename": file.filename,
            "size_bytes": len(contents),
            "processed_size": f"{opencv_image.shape[1]}x{opencv_image.shape[0]}"
        }
        
        total_processing_time = time.time() - total_start_time
        
        return EnhancedTextDetectionResponse(
            success=True,
            message=f"Successfully detected {len(detections)} text regions with LLM enhancement",
            detections=detections,
            processing_time=ocr_processing_time,
            image_info=image_info,
            llm_corrections=llm_corrections,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Enhanced endpoint for base64 detection with LLM correction
@app.post("/detect/base64/enhanced", response_model=EnhancedTextDetectionResponse)
async def detect_text_from_base64_enhanced(
    image_data: Dict[str, Any],
    conf_threshold: float = 0.5,
    enable_llm_correction: bool = True,
    correction_type: str = "general"
):
    """Enhanced base64 text detection with LLM post-processing"""
    
    total_start_time = time.time()
    
    try:
        if 'image' not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        base64_string = image_data['image']
        
        # OCR Processing
        loop = asyncio.get_event_loop()
        opencv_image = await loop.run_in_executor(
            THREAD_POOL,
            decode_and_convert_image,
            base64_string
        )
        
        detections, ocr_processing_time = await process_image_async(
            opencv_image, conf_threshold
        )
        
        # LLM Post-processing
        llm_corrections = None
        if enable_llm_correction and llm_processor and detections:
            try:
                llm_corrections = await llm_processor.correct_text_batch(
                    detections, correction_type
                )
                if llm_corrections.get("corrected_detections"):
                    detections = llm_corrections["corrected_detections"]
            except Exception as e:
                logger.error(f"LLM correction failed: {str(e)}")
                llm_corrections = {
                    "enabled": False,
                    "error": str(e),
                    "processing_time": 0
                }
        
        # Get image info
        height, width = opencv_image.shape[:2]
        image_info = {
            "width": width,
            "height": height,
            "processed_size": f"{opencv_image.shape[1]}x{opencv_image.shape[0]}",
            "size_bytes": len(base64.b64decode(base64_string.split(',')[1] if 'data:image' in base64_string else base64_string))
        }
        
        total_processing_time = time.time() - total_start_time
        
        return EnhancedTextDetectionResponse(
            success=True,
            message=f"Successfully detected {len(detections)} text regions with LLM enhancement",
            detections=detections,
            processing_time=ocr_processing_time,
            image_info=image_info,
            llm_corrections=llm_corrections,
            total_processing_time=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced base64 detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Configuration endpoint for LLM settings
@app.post("/llm/config")
async def update_llm_config(
    enabled: bool = True,
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.1,
    max_tokens: int = 1000
):
    """Update LLM configuration"""
    
    global llm_processor, llm_config
    
    try:
        # Update config
        llm_config = LLMConfig(
            enabled=enabled,
            api_key=api_key or os.getenv('GOOGLE_API_KEY'),
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Reinitialize processor
        if enabled and llm_config.api_key:
            llm_processor = GeminiLLMProcessor(llm_config)
        else:
            llm_processor = None
        
        return {
            "success": True,
            "message": "LLM configuration updated",
            "config": {
                "enabled": enabled,
                "model_name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key_configured": bool(llm_config.api_key)
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating LLM config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")

# Test LLM correction endpoint
@app.post("/llm/test")
async def test_llm_correction(
    text: str,
    correction_type: str = "general"
):
    """Test LLM correction on sample text"""
    
    if not llm_processor:
        raise HTTPException(status_code=400, detail="LLM processor not configured")
    
    try:
        # Create fake detection for testing
        test_detection = [{"text": text, "confidence": 0.9, "box": [[0, 0], [100, 0], [100, 20], [0, 20]]}]
        
        result = await llm_processor.correct_text_batch(test_detection, correction_type)
        
        return {
            "success": True,
            "original_text": text,
            "correction_result": result,
            "model_used": llm_config.model_name
        }
        
    except Exception as e:
        logger.error(f"Error testing LLM correction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")

# Enhanced health check with LLM status
@app.get("/health/enhanced")
async def enhanced_health_check():
    """Enhanced health check including LLM status"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "ocr": {
            "lang": ocr_processor.lang,
            "gpu_enabled": ocr_processor.use_gpu,
            "max_image_size": ocr_processor.max_image_size
        },
        "llm": {
            "enabled": llm_processor is not None,
            "model": llm_config.model_name if llm_config else None,
            "api_key_configured": bool(llm_config and llm_config.api_key)
        },
        "performance": {
            "max_workers": MAX_WORKERS,
            "active_websockets": len(active_websockets)
        }
    }

# Add this at the end of the file for requirements
"""
Additional requirements to add to your requirements.txt:

google-generativeai>=0.3.0

Environment variables to set:
GOOGLE_API_KEY=your_google_ai_studio_api_key
"""
class OptimizedOCRProcessor:
    """Optimized wrapper class for PaddleOCR processing"""
    
    def __init__(self, lang: str = "en", use_gpu: bool = False, config: StreamConfig = None):
        self.lang = lang
        self.use_gpu = use_gpu
        
        # Initialize with optimized parameters
        ocr_params = {
            'use_angle_cls': config.use_angle_cls if config else False,
            'lang': lang,
            'det_db_thresh': config.det_db_thresh if config else 0.3,
            'det_db_box_thresh': config.det_db_box_thresh if config else 0.6,
            'device': 'cpu'
        }
        
        if use_gpu:
            ocr_params.update({
                'det_db_unclip_ratio': 1.5,  # Optimize for GPU
                'max_text_length': 25  # Limit text length for speed
            })
        
        self.ocr = PaddleOCR(
    use_angle_cls=False,
    enable_hpi=True,
    # doc_orientation_classify_model_dir="/media/quannh/DATA/Scene_text_detection_realtime/backend/onnx_models/PP-LCNet_x0_25_textline_ori",
    # layout_detection_model_dir="/media/quannh/DATA/Scene_text_detection_realtime/backend/onnx_models/PP-LCNet_x1_0_doc_ori",	
    det_model_dir='/media/quannh/DATA/Scene_text_detection_realtime/backend/onnx_models/PP-OCRv5_mobile_det',
    rec_model_dir='/media/quannh/DATA/Scene_text_detection_realtime/backend/onnx_models/PP-OCRv5_mobile_rec',
    **ocr_params
)

        self.max_image_size = config.max_image_size if config else 1280
        
        # Cache for frequent operations
        self._detection_cache = {}
        self._cache_lock = threading.Lock()
    
    def _resize_image_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it's too large to speed up processing"""
        height, width = image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Use faster interpolation for resizing
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Optimized image preprocessing with validation"""
        try:
            # Validate input image
            if image is None or image.size == 0:
                raise ValueError("Invalid image: empty or None")
            
            if len(image.shape) not in [2, 3]:
                raise ValueError(f"Invalid image shape: {image.shape}")
            
            # Ensure image has valid dimensions
            height, width = image.shape[:2]
            if height < 10 or width < 10:
                raise ValueError(f"Image too small: {width}x{height}")
            
            if height > 10000 or width > 10000:
                raise ValueError(f"Image too large: {width}x{height}")
            
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Keep as RGB (already in correct format from PIL)
                pass
            else:
                raise ValueError(f"Unsupported image format: {image.shape}")
            
            # Resize if needed
            image = self._resize_image_if_needed(image)
            
            # Ensure contiguous array for PaddleOCR
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
            
            # Ensure correct data type
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    @lru_cache(maxsize=128)
    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Generate hash for image caching"""
        import hashlib
        return hashlib.md5(image_bytes).hexdigest()[:16]
    
    def parse_ocr_results(self, ocr_result_list):
        """Optimized OCR results parsing"""
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

            # Vectorized filtering for better performance
            valid_mask = np.array([
                bool(t and s is not None and b is not None and len(b) > 0)
                for t, s, b in zip(texts, scores, boxes)
            ])
            
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                all_texts = [texts[i] for i in valid_indices]
                all_confidences = [float(scores[i]) for i in valid_indices]
                all_boxes = [boxes[i] for i in valid_indices]
        
        elif isinstance(ocr_data, list):
            # Pre-allocate lists for better performance
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
        """Optimized image processing with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Validate input
            if image_array is None:
                raise ValueError("Input image is None")
            
            if not isinstance(image_array, np.ndarray):
                raise ValueError(f"Input must be numpy array, got {type(image_array)}")
            
            logger.debug(f"Processing image with shape: {image_array.shape}, dtype: {image_array.dtype}")
            
            # Preprocess image for optimal OCR performance
            try:
                processed_image = self._preprocess_image(image_array)
                logger.debug(f"Preprocessed image shape: {processed_image.shape}")
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                raise ValueError(f"Image preprocessing failed: {str(e)}")
            
            # Run OCR detection with error handling
            try:
                # Create a copy to ensure memory safety
                ocr_input = processed_image.copy()
                
                # Run OCR with timeout protection
                ocr_result_list = self.ocr.predict(ocr_input)
                
                if ocr_result_list is None:
                    logger.warning("OCR returned None result")
                    return [], time.time() - start_time
                
            except Exception as e:
                logger.error(f"OCR processing failed: {str(e)}")
                # Try with a smaller image if OCR fails
                try:
                    height, width = processed_image.shape[:2]
                    if max(height, width) > 640:
                        scale = 640 / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        smaller_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        ocr_result_list = self.ocr.ocr(smaller_image)
                        logger.info("Successfully processed with smaller image")
                    else:
                        raise e
                except Exception as retry_error:
                    logger.error(f"Retry with smaller image also failed: {str(retry_error)}")
                    return [], time.time() - start_time
            
            # Parse results with error handling
            try:
                all_boxes, all_texts, all_confidences = self.parse_ocr_results(ocr_result_list)
            except Exception as e:
                logger.error(f"Result parsing failed: {str(e)}")
                return [], time.time() - start_time
            
            # Filter results by confidence
            detections = []
            if all_confidences:
                try:
                    # Vectorized confidence filtering
                    conf_mask = np.array(all_confidences) >= conf_threshold
                    
                    if conf_mask.any():
                        valid_indices = np.where(conf_mask)[0]
                        
                        for i in valid_indices:
                            try:
                                # Scale boxes back if image was resized
                                box = all_boxes[i]
                                original_height, original_width = image_array.shape[:2]
                                processed_height, processed_width = processed_image.shape[:2]
                                
                                if (original_height != processed_height) or (original_width != processed_width):
                                    scale_x = original_width / processed_width
                                    scale_y = original_height / processed_height
                                    box = [[int(point[0] * scale_x), int(point[1] * scale_y)] for point in box]
                                else:
                                    box = [[int(point[0]), int(point[1])] for point in box]
                                
                                detection = {
                                    'text': all_texts[i],
                                    'confidence': float(all_confidences[i]),
                                    'box': box
                                }
                                detections.append(detection)
                                
                            except Exception as e:
                                logger.warning(f"Error processing detection {i}: {str(e)}")
                                continue
                                
                except Exception as e:
                    logger.error(f"Error in confidence filtering: {str(e)}")
            
            processing_time = time.time() - start_time
            logger.debug(f"Processing completed in {processing_time:.3f}s, found {len(detections)} detections")
            return detections, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Critical error processing image: {str(e)}")
            # Return empty results instead of raising exception
            return [], processing_time

class FrameBuffer:
    """Circular buffer for managing frames to prevent memory buildup"""
    
    def __init__(self, maxsize: int = FRAME_QUEUE_SIZE):
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)
        self.dropped_frames = 0
    
    def put_frame(self, frame_data: Dict) -> bool:
        """Add frame to buffer, dropping oldest if full"""
        try:
            self.queue.put_nowait(frame_data)
            return True
        except queue.Full:
            # Drop oldest frame
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame_data)
                self.dropped_frames += 1
                return True
            except queue.Empty:
                return False
    
    def get_frame(self) -> Optional[Dict]:
        """Get next frame from buffer"""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
    
    def clear(self):
        """Clear all frames from buffer"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

# Global OCR processor instance with optimization
default_config = StreamConfig()
ocr_processor = OptimizedOCRProcessor(config=default_config)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Optimized Scene Text Detection API - High Performance Backend"}

@app.get("/health")
async def health_check():
    """Health check with performance metrics"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "ocr_lang": ocr_processor.lang,
        "gpu_enabled": ocr_processor.use_gpu,
        "max_workers": MAX_WORKERS,
        "active_websockets": len(active_websockets),
        "max_image_size": ocr_processor.max_image_size,
        "description": "Optimized backend with concurrent processing and image resizing"
    }

async def process_image_async(image_array: np.ndarray, conf_threshold: float = 0.5):
    """Async wrapper for image processing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        THREAD_POOL, 
        ocr_processor.process_image, 
        image_array, 
        conf_threshold
    )

def decode_and_convert_image(base64_string: str) -> np.ndarray:
    """Optimized image decoding and conversion with robust error handling"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Validate base64 string
        if not base64_string or len(base64_string) < 10:
            raise ValueError("Invalid or empty base64 string")
        
        # Decode base64 with error handling
        try:
            image_bytes = base64.b64decode(base64_string, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        
        if len(image_bytes) < 100:  # Minimum reasonable image size
            raise ValueError("Decoded image data too small")
        
        # Convert to PIL Image with error handling
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Validate image
            if pil_image.width < 10 or pil_image.height < 10:
                raise ValueError(f"Image too small: {pil_image.width}x{pil_image.height}")
            
            if pil_image.width > 10000 or pil_image.height > 10000:
                raise ValueError(f"Image too large: {pil_image.width}x{pil_image.height}")
            
            # Convert to RGB
            if pil_image.mode not in ['RGB', 'L']:
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode == 'L':
                pil_image = pil_image.convert('RGB')  # Convert grayscale to RGB
            
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
        
        # Convert to numpy array safely
        try:
            opencv_image = np.array(pil_image)
            
            # Ensure array is valid
            if opencv_image.size == 0:
                raise ValueError("Converted image array is empty")
            
            # Ensure correct shape and data type
            if len(opencv_image.shape) == 2:
                # Grayscale, convert to RGB
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2RGB)
            elif len(opencv_image.shape) == 3 and opencv_image.shape[2] == 4:
                # RGBA, convert to RGB
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2RGB)
            
            # Ensure uint8 data type
            if opencv_image.dtype != np.uint8:
                opencv_image = opencv_image.astype(np.uint8)
            
            return opencv_image
            
        except Exception as e:
            raise ValueError(f"Failed to convert image to array: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in decode_and_convert_image: {str(e)}")
        raise

@app.post("/detect/image", response_model=TextDetectionResponse)
async def detect_text_from_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5,
    lang: str = "en"
):
    """Optimized text detection from uploaded image"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        opencv_image = await loop.run_in_executor(
            THREAD_POOL,
            lambda: decode_and_convert_image(base64.b64encode(contents).decode())
        )
        
        # Process with OCR asynchronously
        detections, processing_time = await process_image_async(
            opencv_image, conf_threshold
        )
        
        # Get image info
        height, width = opencv_image.shape[:2]
        image_info = {
            "width": width,
            "height": height,
            "filename": file.filename,
            "size_bytes": len(contents),
            "processed_size": f"{opencv_image.shape[1]}x{opencv_image.shape[0]}"
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
    """Optimized text detection from base64 encoded image"""
    
    try:
        if 'image' not in image_data:
            raise HTTPException(status_code=400, detail="Missing 'image' field in request")
        
        base64_string = image_data['image']
        
        # Decode and convert image asynchronously
        loop = asyncio.get_event_loop()
        opencv_image = await loop.run_in_executor(
            THREAD_POOL,
            decode_and_convert_image,
            base64_string
        )
        
        # Process with OCR asynchronously
        detections, processing_time = await process_image_async(
            opencv_image, conf_threshold
        )
        
        # Get image info
        height, width = opencv_image.shape[:2]
        image_info = {
            "width": width,
            "height": height,
            "processed_size": f"{opencv_image.shape[1]}x{opencv_image.shape[0]}",
            "size_bytes": len(base64.b64decode(base64_string.split(',')[1] if 'data:image' in base64_string else base64_string))
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
    """Optimized WebSocket endpoint for real-time frame processing"""
    await websocket.accept()
    
    # Initialize frame buffer for this stream
    frame_buffer = FrameBuffer(maxsize=FRAME_QUEUE_SIZE)
    
    # Store connection info
    active_websockets[stream_id] = {
        "websocket": websocket,
        "config": StreamConfig(),
        "frame_buffer": frame_buffer,
        "processing": False,
        "stats": {
            "frames_processed": 0,
            "frames_dropped": 0,
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
            "message": "Ready to receive frames for high-speed processing"
        })
        
        # Start frame processing task
        processing_task = asyncio.create_task(
            process_stream_frames(stream_id)
        )
        
        # Handle incoming messages
        while True:
            try:
                message = await websocket.receive_json()
                
                if message.get("type") == "frame":
                    # Add frame to buffer (non-blocking)
                    frame_added = frame_buffer.put_frame(message)
                    if not frame_added:
                        active_websockets[stream_id]["stats"]["frames_dropped"] += 1
                    
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
        
        # Cancel processing task
        processing_task.cancel()
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for stream {stream_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket stream {stream_id}: {e}")
    finally:
        # Clean up
        if stream_id in active_websockets:
            active_websockets[stream_id]["frame_buffer"].clear()
            del active_websockets[stream_id]
        logger.info(f"Cleaned up stream {stream_id}")

async def process_stream_frames(stream_id: str):
    """Continuous frame processing for a stream"""
    while stream_id in active_websockets:
        try:
            websocket_info = active_websockets[stream_id]
            frame_buffer = websocket_info["frame_buffer"]
            
            # Get next frame from buffer
            message = frame_buffer.get_frame()
            if message is None:
                # No frames available, sleep briefly
                await asyncio.sleep(0.01)
                continue
            
            # Process the frame
            await process_single_frame(stream_id, message)
            
        except Exception as e:
            logger.error(f"Error in frame processing loop for stream {stream_id}: {e}")
            await asyncio.sleep(0.1)

async def process_single_frame(stream_id: str, message: Dict):
    """Process a single frame from the stream with comprehensive error handling"""
    try:
        if stream_id not in active_websockets:
            logger.warning(f"Stream {stream_id} not found in active websockets")
            return
        
        websocket_info = active_websockets[stream_id]
        websocket = websocket_info["websocket"]
        config = websocket_info["config"]
        stats = websocket_info["stats"]
        
        # Extract frame data
        frame_data = message.get("frame", {})
        base64_image = frame_data.get("image")
        client_timestamp = frame_data.get("timestamp", time.time())
        frame_id = frame_data.get("frame_id", str(uuid.uuid4()))
        
        if not base64_image:
            await websocket.send_json({
                "type": "error",
                "message": "Missing image data in frame",
                "frame_id": frame_id
            })
            return
        
        # Process the frame asynchronously with comprehensive error handling
        start_time = time.time()
        
        try:
            # Decode and convert image in thread pool with timeout
            loop = asyncio.get_event_loop()
            
            # Set a timeout for image decoding
            opencv_image = await asyncio.wait_for(
                loop.run_in_executor(
                    THREAD_POOL,
                    decode_and_convert_image,
                    base64_image
                ),
                timeout=5.0  # 5 second timeout
            )
            
            logger.debug(f"Frame {frame_id}: Decoded image shape {opencv_image.shape}")
            
        except asyncio.TimeoutError:
            logger.error(f"Frame {frame_id}: Image decoding timeout")
            await websocket.send_json({
                "type": "error",
                "message": "Image decoding timeout",
                "frame_id": frame_id
            })
            return
        except Exception as e:
            logger.error(f"Frame {frame_id}: Image decoding failed: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "message": f"Image decoding failed: {str(e)}",
                "frame_id": frame_id
            })
            return
        
        try:
            # Run OCR detection asynchronously with timeout
            detections, processing_time = await asyncio.wait_for(
                process_image_async(opencv_image, config.conf_threshold),
                timeout=10.0  # 10 second timeout for OCR
            )
            
            logger.debug(f"Frame {frame_id}: OCR completed, found {len(detections)} detections")
            
        except asyncio.TimeoutError:
            logger.error(f"Frame {frame_id}: OCR processing timeout")
            await websocket.send_json({
                "type": "error",
                "message": "OCR processing timeout",
                "frame_id": frame_id
            })
            return
        except Exception as e:
            logger.error(f"Frame {frame_id}: OCR processing failed: {str(e)}")
            # Send partial result with error
            await websocket.send_json({
                "type": "error",
                "message": f"OCR processing failed: {str(e)}",
                "frame_id": frame_id,
                "processing_time": time.time() - start_time
            })
            return
        
        # Update stats
        stats["frames_processed"] += 1
        stats["total_detections"] += len(detections)
        stats["total_processing_time"] += processing_time
        stats["frames_dropped"] = websocket_info["frame_buffer"].dropped_frames
        
        # Calculate additional metrics
        total_time = time.time() - start_time
        stream_duration = time.time() - stats["start_time"]
        
        # Send results back to frontend
        result = {
            "type": "detection_result",
            "stream_id": stream_id,
            "frame_id": frame_id,
            "client_timestamp": client_timestamp,
            "server_timestamp": time.time(),
            "processing_time": processing_time,
            "total_time": total_time,
            "detections": detections,
            "frame_info": {
                "width": opencv_image.shape[1],
                "height": opencv_image.shape[0],
                "channels": opencv_image.shape[2] if len(opencv_image.shape) > 2 else 1
            },
            "stats": {
                "frames_processed": stats["frames_processed"],
                "frames_dropped": stats["frames_dropped"],
                "total_detections": stats["total_detections"],
                "avg_processing_time": stats["total_processing_time"] / max(stats["frames_processed"], 1),
                "stream_duration": stream_duration,
                "fps": stats["frames_processed"] / max(stream_duration, 1)
            }
        }
        
        await websocket.send_json(result)
        logger.debug(f"Frame {frame_id}: Results sent successfully")
        
    except Exception as e:
        logger.error(f"Critical error processing frame for stream {stream_id}: {str(e)}")
        if stream_id in active_websockets:
            try:
                websocket = active_websockets[stream_id]["websocket"]
                await websocket.send_json({
                    "type": "critical_error",
                    "message": f"Critical frame processing error: {str(e)}",
                    "frame_id": frame_data.get("frame_id") if 'frame_data' in locals() else "unknown"
                })
            except Exception as send_error:
                logger.error(f"Failed to send error message: {str(send_error)}")

@app.get("/streams")
async def list_active_streams():
    """List all active WebSocket streams with performance metrics"""
    
    streams_info = []
    for stream_id, stream_data in active_websockets.items():
        stats = stream_data["stats"]
        duration = time.time() - stats["start_time"]
        
        streams_info.append({
            "stream_id": stream_id,
            "created_at": stream_data["created_at"],
            "config": stream_data["config"].dict(),
            "stats": {
                "frames_processed": stats["frames_processed"],
                "frames_dropped": stats["frames_dropped"],
                "total_detections": stats["total_detections"],
                "avg_processing_time": stats["total_processing_time"] / max(stats["frames_processed"], 1),
                "stream_duration": duration,
                "fps": stats["frames_processed"] / max(duration, 1),
                "drop_rate": stats["frames_dropped"] / max(stats["frames_processed"] + stats["frames_dropped"], 1)
            }
        })
    
    return {
        "active_streams": len(active_websockets),
        "max_workers": MAX_WORKERS,
        "frame_queue_size": FRAME_QUEUE_SIZE,
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
    
    # Clear frame buffer
    websocket_info["frame_buffer"].clear()
    del active_websockets[stream_id]
    
    return {
        "success": True,
        "message": f"Stream {stream_id} stopped and removed"
    }

@app.post("/ocr/config")
async def update_ocr_config(
    lang: str = "en", 
    use_gpu: bool = False,
    max_image_size: int = 1280,
    use_angle_cls: bool = False,
    det_db_thresh: float = 0.3,
    det_db_box_thresh: float = 0.6
):
    """Update OCR configuration with performance settings"""
    
    global ocr_processor
    
    try:
        config = StreamConfig(
            lang=lang,
            use_gpu=use_gpu,
            max_image_size=max_image_size,
            use_angle_cls=use_angle_cls,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh
        )
        
        ocr_processor = OptimizedOCRProcessor(lang=lang, use_gpu=use_gpu, config=config)
        
        return {
            "success": True,
            "message": "OCR configuration updated with optimizations",
            "config": {
                "language": lang,
                "gpu_enabled": use_gpu,
                "max_image_size": max_image_size,
                "use_angle_cls": use_angle_cls,
                "det_db_thresh": det_db_thresh,
                "det_db_box_thresh": det_db_box_thresh
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
        "current_language": ocr_processor.lang,
        "optimizations_enabled": True,
        "max_image_size": ocr_processor.max_image_size
    }

@app.get("/performance/stats")
async def get_performance_stats():
    """Get overall performance statistics"""
    
    total_streams = len(active_websockets)
    total_frames = sum(stream["stats"]["frames_processed"] for stream in active_websockets.values())
    total_dropped = sum(stream["stats"]["frames_dropped"] for stream in active_websockets.values())
    total_processing_time = sum(stream["stats"]["total_processing_time"] for stream in active_websockets.values())
    
    return {
        "system": {
            "max_workers": MAX_WORKERS,
            "frame_queue_size": FRAME_QUEUE_SIZE,
            "cpu_count": mp.cpu_count()
        },
        "performance": {
            "active_streams": total_streams,
            "total_frames_processed": total_frames,
            "total_frames_dropped": total_dropped,
            "avg_processing_time": total_processing_time / max(total_frames, 1),
            "drop_rate": total_dropped / max(total_frames + total_dropped, 1),
            "threads_active": THREAD_POOL._threads
        },
        "ocr_config": {
            "language": ocr_processor.lang,
            "gpu_enabled": ocr_processor.use_gpu,
            "max_image_size": ocr_processor.max_image_size
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Single worker for WebSocket support
        loop="uvloop" if 'uvloop' in globals() else "asyncio"  # Use uvloop if available
    )