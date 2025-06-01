import cv2
import time
import numpy as np
from pathlib import Path

try:
    from paddleocr import PaddleOCR
except ImportError as e:
    raise ImportError(
        "PaddleOCR not found. Install with: pip install paddleocr paddlepaddle-cpu (or paddlepaddle-gpu for GPU support)"
    ) from e

try:
    from PIL import Image, ImageDraw, ImageFont # Still needed for process_image_file
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class PaddleTextPrinter:
    """
    Real-time scene-text recogniser that streams video with detected text.
    Also supports processing single image files.
    """

    def __init__(
        self,
        camera_index: int | str = 0, # Can be int for camera or str for video file
        lang: str = "en",
        use_gpu: bool = False,
        detection_interval: int = 3,  # OCR every Nth frame.
        conf_threshold: float = 0.5,
        suppress_duplicates_console: bool = True,
        draw_font_size_pil: int = 20,
        # OpenCV drawing parameters
        cv_font_scale: float = 0.6,
        cv_font_thickness: int = 1,
        cv_box_color: tuple = (0, 255, 0),   # Green for boxes
        cv_text_color: tuple = (255, 0, 0), # Blue for text (BGR format)
        **paddle_ocr_kwargs
    ) -> None:
        self.cap: cv2.VideoCapture | None = None
        self.camera_index = camera_index
        
        if 'use_angle_cls' not in paddle_ocr_kwargs:
            paddle_ocr_kwargs['use_angle_cls'] = True
        if 'show_log' not in paddle_ocr_kwargs:
            paddle_ocr_kwargs['show_log'] = False

        self.ocr = PaddleOCR(lang=lang)
        
        self.detection_interval = max(1, detection_interval)
        self.conf_threshold = conf_threshold
        self.suppress_duplicates_console = suppress_duplicates_console
        self.last_printed_console: set[str] = set()
        
        self.draw_font_size_pil = draw_font_size_pil
        self.cv_font = cv2.FONT_HERSHEY_SIMPLEX
        self.cv_font_scale = cv_font_scale
        self.cv_font_thickness = cv_font_thickness
        self.cv_box_color = cv_box_color
        self.cv_text_color = cv_text_color

        self.last_ocr_results_for_display: list = [] # Store {'box': ..., 'text': ...}

    def _parse_ocr_results(self, ocr_result_list_from_paddle):
        if not ocr_result_list_from_paddle: return None, None, None
        ocr_data_for_first_image = ocr_result_list_from_paddle[0]
        if ocr_data_for_first_image is None: return None, None, None

        all_boxes, all_texts, all_confidences = [], [], []

        if isinstance(ocr_data_for_first_image, dict):
            _texts = ocr_data_for_first_image.get('rec_texts', [])
            _scores = ocr_data_for_first_image.get('rec_scores', [])
            _boxes = ocr_data_for_first_image.get('rec_polys') or ocr_data_for_first_image.get('dt_polys', [])

            valid_indices = [i for i, (t, s, b) in enumerate(zip(_texts, _scores, _boxes)) if t and s and b]
            all_texts = [_texts[i] for i in valid_indices]
            all_confidences = [float(_scores[i]) for i in valid_indices]
            all_boxes = [_boxes[i] for i in valid_indices]
        
        elif isinstance(ocr_data_for_first_image, list):
            for item in ocr_data_for_first_image:
                if isinstance(item, list) and len(item) == 2:
                    box_coords, text_info = item[0], item[1]
                    if isinstance(text_info, (tuple, list)) and len(text_info) == 2:
                        text, confidence = text_info
                        if text and confidence is not None and box_coords:
                            all_boxes.append(box_coords)
                            all_texts.append(text)
                            all_confidences.append(float(confidence))
        return all_boxes, all_texts, all_confidences

    def _start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video source: {self.camera_index}")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = self.cap.get(cv2.CAP_PROP_FPS)
        source_type = "Camera" if isinstance(self.camera_index, int) else "Video file"
        print(f"{source_type} '{self.camera_index}' started. Resolution: {width}x{height} @ {fps_cam:.2f} FPS (reported). Press 'q' to quit.\n")

    def _draw_ocr_on_frame_cv(self, frame, ocr_results_to_draw):
        """Draws OCR results on the frame using OpenCV."""
        for detection in ocr_results_to_draw:
            box = np.array(detection['box']).astype(np.int32)
            text = detection['text']
            
            cv2.polylines(frame, [box], isClosed=True, color=self.cv_box_color, thickness=self.cv_font_thickness)
            
            text_x = box[:, 0].min()
            text_y = box[:, 1].min() - 5 
            if text_y < 10: text_y = box[:, 1].min() + self.cv_font_thickness + 10 
            
            cv2.putText(frame, text, (text_x, text_y), self.cv_font, 
                        self.cv_font_scale, self.cv_text_color, self.cv_font_thickness, cv2.LINE_AA)
        return frame

    def run_camera_streaming(self):
        self._start_camera()
        frame_id = 0
        ocr_cycle_count = 0
        
        # For FPS calculation of the main loop (display FPS)
        loop_start_time = time.time()
        loop_frame_count = 0

        # For FPS calculation of OCR processing cycles
        ocr_processing_start_time = time.time()
        ocr_processed_frame_count = 0

        cv2.namedWindow("PaddleOCR Real-time Stream", cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream or camera read failed.")
                    break

                display_frame = frame.copy()

                if frame_id % self.detection_interval == 0:
                    ocr_t_start = time.time()
                    ocr_result_list = self.ocr.predict(frame)
                    ocr_t_taken = time.time() - ocr_t_start
                    ocr_processed_frame_count += 1
                    
                    all_boxes, all_texts, all_confidences = self._parse_ocr_results(ocr_result_list)
                    
                    current_ocr_display_data = [] # Data for this specific OCR pass
                    new_texts_for_console = []

                    if all_texts:
                        for i in range(len(all_texts)):
                            text, confidence, box = all_texts[i], all_confidences[i], all_boxes[i]
                            if confidence >= self.conf_threshold:
                                current_ocr_display_data.append({'box': box, 'text': text, 'confidence': confidence})
                                if not self.suppress_duplicates_console or text not in self.last_printed_console:
                                    new_texts_for_console.append((text, confidence))
                        
                        if new_texts_for_console:
                            timestamp = time.strftime("%H:%M:%S")
                            print(f"-- OCR @ {timestamp} (Frame {frame_id}, OCR time: {ocr_t_taken:.3f}s) --")
                            for txt, conf in new_texts_for_console:
                                print(f"  {txt}  ({conf*100:.1f}%)")
                            if self.suppress_duplicates_console:
                                self.last_printed_console.update(txt for txt, _ in new_texts_for_console)
                                if len(self.last_printed_console) > 200: self.last_printed_console.clear()
                    
                    # Update the persistent display list only if new OCR data was found
                    if current_ocr_display_data:
                        self.last_ocr_results_for_display = current_ocr_display_data
                    elif not all_texts: # No text detected in this OCR cycle, clear old drawings
                        self.last_ocr_results_for_display.clear()


                # Draw the latest valid OCR results on the display_frame
                if self.last_ocr_results_for_display:
                    display_frame = self._draw_ocr_on_frame_cv(display_frame, self.last_ocr_results_for_display)

                loop_frame_count += 1
                # Calculate and display Display FPS
                if loop_frame_count >= 10: # Update every 10 frames
                    current_time = time.time()
                    display_fps = loop_frame_count / (current_time - loop_start_time)
                    cv2.putText(display_frame, f"Display FPS: {display_fps:.1f}", (10, 30), 
                                self.cv_font, 0.7, (0,0,255), 2, cv2.LINE_AA) # Red for FPS
                    loop_frame_count = 0
                    loop_start_time = current_time
                
                # Calculate and display OCR Processing FPS
                if ocr_processed_frame_count >= 5: # Update after 5 OCR cycles
                    current_time = time.time()
                    ocr_fps = ocr_processed_frame_count / (current_time - ocr_processing_start_time)
                    cv2.putText(display_frame, f"OCR FPS: {ocr_fps:.1f} (Interval: {self.detection_interval})", (10, 60), 
                                self.cv_font, 0.7, (0,0,255), 2, cv2.LINE_AA)
                    ocr_processed_frame_count = 0
                    ocr_processing_start_time = current_time
                
                cv2.imshow("PaddleOCR Real-time Stream", display_frame)
                
                frame_id += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
        
        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C).")
        finally:
            self._cleanup()

    def process_image_file( # This method remains for single image processing with PIL
        self,
        image_path: str | Path,
        output_path: str | Path | None = None
    ) -> None:
        image_path_obj = Path(image_path)
        if output_path and not PIL_AVAILABLE:
            print("Error: Pillow (PIL) is not installed for saving annotated image.")
            return

        if not image_path_obj.is_file():
            print(f"Error: Image not found: {image_path_obj}")
            return

        img_cv = cv2.imread(str(image_path_obj))
        if img_cv is None:
            print(f"Error: Could not read image: {image_path_obj}")
            return

        print(f"Processing image: {image_path_obj.name}...")
        ocr_result_list = self.ocr.ocr(img_cv, cls=self.ocr.use_angle_cls)
        all_boxes, all_texts, all_confidences = self._parse_ocr_results(ocr_result_list)

        if not all_texts:
            print(f"No text reliably extracted from {image_path_obj.name}.")
            return

        texts_to_print, detections_to_draw = [], []
        for i in range(len(all_texts)):
            text, confidence, box = all_texts[i], all_confidences[i], all_boxes[i]
            if confidence >= self.conf_threshold:
                detections_to_draw.append({'box': box, 'text': text, 'confidence': confidence})
                if not self.suppress_duplicates_console or text not in self.last_printed_console:
                    texts_to_print.append((text, confidence))

        if texts_to_print:
            print(f"--- Detected text (for console) ---")
            for txt, conf in texts_to_print: print(f"{txt} ({conf*100:.1f}%)")
            if self.suppress_duplicates_console:
                self.last_printed_console.update(txt for txt, _ in texts_to_print)
                if len(self.last_printed_console) > 500: self.last_printed_console.clear()
        
        if output_path and detections_to_draw:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            pil_img = Image.fromarray(cv2.cvtColor(img_cv.copy(), cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            try: font = ImageFont.truetype(str(self.ocr.font_path), self.draw_font_size_pil)
            except: font = ImageFont.load_default()
            
            for det in detections_to_draw:
                box_np = np.array(det['box']).astype(np.int32)
                draw.polygon([tuple(p) for p in box_np], outline="red", width=2)
                tx, ty = int(box_np[:, 0].min()), int(box_np[:, 1].min()) - self.draw_font_size_pil - 2
                if ty < 0: ty = int(box_np[:, 1].min()) + 2
                if tx < 0: tx = 0
                draw.text((tx, ty), det['text'], fill="green", font=font)
            try:
                pil_img.save(str(output_path_obj))
                print(f"Annotated image saved to: {output_path_obj}")
            except Exception as e: print(f"Error saving image: {e}")
        elif output_path: print(f"No text met threshold for drawing. Output image not saved.")

    def _cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

# ------------------------------- ENTRY ---------------------------------- #
def main():
    # --- Option 1: Run real-time camera streaming with OCR ---
    print("Starting real-time camera streaming with OCR...")
    streamer = PaddleTextPrinter(
        camera_index=0, # Use 0 for webcam, or "path/to/your/video.mp4"
        lang="en",      # Or your desired language e.g., "ch", "korean", "japan"
        use_gpu=False,  # Set to True if you have a compatible GPU and paddlepaddle-gpu
        detection_interval=5, # OCR every Nth frame. Adjust for performance vs. responsiveness.
                              # Lower value = more responsive OCR but lower display FPS.
                              # Higher value = smoother display FPS but less frequent OCR updates.
        conf_threshold=0.6,   # Confidence threshold for displaying detections
        cv_font_scale=0.5,
        cv_box_color=(0, 200, 0), # Slightly darker green
        cv_text_color=(200, 50, 50), # Slightly darker blue
        # Example of passing other PaddleOCR arguments:
        # paddle_ocr_kwargs={'det_limit_side_len': 736}
    )
    streamer.run_camera_streaming()

    # --- Option 2: Process a single image file (example) ---
    # print("\nStarting single image text detection...")
    # # Replace with your image path:
    # image_file_path = Path("D:\\Scene_text_detection_realtime\\screenshot_1748712581.jpg")
    # output_annotated_image = Path("output_annotated_image.png")

    # if not image_file_path.exists():
    #     print(f"Test image not found: {image_file_path}")
    # else:
    #     if not PIL_AVAILABLE and output_annotated_image:
    #         print("Pillow (PIL) is not installed. Cannot save annotated image.")
    #     else:
    #         image_processor = PaddleTextPrinter(lang="en", use_gpu=False, conf_threshold=0.5)
    #         image_processor.process_image_file(
    #             image_path=image_file_path,
    #             output_path=output_annotated_image
    #         )
    #     print("Single image processing finished.")

if __name__ == "__main__":
    main()