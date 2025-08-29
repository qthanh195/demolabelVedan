import numpy as np
import cv2
import logging
from src.model.model import ModelYolo
import pytesseract
import re

custom_class_names_model_classifi = {
    0: "Label-10",
    1: "Label-11",
    2: "Label-12",
    3: "Label-13",
    4: "Label-14",
    5: "Label-15",
    6: "Label-16",
    7: "Label-17",
    8: "Label-18",
    9: "Label-19",
    10: "Label-1",
    11: "Label-20",
    12: "Label-21",
    13: "Label-22",
    14: "Label-23",
    15: "Label-24",
    16: "Label-25",
    17: "Label-26",
    18: "Label-27",
    19: "Label-28",
    20: "Label-29",
    21: "Label-2",
    22: "Group-04 (TDC)",
    23: "Label-31",
    24: "Label-32",
    25: "Label-33",
    26: "Group-03",
    27: "Label-35",
    28: "Group-04",
    29: "Label-37",
    30: "Label-38",
    31: "Label-39",
    32: "Label-3",
    33: "Label-40",
    34: "Label-41",
    35: "Label-42",
    36: "Label-43",
    37: "Label-44",
    38: "Group-02",
    39: "Label-46",
    40: "Group-01",
    41: "Label-48",
    42: "Label-4",
    43: "Label-5",
    44: "Label-6",
    45: "Label-7",
    46: "Label-8",
    47: "Label-9",
}

class AiHander(ModelYolo):
    def __init__(self):
        super().__init__()
        
    def detectLabel(self, image):
        results = self.model_segment_label.predict(image, conf=0.9, retina_masks=True)
        
        if not results or results[0].masks is None or len(results[0].masks.xy) == 0:
            print("No Label detected!")
            return None, None, 0.00, 0
        
        for _, result in enumerate(results):
            for i, seg in enumerate(result.masks.xy):
                
                polygon = np.array(seg, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(polygon)
                rect_label = ((x, y), (x + w, y + h))
                
                #cắt ảnh theo contour offset 20
                crop_image = self.crop_image_with_contour(image, polygon, offset_weight = 20, offset_height = 20)
                # crop_image = self.rotate_with_ocr(crop_image)
                # Tìm contours Áp dụng threshold
                gray_img = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=4)
                cv2.imwrite("threshold.jpg", thresh)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                filtered_contours = [c for c in contours if cv2.contourArea(c) < ((crop_image.shape[0]+10) * (crop_image.shape[1]+10))]

                if filtered_contours:
                    rect = cv2.minAreaRect(max(filtered_contours, key=cv2.contourArea))
                    area_infos = (cv2.contourArea(max(filtered_contours, key=cv2.contourArea)), (rect[1]), cv2.arcLength(max(filtered_contours, key=cv2.contourArea), True))
                    image_crop_cpo = crop_image.copy()
                    cv2.drawContours(image_crop_cpo, [max(filtered_contours, key=cv2.contourArea)], 0, (0,255,0), 3)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(image_crop_cpo,[box],0,(0,0,255),2)
                    cv2.imwrite("image_crop_cpo.jpg", image_crop_cpo)
                    
                    # return self.crop_image_with_contour(crop_image, max(filtered_contours, key=cv2.contourArea)), rect_label, result.boxes.conf[i].item(), area_infos
                    return self.rotate_with_ocr(self.crop_image_with_contour(crop_image, max(filtered_contours, key=cv2.contourArea))), rect_label, result.boxes.conf[i].item(), area_infos

                
        return None, None, 0.00, 0
        
    def classifiLabel(self, image):
        # kiểm tra có ảnh nhãn
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None, None, 0.0
        
        #Phân loại nhãn
        results = self.model_classifi_label.predict(image)
        
        #Kiểm tra nhãn có confidence >= 0.6
        if results[0].probs.top1conf.item() >= 0.6:
            id = results[0].probs.top1
            #return id, class_name, confidence
            return id, custom_class_names_model_classifi.get(id, results[0].names[id]), results[0].probs.top1conf.item()
        
        return None, None, 0.0

    def crop_image_with_contour(self, image, contour, offset_weight=0, offset_height=0):
        # Đảm bảo contour đúng định dạng
        contour = np.array(contour, dtype=np.int32)
        
        # 1. Tìm hình chữ nhật xoay bao quanh contour
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        size = tuple([int(s) for s in size])
        
        # 2. Tính toán kích thước ảnh sau khi xoay để không bị cắt
        h, w = image.shape[:2]
        
        # Tạo ma trận xoay với center gốc
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Tính toán kích thước ảnh mới sau khi xoay
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int((h * sin_a) + (w * cos_a))
        new_h = int((h * cos_a) + (w * sin_a))
        
        # Điều chỉnh ma trận xoay để đảm bảo toàn bộ ảnh được giữ lại
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 3. Xoay ảnh với kích thước mới
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        
        # 4. Tính tọa độ center mới sau khi xoay
        new_center_x = new_w / 2
        new_center_y = new_h / 2
        
        # 5. Crop vùng rectangle tại vị trí mới
        x = int(new_center_x - size[0] / 2)
        y = int(new_center_y - size[1] / 2)
        w, h = size
        
        # Đảm bảo không vượt quá biên ảnh
        x = max(0, x - offset_weight)
        y = max(0, y - offset_height)
        x_end = min(rotated.shape[1], x + w + 2 * offset_weight)
        y_end = min(rotated.shape[0], y + h + 2 * offset_height)
        
        return rotated[y:y_end, x:x_end]

    def rotate_with_ocr(self, image: np.ndarray) -> np.ndarray:
        image_osd = image.copy()
        image_osd = image_osd[:int(image.shape[0]/2), :]
        image_osd = cv2.resize(image_osd, (0,0), fx = 2, fy = 2, interpolation= cv2.INTER_LANCZOS4)
        try:
            osd_data = pytesseract.image_to_osd(image_osd, config='--oem 3 --psm 0')
            print("OSD output:", osd_data)
            match = re.search(r'(?<=Rotate: )\d+', osd_data)
            if match:
                angle = int(match.group(0))
                if angle == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception as e:
            print("Không thể chạy OSD:", e)
            if image.shape[0] < image.shape[1]:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        return image
        
        
# import numpy as np
# import cv2
# import logging
# from src.model.model import ModelYolo
# import pytesseract
# import re
# from typing import Optional, Tuple, Union, Dict, Any
# from functools import lru_cache
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import time
# from dataclasses import dataclass

# # Optimized class names mapping - sử dụng dict comprehension
# CUSTOM_CLASS_NAMES_MODEL_CLASSIFI = {
#     i: f"Label-{label_num}" for i, label_num in enumerate([
#         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
#         29, 2, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 3, 40, 41, 42, 43, 44, 45, 46, 
#         47, 48, 4, 5, 6, 7, 8, 9
#     ])
# }

# @dataclass
# class ProcessingResult:
#     """Data class cho kết quả xử lý"""
#     image: Optional[np.ndarray] = None
#     rect_label: Optional[Tuple] = None
#     confidence: float = 0.0
#     area: int = 0
#     class_id: Optional[int] = None
#     class_name: Optional[str] = None

# class AiHander(ModelYolo):
#     def __init__(self, enable_cache: bool = True, debug: bool = False, max_workers: int = 2):
#         """
#         Optimized AI Handler với multi-threading và caching
        
#         Args:
#             enable_cache: Bật cache cho các tính toán
#             debug: Bật debug mode
#             max_workers: Số luồng tối đa cho parallel processing
#         """
#         super().__init__()
        
#         # Configuration
#         self.enable_cache = enable_cache
#         self.debug = debug
#         self.max_workers = max_workers
#         self.classification_threshold = 0.6
        
#         # Setup logging
#         self.logger = self._setup_logger()
        
#         # Thread-safe caches
#         self._rotation_cache = {}
#         self._classification_cache = {}
#         self._cache_lock = threading.Lock()
        
#         # Tối ưu Tesseract config (giữ đơn giản như gốc)
#         self.detection_confidence = 0.9
        
#         # Pre-compiled regex cho performance (chỉ tối ưu regex)
#         self._rotation_pattern = re.compile(r'(?<=Rotate: )\d+')
        
#         # Performance tracking
#         self.stats = {
#             'total_detections': 0,
#             'successful_detections': 0,
#             'cache_hits': 0,
#             'processing_times': []
#         }
    
#     def _setup_logger(self) -> logging.Logger:
#         """Setup optimized logger"""
#         logger = logging.getLogger(self.__class__.__name__)
#         if self.debug and not logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#             handler.setFormatter(formatter)
#             logger.addHandler(handler)
#             logger.setLevel(logging.DEBUG)
#         return logger
    
#     def detectLabel(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple], float, int]:
#         """
#         Optimized label detection với early returns và parallel processing
        
#         Args:
#             image: Input image
            
#         Returns:
#             Tuple of (processed_image, rect_label, confidence, area)
#         """
#         start_time = time.time()
#         self.stats['total_detections'] += 1
        
#         try:
#             if not self._validate_input_image(image):
#                 return None, None, 0.0, 0
            
#             # Model prediction với error handling
#             results = self._predict_with_timeout(image)
            
#             if not self._has_valid_results(results):
#                 if self.debug:
#                     self.logger.debug("No valid label detected")
#                 return None, None, 0.0, 0
            
#             # Process results với optimization
#             result = self._process_detection_results(image, results[0])
            
#             if result.image is not None:
#                 self.stats['successful_detections'] += 1
            
#             return result.image, result.rect_label, result.confidence, result.area
            
#         except Exception as e:
#             if self.debug:
#                 self.logger.error(f"Error in detectLabel: {e}")
#             return None, None, 0.0, 0
        
#         finally:
#             processing_time = time.time() - start_time
#             self.stats['processing_times'].append(processing_time)
#             if self.debug:
#                 self.logger.debug(f"Detection processing time: {processing_time:.3f}s")
    
#     def _validate_input_image(self, image: np.ndarray) -> bool:
#         """Validate input image"""
#         return (image is not None and 
#                 isinstance(image, np.ndarray) and 
#                 image.size > 0 and 
#                 len(image.shape) in [2, 3])
    
#     def _predict_with_timeout(self, image: np.ndarray, timeout: float = 10.0):
#         """Model prediction với timeout protection"""
#         try:
#             # Sử dụng ThreadPoolExecutor để implement timeout
#             with ThreadPoolExecutor(max_workers=1) as executor:
#                 future = executor.submit(
#                     self.model_segment_label.predict, 
#                     image, 
#                     conf=self.detection_confidence, 
#                     retina_masks=True
#                 )
#                 return future.result(timeout=timeout)
#         except Exception as e:
#             if self.debug:
#                 self.logger.warning(f"Prediction timeout or error: {e}")
#             return None
    
#     def _has_valid_results(self, results) -> bool:
#         """Kiểm tra kết quả có hợp lệ không - optimized"""
#         return (results and 
#                 len(results) > 0 and
#                 results[0].masks is not None and 
#                 len(results[0].masks.xy) > 0)
    
#     def _process_detection_results(self, image: np.ndarray, result) -> ProcessingResult:
#         """
#         Xử lý kết quả detection với vectorized operations
#         """
#         masks = result.masks.xy
#         confidences = result.boxes.conf
        
#         # Vectorized area calculation
#         areas = np.array([cv2.contourArea(np.array(seg, dtype=np.int32)) 
#                          for seg in masks])
        
#         # Sort by area descending để xử lý mask lớn nhất trước
#         sorted_indices = np.argsort(areas)[::-1]
        
#         # Process masks in parallel nếu có nhiều masks
#         if len(sorted_indices) > 1 and self.max_workers > 1:
#             return self._process_masks_parallel(image, masks, confidences, sorted_indices)
#         else:
#             return self._process_masks_sequential(image, masks, confidences, sorted_indices)
    
#     def _process_masks_parallel(self, image: np.ndarray, masks, confidences, indices) -> ProcessingResult:
#         """Process multiple masks in parallel"""
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
#             for idx in indices[:self.max_workers]:  # Giới hạn số masks xử lý
#                 future = executor.submit(
#                     self._process_single_mask, 
#                     image, masks[idx], confidences[idx].item()
#                 )
#                 futures.append(future)
            
#             # Return first successful result
#             for future in futures:
#                 try:
#                     result = future.result(timeout=5.0)
#                     if result.image is not None:
#                         return result
#                 except Exception as e:
#                     if self.debug:
#                         self.logger.warning(f"Parallel mask processing error: {e}")
        
#         return ProcessingResult()
    
#     def _process_masks_sequential(self, image: np.ndarray, masks, confidences, indices) -> ProcessingResult:
#         """Process masks sequentially"""
#         for idx in indices:
#             result = self._process_single_mask(image, masks[idx], confidences[idx].item())
#             if result.image is not None:
#                 return result
#         return ProcessingResult()
    
#     def _process_single_mask(self, image: np.ndarray, seg, confidence: float) -> ProcessingResult:
#         """
#         Xử lý một mask đơn lẻ - highly optimized
#         """
#         try:
#             # Convert segment to polygon
#             polygon = np.array(seg, dtype=np.int32)
            
#             # Quick validation
#             if len(polygon) < 3:
#                 return ProcessingResult()
            
#             # Bounding rectangle
#             x, y, w, h = cv2.boundingRect(polygon)
#             rect_label = ((x, y), (x + w, y + h))
            
#             # Crop image với optimized method
#             crop_image = self.crop_image_with_contour_optimized(
#                 image, polygon, offset_weight=20, offset_height=20
#             )
            
#             if crop_image is None or crop_image.size == 0:
#                 return ProcessingResult()
            
#             # Process threshold với caching
#             final_crop, area = self._process_threshold_optimized(crop_image)
            
#             if final_crop is not None and area > 0:
#                 # Rotate với OCR - optimized
#                 rotated_crop = self.rotate_with_ocr_optimized(final_crop)
                
#                 return ProcessingResult(
#                     image=rotated_crop,
#                     rect_label=rect_label,
#                     confidence=confidence,
#                     area=area
#                 )
            
#         except Exception as e:
#             if self.debug:
#                 self.logger.error(f"Error processing mask: {e}")
        
#         return ProcessingResult()
    
#     def _process_threshold_optimized(self, crop_image: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
#         """
#         Highly optimized threshold processing
#         """
#         try:
#             # Convert to grayscale efficiently
#             gray_img = (cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY) 
#                        if len(crop_image.shape) == 3 else crop_image)
            
#             # Optimized threshold với adaptive method
#             _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
#             # Minimal morphological operations
#             kernel = np.ones((2, 2), np.uint8)
#             thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            
#             # Find contours với minimal approximation
#             contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             if not contours:
#                 return None, 0
            
#             # Vectorized filtering
#             max_area = (crop_image.shape[0] + 10) * (crop_image.shape[1] + 10)
#             valid_contours = [c for c in contours if cv2.contourArea(c) < max_area]
            
#             if not valid_contours:
#                 return None, 0
            
#             # Get largest contour
#             largest_contour = max(valid_contours, key=cv2.contourArea)
#             area = int(cv2.contourArea(largest_contour))
            
#             # Final crop
#             final_crop = self.crop_image_with_contour_optimized(crop_image, largest_contour)
            
#             return final_crop, area
            
#         except Exception as e:
#             if self.debug:
#                 self.logger.error(f"Error in threshold processing: {e}")
#             return None, 0
    
#     def classifiLabel(self, image: np.ndarray) -> Tuple[Optional[int], Optional[str], Optional[float]]:
#         """
#         Optimized label classification với caching
        
#         Args:
#             image: Input image
            
#         Returns:
#             Tuple of (id, class_name, confidence)
#         """
#         if not self._validate_input_image(image):
#             return None, None, None
        
#         try:
#             # Generate cache key nếu enable_cache
#             cache_key = None
#             if self.enable_cache:
#                 # Simple hash based on image properties
#                 cache_key = hash((image.shape, image.mean(), image.std()))
                
#                 with self._cache_lock:
#                     if cache_key in self._classification_cache:
#                         self.stats['cache_hits'] += 1
#                         if self.debug:
#                             self.logger.debug("Classification cache hit")
#                         return self._classification_cache[cache_key]
            
#             # Model prediction
#             results = self.model_classifi_label.predict(image)
            
#             if not results or len(results) == 0:
#                 return None, None, None
            
#             confidence = results[0].probs.top1conf.item()
            
#             if confidence >= self.classification_threshold:
#                 class_id = results[0].probs.top1
#                 class_name = CUSTOM_CLASS_NAMES_MODEL_CLASSIFI.get(
#                     class_id, results[0].names[class_id]
#                 )
                
#                 result = (class_id, class_name, confidence)
                
#                 # Cache result
#                 if cache_key is not None:
#                     with self._cache_lock:
#                         self._classification_cache[cache_key] = result
                
#                 if self.debug:
#                     self.logger.debug(f"Classified label: {class_name} with confidence: {confidence:.3f}")
                
#                 return result
            
#             return None, None, confidence
            
#         except Exception as e:
#             if self.debug:
#                 self.logger.error(f"Error in classifiLabel: {e}")
#             return None, None, None
    
#     def crop_image_with_contour_optimized(self, image: np.ndarray, contour: np.ndarray, 
#                                         offset_weight: int = 0, offset_height: int = 0) -> Optional[np.ndarray]:
#         """
#         Highly optimized crop với rotation caching và vectorized operations
#         """
#         try:
#             if not self._validate_input_image(image):
#                 return None
            
#             contour = np.array(contour, dtype=np.int32)
#             if len(contour) < 3:
#                 return None
            
#             # Minimum area rectangle
#             rect = cv2.minAreaRect(contour)
#             center, size, angle = rect
            
#             if size[0] <= 0 or size[1] <= 0:
#                 return None
            
#             size = tuple([int(s) for s in size])
#             h, w = image.shape[:2]
            
#             # Cache key for rotation matrix
#             cache_key = None
#             if self.enable_cache:
#                 cache_key = (tuple(center), angle, h, w)
                
#                 with self._cache_lock:
#                     if cache_key in self._rotation_cache:
#                         M, new_w, new_h = self._rotation_cache[cache_key]
#                         self.stats['cache_hits'] += 1
#                     else:
#                         M, new_w, new_h = self._compute_rotation_matrix(center, angle, h, w)
#                         self._rotation_cache[cache_key] = (M, new_w, new_h)
#             else:
#                 M, new_w, new_h = self._compute_rotation_matrix(center, angle, h, w)
            
#             # Warp affine với optimized interpolation
#             rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
            
#             # Vectorized crop coordinates calculation
#             new_center = np.array([new_w / 2, new_h / 2])
#             crop_coords = np.array([
#                 new_center[0] - size[0] / 2,
#                 new_center[1] - size[1] / 2
#             ], dtype=int)
            
#             # Apply offsets với boundary check
#             x = max(0, crop_coords[0] - offset_weight)
#             y = max(0, crop_coords[1] - offset_height)
#             x_end = min(rotated.shape[1], crop_coords[0] + size[0] + offset_weight)
#             y_end = min(rotated.shape[0], crop_coords[1] + size[1] + offset_height)
            
#             if x >= x_end or y >= y_end:
#                 return None
            
#             return rotated[y:y_end, x:x_end]
            
#         except Exception as e:
#             if self.debug:
#                 self.logger.error(f"Error in crop_image_with_contour: {e}")
#             return None
    
#     def _compute_rotation_matrix(self, center, angle, h, w):
#         """Compute rotation matrix và new dimensions"""
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
#         # Vectorized calculation
#         cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
#         new_w = int((h * sin_a) + (w * cos_a))
#         new_h = int((h * cos_a) + (w * sin_a))
        
#         # Adjust translation
#         M[0, 2] += (new_w / 2) - center[0]
#         M[1, 2] += (new_h / 2) - center[1]
        
#         return M, new_w, new_h
    
#     def rotate_with_ocr_optimized(self, image: np.ndarray) -> np.ndarray:
#         """
#         Optimized rotation giữ nguyên logic gốc của bạn
#         """
#         if not self._validate_input_image(image) or image.size == 0:
#             return image
        
#         try:
#             # Giữ nguyên logic OCR gốc của bạn
#             osd_data = pytesseract.image_to_osd(image, config='--oem 3 --psm 0')
            
#             if self.debug:
#                 self.logger.debug(f"OSD output: {osd_data}")
            
#             # Sử dụng regex pattern đã compile để tối ưu
#             match = self._rotation_pattern.search(osd_data)
            
#             if match:
#                 angle = int(match.group(0))
                
#                 # Giữ nguyên logic xoay gốc của bạn
#                 if angle == 90:
#                     image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#                 elif angle == 180:
#                     image = cv2.rotate(image, cv2.ROTATE_180)
#                 elif angle == 270:
#                     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
#         except Exception as e:
#             if self.debug:
#                 self.logger.warning(f"Không thể chạy OSD: {e}")
            
#             # Giữ nguyên fallback logic gốc của bạn
#             if image.shape[0] < image.shape[1]:
#                 image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
#         return image
    

    
#     # Performance monitoring và management methods
#     def clear_cache(self):
#         """Clear all caches"""
#         with self._cache_lock:
#             self._rotation_cache.clear()
#             self._classification_cache.clear()
#         self._extract_rotation_angle.cache_clear()
        
#         if self.debug:
#             self.logger.info("All caches cleared")
    
#     def get_performance_stats(self) -> Dict[str, Any]:
#         """Get comprehensive performance statistics"""
#         avg_time = (np.mean(self.stats['processing_times']) 
#                    if self.stats['processing_times'] else 0)
        
#         return {
#             'total_detections': self.stats['total_detections'],
#             'successful_detections': self.stats['successful_detections'],
#             'success_rate': (self.stats['successful_detections'] / max(1, self.stats['total_detections'])),
#             'cache_hits': self.stats['cache_hits'],
#             'average_processing_time': avg_time,
#             'rotation_cache_size': len(self._rotation_cache),
#             'classification_cache_size': len(self._classification_cache),
#             'angle_cache_info': self._extract_rotation_angle.cache_info()._asdict()
#         }
    
#     def optimize_for_batch_processing(self):
#         """Optimize settings for batch processing"""
#         self.max_workers = min(4, self.max_workers * 2)
#         self.classification_threshold = 0.7  # Slightly higher for batch
        
#         if self.debug:
#             self.logger.info(f"Optimized for batch processing: max_workers={self.max_workers}")
    
#     def reset_stats(self):
#         """Reset performance statistics"""
#         self.stats = {
#             'total_detections': 0,
#             'successful_detections': 0,
#             'cache_hits': 0,
#             'processing_times': []
#         }

# # # Convenience functions
# # def create_optimized_handler(debug: bool = False, max_workers: int = 2) -> OptimizedAiHandler:
# #     """Factory function để tạo optimized handler"""
# #     return OptimizedAiHandler(enable_cache=True, debug=debug, max_workers=max_workers)

# # # Example usage và benchmarking
# # if __name__ == "__main__":
# #     import time
    
# #     # Create optimized handler
# #     handler = create_optimized_handler(debug=True, max_workers=4)
    
# #     # Performance test với mock data
# #     test_image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    
# #     print("Starting performance test...")
# #     start_time = time.time()
    
# #     # Test detection
# #     for i in range(10):
# #         result = handler.detectLabel(test_image)
# #         print(f"Detection {i+1}: Success={result[0] is not None}")
    
# #     # Test classification
# #     for i in range(10):
# #         class_result = handler.classifiLabel(test_image)
# #         print(f"Classification {i+1}: {class_result[1] if class_result[1] else 'None'}")
    
# #     total_time = time.time() - start_time
    
# #     # Performance report
# #     stats = handler.get_performance_stats()
# #     print(f"\n{'='*50}")
# #     print("PERFORMANCE REPORT:")
# #     print(f"{'='*50}")
# #     print(f"Total processing time: {total_time:.3f}s")
# #     print(f"Average per operation: {total_time/20:.3f}s")
# #     print(f"Success rate: {stats['success_rate']:.2%}")
# #     print(f"Cache hits: {stats['cache_hits']}")
# #     print(f"Cache sizes - Rotation: {stats['rotation_cache_size']}, Classification: {stats['classification_cache_size']}")
# #     print(f"Average processing time: {stats['average_processing_time']:.3f}s")