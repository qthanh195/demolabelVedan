import cv2
import numpy as np
import base64
import logging
import time
from src.function.ocr import OCR_Engine
import cv2
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict
import os
import threading
from PIL import Image
import json

class ProcessImage(OCR_Engine):
    def __init__(self):
        super().__init__()

    def image_to_base64(self, image_np: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image_np)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def handle_special_labels(self, id, label_image, height, width):
        logging.debug(f"Xử lý nhãn đặc biệt: id={id}")
        
        match id:
            case 22:  # tdc
                return self.classifi_tdc_with_ocr(label_image, height, width)
            case 40:  # recycling
                return self.classify_label_logo_recycling(label_image, height, width)
            case 38:  # halal
                return self.classify_label_logo_halal(label_image, height, width)
            case 26:  # unu
                return self.classify_label_logo_unu(label_image, height, width)
            case _:
                return "", 0, "", ""
    
    def get_image_size(self, image_name = "", json_file = "data\label_sizes.json"):
        # Đọc file JSON
        with open(json_file, "r", encoding="utf-8") as f:
            image_info = json.load(f)

        # Tìm theo tên ảnh
        if image_name in image_info:
            size = image_info[image_name]
            width, height = size["width"], size["height"]
            return width, height
        else:
            print(f"❌ Không tìm thấy {image_name} trong {json_file}")
            return None
    
    def estimate_original_area(self, area_new: float, lost_percent: float) -> float:
        """
        Ước tính diện tích gốc (100%) dựa trên diện tích còn lại và phần trăm đã mất.

        :param area_new: diện tích hiện tại (sau khi mất một phần).
        :param lost_percent: phần trăm diện tích đã mất (0–100).
        :return: diện tích gốc (100%).
        """
        if lost_percent >= 100:
            raise ValueError("lost_percent phải nhỏ hơn 100")
        if lost_percent < 0:
            raise ValueError("lost_percent không được âm")

        original_area = area_new / (1 - lost_percent / 100)
        return original_area

    # def evaluate_label(original_size, contour):
    #     """
    #     Đánh giá % nhãn còn lại so với nhãn gốc dựa trên contour tìm được.

    #     Args:
    #         original_size (tuple): (original_width, original_height) kích thước nhãn gốc
    #         contour (ndarray): contour tìm được từ cv2.findContours

    #     Returns:
    #         float: phần trăm diện tích nhãn còn lại (%)
    #     """
    #     original_width, original_height = original_size

    #     # diện tích thực tế của contour
    #     rect = cv2.minAreaRect(contour)
    #     w, h = rect[1]

    #     #sort
    #     if 

    #     # tỉ lệ dài/rộng gốc và mới
    #     orig_ratio = original_width / original_height
    #     new_ratio = w / h if h != 0 else 1e-6

    #     # độ méo hình (%)
    #     shape_ratio = abs(new_ratio / orig_ratio - 1) * 100

    #     # ước lượng diện tích gốc dựa trên bounding rect + hiệu chỉnh méo hình
    #     est_original_area = w * h * (1 + shape_ratio / 100)

    #     # % nhãn còn lại
    #     percent_remain = (label_area / est_original_area) * 100

    #     return percent_remain

    
    def full_ocr(self, image):
        pass
    
    def read_netweight(self, image):
        weight = ""
        results = self.model_detect_khoiluong_tdc.predict(source=image)
        if results is None or len(results) == 0 or results[0].boxes is None:
            return False, None
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                image_crop = image[y1-4:y2+4, x1:x2]
                weight = pytesseract.image_to_string(image_crop, config=r'--oem 3 --psm 8 -l eng')
                
                if weight != "":
                    return weight
        return weight  
    
class SmartOCR:
    def __init__(self, tesseract_path=None, max_workers=4):
        """
        Khởi tạo SmartOCR
        
        Args:
            tesseract_path: Đường dẫn đến tesseract executable (Windows)
            max_workers: Số luồng xử lý song song
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.max_workers = max_workers
        self.results_lock = threading.Lock()
        
    def detect_text_regions(self, image: np.ndarray, min_height=10, debug=False) -> List[Tuple[int, int]]:
        """
        Phát hiện các vùng có text bằng contours và nhóm theo hàng
        
        Args:
            image: Ảnh đầu vào (numpy array)
            min_height: Chiều cao tối thiểu của text line
            debug: Có lưu ảnh debug không
            
        Returns:
            List các tuple (y_start, y_end) của các vùng text theo hàng
        """
        # Chuyển sang ảnh xám
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Tiền xử lý theo đoạn code của bạn
        gray = cv2.medianBlur(gray, 3)
        
        # Áp dụng threshold để tạo ảnh binary
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        binary = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations=5)
        
        cv2.imwrite("binary.jpg", binary)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Len Contours:", len(contours) )
        # Lấy bounding boxes
        text_boxes = []
        i = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Lọc các box quá nhỏ
            if 79 > h > 8:
                text_boxes.append((x, y, x + w, y + h))
                # i+=1
                # cv2.imwrite(f"region_text/image_{i}.jpg", image[y:y+h, x:x+w])   
        
        if not text_boxes:
            return []
        
        # Nhóm các boxes theo hàng
        text_lines = self.group_boxes_by_lines(text_boxes)
        
        # Chuyển đổi thành format (y_start, y_end)
        text_regions = []
        
        for line_boxes in text_lines:
            # Lấy y_min và y_max của cả dòng
            y_coords = [box[1] for box in line_boxes] + [box[3] for box in line_boxes]
            x_coords = [box[0] for box in line_boxes] + [box[2] for box in line_boxes]
            # text_regions.append(min(y_coords), max(y_coords),min(x_coords),max(x_coords))
            y_min, y_max = min(y_coords), max(y_coords)
            x_min, x_max = min(x_coords), max(x_coords)
            
            # Sửa syntax - thêm dấu ngoặc vuông
            text_regions.append((y_min, y_max))
            
            i+=1
            cv2.imwrite(f"region_text/image_{i}.jpg", image[min(y_coords):max(y_coords), min(x_coords):max(x_coords)])
        
        # Sắp xếp theo thứ tự từ trên xuống
        text_regions.sort(key=lambda x: x[0])
        # self.visualize_text_regions(image, text_regions, text_boxes)
        
        return text_regions
    
    def group_boxes_by_lines(self, boxes: List[Tuple[int, int, int, int]], 
                            line_threshold=10) -> List[List[Tuple[int, int, int, int]]]:
        """
        Nhóm các bounding boxes thành các dòng text
        
        Args:
            boxes: List các bounding box (x1, y1, x2, y2)
            line_threshold: Khoảng cách tối đa giữa các box trong cùng 1 dòng
            
        Returns:
            List các dòng, mỗi dòng chứa list các boxes
        """
        if not boxes:
            return []
        
        # Sắp xếp boxes theo y coordinate
        sorted_boxes = sorted(boxes, key=lambda box: box[1])
        
        lines = []
        current_line = [sorted_boxes[0]]
        current_line_y = (sorted_boxes[0][1] + sorted_boxes[0][3]) // 2
        
        for box in sorted_boxes[1:]:
            box_center_y = (box[1] + box[3]) // 2
            
            # Kiểm tra xem box có thuộc dòng hiện tại không
            if abs(box_center_y - current_line_y) <= line_threshold:
                current_line.append(box)
            else:
                # Bắt đầu dòng mới
                if current_line:
                    # Sắp xếp boxes trong dòng theo x coordinate
                    current_line.sort(key=lambda b: b[0])
                    lines.append(current_line)
                
                current_line = [box]
                current_line_y = box_center_y
        
        # Thêm dòng cuối cùng
        if current_line:
            current_line.sort(key=lambda b: b[0])
            lines.append(current_line)
        
        return lines
    
    def visualize_text_regions(self, image: np.ndarray, text_regions: List[Tuple[int, int]], 
                              text_boxes: List[Tuple[int, int, int, int]]):
        """
        Vẽ và lưu ảnh debug với các vùng text được highlight
        """
        debug_image = image.copy()
        
        # Vẽ các text boxes
        for i, (x1, y1, x2, y2) in enumerate(text_boxes):
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Vẽ các text regions (theo hàng)
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, (y_start, y_end) in enumerate(text_regions):
            color = colors[i % len(colors)]
            cv2.line(debug_image, (0, y_start), (image.shape[1], y_start), color, 2)
            cv2.line(debug_image, (0, y_end), (image.shape[1], y_end), color, 2)
            
            # Thêm text label
            cv2.putText(debug_image, f'Line {i}', (10, y_start - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite("debug_text_regions.jpg", debug_image)
    
    def merge_close_regions(self, regions: List[Tuple[int, int]], 
                          max_gap=15, max_region_height=300) -> List[Tuple[int, int]]:
        """
        Gộp các vùng text gần nhau và chia các vùng quá lớn
        
        Args:
            regions: List các vùng text
            max_gap: Khoảng cách tối đa để gộp 2 vùng
            max_region_height: Chiều cao tối đa của 1 vùng
            
        Returns:
            List các vùng sau khi gộp và chia
        """
        if not regions:
            return []
            
        merged_regions = []
        current_start, current_end = regions[0]
        
        for start, end in regions[1:]:
            # Nếu khoảng cách nhỏ, gộp lại
            if start - current_end <= max_gap:
                current_end = end
            else:
                merged_regions.append((current_start, current_end))
                current_start, current_end = start, end
                
        merged_regions.append((current_start, current_end))
        
        # Chia các vùng quá lớn
        final_regions = []
        for start, end in merged_regions:
            height = end - start
            if height <= max_region_height:
                final_regions.append((start, end))
            else:
                # Chia thành các phần nhỏ hơn
                num_parts = int(np.ceil(height / max_region_height))
                part_height = height / num_parts
                
                for i in range(num_parts):
                    part_start = int(start + i * part_height)
                    part_end = int(start + (i + 1) * part_height)
                    if i == num_parts - 1:  # Phần cuối cùng
                        part_end = end
                    final_regions.append((part_start, part_end))
                    
        return final_regions
    
    def create_smart_chunks(self, image: np.ndarray, 
                          target_chunks=None) -> List[Tuple[np.ndarray, int, int]]:
        """
        Chia ảnh thành các chunks thông minh dựa trên vùng text
        
        Args:
            image: Ảnh đầu vào
            target_chunks: Số chunks mong muốn (None = tự động)
            
        Returns:
            List các tuple (chunk_image, y_start, y_end)
        """
        height, width = image.shape[:2]
        
        # Phát hiện vùng text bằng contours
        text_regions = self.detect_text_regions(image)
        
        if not text_regions:
            # Nếu không tìm thấy text, chia đều
            if target_chunks is None:
                target_chunks = self.max_workers
            chunk_height = height // target_chunks
            chunks = []
            for i in range(target_chunks):
                start_y = i * chunk_height
                end_y = height if i == target_chunks - 1 else (i + 1) * chunk_height
                chunk = image[start_y:end_y, :]
                chunks.append((chunk, start_y, end_y))
            return chunks
        
        # Gộp và chia các vùng text
        if target_chunks is None:
            target_chunks = min(len(text_regions), self.max_workers * 2)
            
        # Tính toán chiều cao trung bình mong muốn
        avg_chunk_height = height // target_chunks
        merged_regions = self.merge_close_regions(
            text_regions, 
            max_region_height=avg_chunk_height * 2
        )
        
        # Tạo chunks
        chunks = []
        for i, (start_y, end_y) in enumerate(merged_regions):
            # Thêm một chút padding
            padded_start = max(0, start_y - 5)
            padded_end = min(height, end_y + 5)
            
            chunk = image[padded_start:padded_end, :]
            chunks.append((chunk, padded_start, padded_end))
            
        return chunks
    
    def ocr_chunk(self, chunk_data: Tuple[np.ndarray, int, int], 
                  chunk_index: int, lang='vie+eng') -> Dict:
        """
        Xử lý OCR cho một chunk
        
        Args:
            chunk_data: Tuple (chunk_image, y_start, y_end)
            chunk_index: Index của chunk
            lang: Ngôn ngữ OCR
            
        Returns:
            Dictionary chứa kết quả OCR
        """
        chunk_image, y_start, y_end = chunk_data
        
        try:
            # Tiền xử lý ảnh
            processed_chunk = self.preprocess_image(chunk_image)
            
            # Chuyển sang PIL Image
            pil_image = Image.fromarray(processed_chunk)
            
            # OCR với pytesseract
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            # Lấy text và thông tin chi tiết
            ocr_data = pytesseract.image_to_data(
                pil_image, 
                lang=lang, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Lấy text thuần
            text = pytesseract.image_to_string(
                pil_image, 
                lang=lang, 
                config=custom_config
            ).strip()
            
            return {
                'chunk_index': chunk_index,
                'y_start': y_start,
                'y_end': y_end,
                'text': text,
                'ocr_data': ocr_data,
                'success': True,
                'processing_time': time.time()
            }
            
        except Exception as e:
            return {
                'chunk_index': chunk_index,
                'y_start': y_start,
                'y_end': y_end,
                'text': '',
                'ocr_data': None,
                'success': False,
                'error': str(e),
                'processing_time': time.time()
            }
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh để tăng độ chính xác OCR
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh sau khi xử lý
        """
        # Chuyển sang ảnh xám nếu cần
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Khử nhiễu
        denoised = cv2.medianBlur(gray, 3)
        
        # Tăng độ tương phản
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Resize nếu ảnh quá nhỏ
        height, width = enhanced.shape
        if height < 50:
            scale_factor = 100 / height
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            enhanced = cv2.resize(enhanced, (new_width, new_height), 
                                interpolation=cv2.INTER_CUBIC)
        
        return enhanced
    
    def read_image_ocr(self, image_path: str, lang='vie+eng', 
                      target_chunks=None, save_debug=False) -> Dict:
        """
        Đọc OCR từ ảnh với xử lý song song thông minh
        
        Args:
            image_path: Đường dẫn ảnh
            lang: Ngôn ngữ OCR ('vie' cho tiếng Việt, 'eng' cho tiếng Anh, 'vie+eng' cho cả hai)
            target_chunks: Số chunks mong muốn (None = tự động)
            save_debug: Có lưu ảnh debug không
            
        Returns:
            Dictionary chứa kết quả OCR tổng hợp
        """
        start_time = time.time()
        
        # Đọc ảnh
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        print(f"Đang xử lý ảnh: {image_path}")
        print(f"Kích thước ảnh: {image.shape}")
        
        # Chia ảnh thành chunks thông minh
        chunks = self.create_smart_chunks(image, target_chunks)
        print(f"Đã chia thành {len(chunks)} chunks")
        
        # Lưu debug images nếu cần
        if save_debug:
            debug_dir = "debug_chunks"
            os.makedirs(debug_dir, exist_ok=True)
            for i, (chunk, y_start, y_end) in enumerate(chunks):
                cv2.imwrite(f"{debug_dir}/chunk_{i}_{y_start}_{y_end}.jpg", chunk)
        
        # Xử lý song song
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tất cả các chunks
            future_to_chunk = {
                executor.submit(self.ocr_chunk, chunk_data, i, lang): i 
                for i, chunk_data in enumerate(chunks)
            }
            
            # Thu thập kết quả
            for future in future_to_chunk:
                try:
                    result = future.result(timeout=30)  # Timeout 30s cho mỗi chunk
                    results.append(result)
                    print(f"Hoàn thành chunk {result['chunk_index']}")
                except Exception as e:
                    chunk_index = future_to_chunk[future]
                    print(f"Lỗi xử lý chunk {chunk_index}: {e}")
                    results.append({
                        'chunk_index': chunk_index,
                        'success': False,
                        'error': str(e),
                        'text': ''
                    })
        
        # Sắp xếp kết quả theo thứ tự chunk
        results.sort(key=lambda x: x['chunk_index'])
        
        # Gộp text
        full_text_lines = []
        successful_chunks = 0
        total_processing_time = 0
        
        for result in results:
            if result['success'] and result['text'].strip():
                full_text_lines.append(result['text'].strip())
                successful_chunks += 1
        
        full_text = '\n'.join(full_text_lines)
        end_time = time.time()
        
        return {
            'text': full_text,
            'total_chunks': len(chunks),
            'successful_chunks': successful_chunks,
            'processing_time': end_time - start_time,
            'chunk_results': results,
            'image_shape': image.shape,
            'lang': lang
        }