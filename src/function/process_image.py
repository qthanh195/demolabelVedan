import cv2
import numpy as np
import base64
import pytesseract
import re
import logging
import time
from src.function.ocr import OCR_Engine

class ProcessImage(OCR_Engine):
    def __init__(self):
        super().__init__()

    def image_to_base64(self, image_np: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image_np)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def rotate_label_image(self, label_image: np.ndarray) -> np.ndarray:
        try:
            osd_data = pytesseract.image_to_osd(label_image, config='--oem 3 --psm 0')
            print("OSD output:", osd_data)
            match = re.search(r'(?<=Rotate: )\d+', osd_data)
            if match:
                angle = int(match.group(0))
                if angle == 90:
                    label_image = cv2.rotate(label_image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    label_image = cv2.rotate(label_image, cv2.ROTATE_180)
                elif angle == 270:
                    label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception as e:
            print("Không thể chạy OSD:", e)
            if label_image.shape[0] < label_image.shape[1]:
                label_image = cv2.rotate(label_image, cv2.ROTATE_90_CLOCKWISE)
        
        return label_image
    
    def handle_special_labels(self, id, class_name, label_image):
        logging.debug(f"Xử lý nhãn đặc biệt: id={id}, class_name={class_name}")
        class_name, label_image, confidence_ocr, text, weight = class_name, label_image, 0, "", ""
        match id:
            case 22:  # tdc
                class_name, label_image, confidence_ocr, text, weight = self.classifi_tdc_with_ocr(label_image)
            case 40:  # recycling
                start_time = time.time()
                class_name, label_image, confidence_ocr, text, weight = self.classify_label_logo_recycling(label_image)
                print(f"Thời gian chạy: {time.time() - start_time:.4f} giây")
            case 38:  # halal
                class_name, label_image, confidence_ocr, text, weight = self.classify_label_logo_halal(label_image)
            case 26:  # unu
                class_name, label_image, confidence_ocr, text, weight = self.classify_label_logo_unu(label_image)
        logging.debug(f"Kết quả nhãn đặc biệt: class_name={class_name}, confidence_ocr={confidence_ocr}")
        return class_name, label_image, confidence_ocr, text, weight