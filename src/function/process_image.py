import cv2
import numpy as np
import base64
import logging
import time
from src.function.ocr import OCR_Engine
import os

class ProcessImage(OCR_Engine):
    def __init__(self):
        super().__init__()

    def image_to_base64(self, image_np: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image_np)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
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
    