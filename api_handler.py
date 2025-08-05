import cv2
import numpy as np
import base64
import re
from camera_handler import BaslerCamera
from ai_hander import AiHander
from ocr_engine import OCR_Engine
from log import logging

class ApiHandler(BaslerCamera, OCR_Engine, AiHander):
    def __init__(self):
        super().__init__()
    
    def analyze_image(self, pallet_infos):
        logging.info(f"Nhận pallet_infos: {pallet_infos}")

        result_ui = {
            "label_detected": None,
            "pallet_detected": "F",
            "confidence": 0,
            "confidence_detect": 0,
            "confidence_classify": 0,
            "confidence_ocr": 0,
            "origin_image": None,
            "label_image": None,
        }

        logging.info("Bắt đầu phân tích ảnh...")
        result = self._get_image_and_classify()

        if result["origin_image"] is None:
            logging.warning("Không lấy được ảnh gốc từ camera hoặc file.")
            return result_ui

        result_ui["origin_image"] = result["origin_image"]

        if result["label_detected"] is None:
            logging.warning("Không phát hiện được nhãn!")
            return result_ui
        result_ui["label_detected"] = result["label_detected"]
        
        # Xử lý pallet
        if result["label_detected"] in [pallet_info[0] for pallet_info in pallet_infos[:5]]:
            for idx, (name, thresh) in enumerate(pallet_infos[:5]):
                logging.debug(f"So sánh label_detected={result['label_detected']} với name={name}, "
                              f"confidence={result['confidence']} >= thresh={thresh}")
                if result["label_detected"] == name and (result.get("confidence") or 0.0) >= thresh:
                    result_ui["pallet_detected"] = f"{chr(65 + idx)}"
                    logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                    break
        elif "" in [pallet_info[0] for pallet_info in pallet_infos[:5]]: # Nếu có pallet rỗng lấy vị trí đó
            for idx, (name, thresh) in enumerate(pallet_infos[:5]):
                if name == "":  # Đây là pallet rỗng
                    if (result.get("confidence") or 0.0) >= thresh:
                        result_ui["pallet_detected"] = f"{chr(65 + idx)}"  # A, B, C...
                        logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                        break
        else:
            result_ui["pallet_detected"] = "F"
            logging.info("Không phát hiện pallet hợp lệ, gán pallet_detected là 'F'.")

        
        result_ui["label_image"] = result["label_image"]
        result_ui["confidence_detect"] = result["confidence_detect"]
        result_ui["confidence_classify"] = result["confidence_classify"]
        result_ui["confidence_ocr"] = result["confidence_ocr"]
        result_ui["confidence"] = result["confidence"]

        logging.info(f"Kết quả trả về UI: {result_ui}")
        return result_ui

    def _handle_special_labels(self, id, class_name, label_image):
        logging.debug(f"Xử lý nhãn đặc biệt: id={id}, class_name={class_name}")
        class_name, label_image, confidence_ocr = class_name, label_image, 0
        match id:
            case 22:  # tdc
                class_name, label_image, confidence_ocr = self.classifi_tdc_with_ocr(label_image)
            case 40:  # recycling
                class_name, label_image, confidence_ocr = self.classify_label_logo_recycling(label_image)
            case 38:  # halal
                class_name, label_image, confidence_ocr = self.classify_label_logo_halal(label_image)
            case 26:  # unu
                class_name, label_image, confidence_ocr = self.classify_label_logo_unu(label_image)
        logging.debug(f"Kết quả nhãn đặc biệt: class_name={class_name}, confidence_ocr={confidence_ocr}")
        return class_name, label_image, confidence_ocr
            
    def _image_to_base64(self, image_np: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image_np)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def api_open_camera(self):
        self.open_camera()
        logging.info("Đã mở camera." if self.is_open else "Không mở được camera.")
        
    def api_close_camera(self):
        self.close_camera()
        logging.info("Đã tắt camera." if not self.is_open else "Đang mở camera.")
    
    def _get_image_and_classify(self):
        result_label_default = {
            "label_detected": None,
            "label_image": None,
            "origin_image": None,
            "confidence_detect": 0,
            "confidence_classify": 0,
            "confidence_ocr": 0,
            "confidence": 0,
        }
        
        # Chụp ảnh từ camera
        image = self.get_image()
        # image = cv2.imread("E:/2. GE/22. Vedan Vision Ocr/Image0505/image30/tdctest2.png")  # Thay thế bằng phương thức lấy ảnh từ camera thực tế

        if image is None or not hasattr(image, "shape"):
            logging.error("Ảnh đầu vào không hợp lệ!")
            return result_label_default
        result_label_default["origin_image"] = image

        # Đảm bảo ảnh là BGR
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Phát hiện nhãn
        label_image, rect_label, confidence_detect = self.detectLabel(image)
        logging.debug(f"Kết quả detectLabel: label_image is None? {label_image is None}, "
                      f"rect_label: {rect_label}, confidence_detect: {confidence_detect}")

        if label_image is None or not isinstance(label_image, np.ndarray) or label_image.size == 0:
            logging.error("Không phát hiện được nhãn hoặc nhãn bị lỗi!")
            return result_label_default
        result_label_default["confidence_detect"] = confidence_detect
        cv2.rectangle(image, rect_label[0], rect_label[1], (0, 255, 0), thickness=6)

        # Chỉ gọi phân loại khi label_image hợp lệ
        id, class_name, confidence_classify = self.classifiLabel(label_image)
        logging.debug(f"Kết quả classifiLabel: id={id}, class_name={class_name}, confidence_classify={confidence_classify}")
        result_label_default["confidence_classify"] = confidence_classify

        if id in [22, 40, 38, 26]:  # Các nhãn đặc biệt
            class_name, label_image, confidence_ocr = self._handle_special_labels(id, class_name, label_image)
            result_label_default["confidence"] = confidence_ocr
            result_label_default["confidence_ocr"] = confidence_ocr
            logging.debug(f"Kết quả _handle_special_labels: class_name={class_name}, confidence_ocr={confidence_ocr}")
        else:
            result_label_default["confidence"] = confidence_classify
            label_image = self._rotate_label_image(label_image)
            
        result_label_default["origin_image"] = image
        result_label_default["label_detected"] = class_name
        result_label_default["label_image"] = label_image

        return result_label_default

    def _rotate_label_image(self, label_image: np.ndarray) -> np.ndarray:
        #kiểm tra kích thước ảnh và luôn cho ảnh nằm ngang
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_CLOCKWISE)
        return label_image