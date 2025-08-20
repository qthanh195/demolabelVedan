import cv2
import numpy as np
from src.camera.camera_handler import BaslerCamera
from src.function.process_image import ProcessImage
from logs.log import logging
import datetime

class ApiHandler(BaslerCamera, ProcessImage):
    def __init__(self):
        super().__init__()
    
    def api_open_camera(self):
        self.open_camera()
        logging.info("Đã mở camera." if self.is_open else "Không mở được camera.")
        
    def api_close_camera(self):
        self.close_camera()
        logging.info("Đã tắt camera." if not self.is_open else "Đang mở camera.")
  
    def process(self, pallet_infos):
        logging.info(f"Nhận pallet_infos: {pallet_infos}")

        result_ui = {
            "label_detected": "",
            "pallet_detected": "F",
            "confidence_detect": 0,
            "confidence_classify": 0,
            "confidence_ocr": 0,
            "origin_image": None,
            "label_image": None,
            "text": "",
            "weight": "",
            "%_area": "",
        }
        
        # Chụp ảnh từ camera
        # image = self.get_image()
        image = cv2.imread("E:/2. GE/22. Vedan Vision Ocr\Image0505\image34\img_20250506_174557.png")  # Thay thế bằng phương thức lấy ảnh từ camera thực tế
        if image is None or not hasattr(image, "shape"): 
            logging.error("Ảnh đầu vào không hợp lệ!")
            return result_ui
        result_ui["origin_image"] = image
        # cv2.imwrite(f"data\capture/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", image)
        
        # Đảm bảo ảnh là BGR
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # Phát hiện nhãn
        label_image, rect_label, confidence_detect, area = self.detectLabel(image)
        logging.debug(f"Kết quả detectLabel: label_image is None? {label_image is None}, "
                      f"rect_label: {rect_label}, confidence_detect: {confidence_detect}")
        
        if confidence_detect < min(pallet_infos[0][1], pallet_infos[1][1], pallet_infos[2][1], pallet_infos[3][1], pallet_infos[4][1]):
            return result_ui
        
        if label_image is None or not isinstance(label_image, np.ndarray) or label_image.size == 0:
            logging.error("Không phát hiện được nhãn hoặc nhãn bị lỗi!")
            return result_ui
        result_ui["confidence_detect"] = confidence_detect
        cv2.rectangle(image, rect_label[0], rect_label[1], (0, 255, 0), thickness=6)
        
        # Chỉ gọi phân loại khi label_image hợp lệ
        id, class_name, confidence_classify = self.classifiLabel(label_image)
        logging.debug(f"Kết quả classifiLabel: id={id}, class_name={class_name}, confidence_classify={confidence_classify}")
        result_ui["confidence_classify"] = confidence_classify
        result_ui["label_image"] = label_image
        
        if confidence_classify < min(pallet_infos[0][2], pallet_infos[1][2], pallet_infos[2][2], pallet_infos[3][2], pallet_infos[4][2]):
             return result_ui
         
        image_sample = None
        image_sample = cv2.imread(f"data/SampleData/{class_name}.jpg")
        if image_sample is not None:
            border_size = 5
            expanded_image_sample = cv2.copyMakeBorder(image_sample,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)  # pixel đen
            )
            cv2.imwrite("expanded_image_sample.jpg", expanded_image_sample)
            # Tìm contours Áp dụng threshold
            gray_img = cv2.cvtColor(expanded_image_sample, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)
            # cv2.imwrite("threshold.jpg", thresh)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ref_area = cv2.contourArea(max([c for c in contours if cv2.contourArea(c) < ((expanded_image_sample.shape[0]+10) * (expanded_image_sample.shape[1]+10))], key=cv2.contourArea))

            result_ui["%_area"] = ((area / ref_area * 100) if ref_area > 0 else 0) if (area / ref_area * 100) < 100 else 100
            logging.debug(f"Diện tích của nhãn là {area} so với nhãn gốc: {result_ui['%_area']}")
        
            if result_ui["%_area"] < pallet_infos[6]:
                logging.debug(f"Diện tích của nhãn không thõa threshold_area {pallet_infos[6]}")
                return result_ui
        
        if id in [22, 40, 38, 26]:  # Các nhãn đặc biệt
            class_name, label_image, confidence_ocr, text, weight = self.handle_special_labels(id, class_name, label_image)
            result_ui["confidence_ocr"] = confidence_ocr
            result_ui["text"] = text
            result_ui["weight"] = weight
        elif id is not None:
            match id:
                case 28:  # Label-28
                    class_name = self.classify_label_36(label_image)  # Gọi hàm classify_label_36 nếu id là 28
                    logging.debug(f"Kết quả phân loại label 36: class_name={class_name}")
        
        result_ui["label_detected"] = class_name
        
        # Xử lý pallet
        if result_ui["label_detected"] in [pallet_info[0] for pallet_info in pallet_infos[:5]]:
            for idx, (name,thresh_object,thresh_group, thresh_ocr) in enumerate(pallet_infos[:5]):
                logging.debug(f"So sánh label_detected={result_ui['label_detected']} với name={name}")
                if result_ui["label_detected"] == name and (result_ui.get("confidence_ocr") or 0.0) >= thresh_ocr and (result_ui.get("confidence_detect") or 0.0) >= thresh_object and (result_ui.get("confidence_classify") or 0.0) >= thresh_group:
                    result_ui["pallet_detected"] = f"{chr(65 + idx)}"
                    logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                    break
        elif "" in [pallet_info[0] for pallet_info in pallet_infos[:5]]: # Nếu có pallet rỗng lấy vị trí đó
            for idx, (name,thresh_object,thresh_group, thresh_ocr) in enumerate(pallet_infos[:5]):
                if name == ""and (result_ui.get("confidence_ocr") or 0.0) >= thresh_ocr and (result_ui.get("confidence_detect") or 0.0) >= thresh_object and (result_ui.get("confidence_classify") or 0.0) >= thresh_group:
                    result_ui["pallet_detected"] = f"{chr(65 + idx)}"  # A, B, C...
                    logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                    break
        else:
            result_ui["pallet_detected"] = "F"
            logging.info("Không phát hiện pallet hợp lệ, gán pallet_detected là 'F'.")
        
        return result_ui