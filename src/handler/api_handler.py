import cv2
import numpy as np
from src.camera.camera_handler import BaslerCamera
from src.function.process_image import ProcessImage
from logs.log import logging
import datetime

class ApiHandler(BaslerCamera, ProcessImage):
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
            "text": "",
            "weight": "",
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
        result_ui["text"] = result["text"]
        result_ui["weight"] = result["weight"]

        logging.info(f"Kết quả trả về UI: {result_ui}")
        return result_ui
     
    def _get_image_and_classify(self):
        result_label_default = {
            "label_detected": None,
            "label_image": None,
            "origin_image": None,
            "confidence_detect": 0,
            "confidence_classify": 0,
            "confidence_ocr": 0,
            "confidence": 0,
            "text": "",
            "weight": "",
            
        }
        
        # Chụp ảnh từ camera
        # image = self.get_image()
        image = cv2.imread("data\capture/20250820_111405.jpg")  # Thay thế bằng phương thức lấy ảnh từ camera thực tế

        if image is None or not hasattr(image, "shape"): 
            logging.error("Ảnh đầu vào không hợp lệ!")
            return result_label_default
        result_label_default["origin_image"] = image
        # cv2.imwrite(f"data\capture/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", image)

        # Đảm bảo ảnh là BGR
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Phát hiện nhãn
        label_image, rect_label, confidence_detect, area = self.detectLabel(image)
        print("Area: ", area)
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
            class_name, label_image, confidence_ocr, text, weight = self.handle_special_labels(id, class_name, label_image)
            result_label_default["confidence"] = confidence_ocr
            result_label_default["confidence_ocr"] = confidence_ocr
            result_label_default["text"] = text
            result_label_default["weight"] = weight
            logging.debug(f"Kết quả handle_special_labels: class_name={class_name}, confidence_ocr={confidence_ocr}")
        elif id is not None:
            result_label_default["confidence"] = confidence_classify

            match id:
                case 28:  # Label-28
                    class_name = self.classify_label_36(label_image)  # Gọi hàm classify_label_36 nếu id là 28
                    logging.debug(f"Kết quả phân loại label 36: class_name={class_name}")
        
        image_sample = None
        image_sample = cv2.imread(f"data/SampleData/{class_name}.jpg")
        if image_sample is not None:
            height, width = image_sample.shape[:2]
            ref_area = height * width

            percentage = (area / ref_area * 100) if ref_area > 0 else 0
            print(f"percentage: {percentage:.2f}%")
            
        result_label_default["origin_image"] = image
        result_label_default["label_detected"] = class_name
        result_label_default["label_image"] = label_image
        cv2.imwrite(f"data/{class_name}.jpg", label_image)
        return result_label_default

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
        image = cv2.imread("data\capture/20250820_110954.jpg")  # Thay thế bằng phương thức lấy ảnh từ camera thực tế
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
            result_ui["confidence"] = confidence_ocr
            result_ui["confidence_ocr"] = confidence_ocr
            result_ui["text"] = text
            result_ui["weight"] = weight
            logging.debug(f"Kết quả handle_special_labels: class_name={class_name}, confidence_ocr={confidence_ocr}")
        elif id is not None:
            match id:
                case 28:  # Label-28
                    class_name = self.classify_label_36(label_image)  # Gọi hàm classify_label_36 nếu id là 28
                    logging.debug(f"Kết quả phân loại label 36: class_name={class_name}")
        
        result_ui["origin_image"] = image
        result_ui["label_detected"] = class_name
        result_ui["label_image"] = label_image
        
        # Xử lý pallet
        if result_ui["label_detected"] in [pallet_info[0] for pallet_info in pallet_infos[:5]]:
            for idx, (name, thresh,_,_) in enumerate(pallet_infos[:5]):
                logging.debug(f"So sánh label_detected={result_ui['label_detected']} với name={name}, "
                              f"confidence={result_ui['confidence']} >= thresh={thresh}")
                if result_ui["label_detected"] == name and (result_ui.get("confidence") or 0.0) >= thresh:
                    result_ui["pallet_detected"] = f"{chr(65 + idx)}"
                    logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                    break
        elif "" in [pallet_info[0] for pallet_info in pallet_infos[:5]]: # Nếu có pallet rỗng lấy vị trí đó
            for idx, (name, thresh,_,_) in enumerate(pallet_infos[:5]):
                if name == "":  # Đây là pallet rỗng
                    if (result_ui.get("confidence") or 0.0) >= thresh:
                        result_ui["pallet_detected"] = f"{chr(65 + idx)}"  # A, B, C...
                        logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                        break
        else:
            result_ui["pallet_detected"] = "F"
            logging.info("Không phát hiện pallet hợp lệ, gán pallet_detected là 'F'.")
        
        return result_ui