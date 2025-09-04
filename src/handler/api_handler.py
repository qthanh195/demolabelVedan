import cv2
import numpy as np
from src.camera.camera_handler import BaslerCamera, CameraWebcam
from src.function.process_image import ProcessImage
from logs.log import logging
import datetime
import time

# class ApiHandler(BaslerCamera, ProcessImage):
class ApiHandler(CameraWebcam, ProcessImage): 
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
        start_time = time.time()
        
        # Chụp ảnh từ camera
        image = None
        image = self.get_image()
        # image = cv2.imread("data\\capture\\20250826_155646.jpg")
        original_image_height = image.shape[0]
        original_image_width = image.shape[1]
        
        
        # print("Thoi gian chup: ", time.time()-start_time)
        # image = cv2.imread("resize_5mp.jpg")  # Thay thế bằng phương thức lấy ảnh từ camera thực tế
        
        #resize anh neu can
        # image = cv2.resize(image,(0, 0), fx= 0.5, fy= 0.5)
        cv2.imwrite(f"resize_5mp.jpg", image)
        
        #Kiểm tra có ảnh không?
        if image is None or not hasattr(image, "shape"): 
            logging.error("Ảnh đầu vào không hợp lệ!")
            return result_ui
        
        #Gán ảnh result nếu có ảnh và lưu ảnh
        result_ui["origin_image"] = image
        cv2.imwrite(f"data\capture/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", image)
        
        # Đảm bảo ảnh là BGR
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        
        # Phát hiện nhãn
        label_image, rect_label, confidence_detect, label_infos = self.detectLabel(image)
        logging.debug(f"Kết quả detectLabel: label_image is None? {label_image is None}, "
                      f"rect_label: {rect_label}, confidence_detect: {confidence_detect}")
        print("Thoi gian detect: ", time.time()-start_time)
        
        #Kiểm tra có nhãn không?
        if label_image is None or not isinstance(label_image, np.ndarray) or label_image.size == 0:
            logging.error("Không phát hiện được nhãn hoặc nhãn bị lỗi!")
            return result_ui
        
        cv2.imwrite("label.jpg", label_image)
        
        #gán giá trị confidence_detect cho result
        result_ui["confidence_detect"] = confidence_detect
        cv2.rectangle(image, rect_label[0], rect_label[1], (0, 255, 0), thickness=6)
        
        #Kiểm tra giá giá trị Confidence detect so cới giá trị threshold tại 5 vị trí
        if confidence_detect < min(pallet_infos[0][1], pallet_infos[1][1], pallet_infos[2][1], pallet_infos[3][1], pallet_infos[4][1]):
            logging.debug(f"Confidence detect = {confidence_detect} không thõa các vị trí.")
            return result_ui
        
        # Chỉ gọi phân loại khi label_image hợp lệ
        id, class_name, confidence_classify = self.classifiLabel(label_image)
        print("Thoi gian phan loai: ", time.time()-start_time)
        
        # Kiểm tra nhã có được phân loại không?
        if id is None:
            logging.debug(f"Phân nhóm nhãn lỗi.")
            return result_ui
        
        #gán giá trị phân loại cho result
        logging.debug(f"Kết quả classifiLabel: id={id}, class_name={class_name}, confidence_classify={confidence_classify}")
        result_ui["confidence_classify"] = confidence_classify
        result_ui["label_image"] = label_image
        
        #Kiểm tra giá trị confidence_classify có thõa threshold tại 5 vị trí pallet
        if confidence_classify < min(pallet_infos[0][2], pallet_infos[1][2], pallet_infos[2][2], pallet_infos[3][2], pallet_infos[4][2]):
             return result_ui
        
        #Phân loại với những nhãn đặc biệt bằng ocr 
        if id in [22, 40, 38, 26]:  # Các nhãn đặc biệt
            result_special_labels = self.handle_special_labels(id, label_image, original_image_height, original_image_width) # class_name, confidence_ocr, text, weight
            if result_special_labels[0] != "":
                class_name = result_special_labels[0]
                result_ui["confidence_ocr"] = result_special_labels[1]
                result_ui["text"] = result_special_labels[2]
                result_ui["weight"] = result_special_labels[3]
        elif id is not None:
            match id:
                case 28:  # Label-28
                    class_name = self.classify_label_36(label_image, original_image_height, original_image_width)  # Gọi hàm classify_label_36 nếu id là 28
                    logging.debug(f"Kết quả phân loại label 36: class_name={class_name}")
        print("Thoi gian ocr: ", time.time()-start_time)
        
        #gán class_name
        result_ui["label_detected"] = class_name
        print("class name: ", class_name)
        
        #Kiểm tra tỉ lệ nhãn

        (label_width, label_height) =  label_infos[1]
        if label_image.shape[1] >= label_image.shape[0]:
            label_width, label_height = max(label_width, label_height), min(label_width, label_height)
        else:
            label_width, label_height = min(label_width, label_height), max(label_width, label_height)
        logging.debug(f"Kích thước nhãn phát hiện được: width={label_width}, height={label_height}, perimeter={label_infos[2]}, area={label_infos[0]}   ")

        original_size = self.get_image_size(image_name = f"{class_name}.jpg")
        if original_size is None:
            logging.debug(f"Không tìm thấy kích thước gốc cho {class_name}.jpg")
            return result_ui

        original_width, original_height, original_area, original_perimeter = original_size
        logging.debug(f"Kích thước gốc của {class_name}.jpg: width={original_width}, height={original_height}, perimeter={original_perimeter}, area={original_area}")
        
        
        #Tính độ biến dạng của nhãn so với nhãn gốc
        distortion = (label_infos[2] ** 2 / label_infos[0]) / (original_perimeter ** 2 / original_area) if original_area != 0 else 0

        p = (label_width/original_width) - (label_height/ original_height)
        logging.debug(f"Tỉ lệ P: {p}")
        
        if p == 0:
            logging.debug(f"Tỉ lệ bằng 0")
            result_ui["%_area"] = 100
        elif p > 0:
            logging.debug(f"Tỉ lệ dương")
            h = (label_width*original_height)/original_width - label_height
            print(f"h: {h}")
            a_label = label_width * (h+label_height)
        else:
            logging.debug(f"Tỉ lệ âm")
            w = (label_height*original_width)/original_height - label_width
            print(f"w: {w}")
            a_label = label_height * (w+label_width)
        
        # per_area = label_infos[0]/a_label*100# + (distortion-1)*100
        per_area = label_infos[0]/a_label*100 + (1 - distortion)*100

        result_ui["%_area"] = per_area if per_area < 100 else 100
        logging.debug(f"Kiểm tra nhãn với giá trị diện tích {label_infos[0]/a_label*100}% độ biến dạng so với nhãn gốc {(distortion-1)*100} trả về kết quả {result_ui['%_area']}")

        print("Thoi gian Kiem tra kich thuoc: ", time.time()-start_time)
        
        if result_ui["%_area"] < pallet_infos[6]:
            logging.debug(f"Diện tích của nhãn không thõa threshold_area {pallet_infos[6]}")
            return result_ui

        # Xử lý pallet
        if result_ui["label_detected"] in [pallet_info[0] for pallet_info in pallet_infos[:5]]:
            for idx, (name,thresh_object,thresh_group, thresh_ocr) in enumerate(pallet_infos[:5]):
                logging.debug(f"So sánh label_detected={result_ui['label_detected']} với name={name}")
                if result_ui["text"] == "":
                    if result_ui["label_detected"] == name and (result_ui.get("confidence_detect") or 0.0) >= thresh_object and (result_ui.get("confidence_classify") or 0.0) >= thresh_group:
                        result_ui["pallet_detected"] = f"{chr(65 + idx)}"
                        logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                        break
                        
                else:
                    if result_ui["label_detected"] == name and (result_ui.get("confidence_ocr") or 0.0) >= thresh_ocr and (result_ui.get("confidence_detect") or 0.0) >= thresh_object and (result_ui.get("confidence_classify") or 0.0) >= thresh_group:
                        result_ui["pallet_detected"] = f"{chr(65 + idx)}"
                        logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                        break
                    
        elif "" in [pallet_info[0] for pallet_info in pallet_infos[:5]]: # Nếu có pallet rỗng lấy vị trí đó
            for idx, (name,thresh_object,thresh_group, thresh_ocr) in enumerate(pallet_infos[:5]):
                if result_ui["text"] == "":
                    if name == ""and (result_ui.get("confidence_detect") or 0.0) >= thresh_object and (result_ui.get("confidence_classify") or 0.0) >= thresh_group:
                        result_ui["pallet_detected"] = f"{chr(65 + idx)}"  # A, B, C...
                        logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                        break
                else:
                    if name == ""and (result_ui.get("confidence_ocr") or 0.0) >= thresh_ocr and (result_ui.get("confidence_detect") or 0.0) >= thresh_object and (result_ui.get("confidence_classify") or 0.0) >= thresh_group:
                        result_ui["pallet_detected"] = f"{chr(65 + idx)}"  # A, B, C...
                        logging.info(f"Đã gán pallet_detected: {result_ui['pallet_detected']}")
                        break
                    
        else:
            result_ui["pallet_detected"] = "F"
            logging.info("Không phát hiện pallet hợp lệ, gán pallet_detected là 'F'.")
        print("Thoi gian chon vi tri: ", time.time()-start_time)
        return result_ui