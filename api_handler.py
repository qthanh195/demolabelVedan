import cv2
import numpy as np
import base64
import re
from camera_handler import BaslerCamera
from ai_hander import AiHander
from ocr_engine import OCR_Engine


class ApiHandler(BaslerCamera, OCR_Engine, AiHander):
    def __init__(self):
        super().__init__()
    
    def analyze_image(self, name_a, name_b, name_c, 
                            name_d, name_e, name_f, 
                            thresh_a, thresh_b, thresh_c, 
                            thresh_d, thresh_e, thresh_f):
        """
            1. Chụp một ảnh gốc
            2. Phát hiện có nhãn hay không
            3. Phát hiện là nhãn nào
            4. Nếu nhãn đã xuất hiện trong 1 trong 5 pallet rồi thì trả về nhãn đó, pallet đó
            5. Nếu nhãn chưa có trong 1 trong 5 pallet thì thêm trả về nhãn đó và pallet trống 
            hoặc không có pallet trống thì trả về pallet other (pallet7)
            
        Args:
            name_a (string): Tên label tại Pallet A
            name_b (string): Tên label tại Pallet B
            name_c (string): Tên label tại Pallet C
            name_d (string): Tên label tại Pallet D
            name_e (string): Tên label tại Pallet E
            name_f (string): Fallet other label
            thresh_a (float): Threshold tại Pallet A 
            thresh_b (float): Threshold tại Pallet B 
            thresh_c (float): Threshold tại Pallet C 
            thresh_d (float): Threshold tại Pallet D 
            thresh_e (float): Threshold tại Pallet E 
            thresh_f (float): __ không dùng

        Returns:
            label_detect (string): Xác đinh là label nào
            conf (float): Confident đó là bao nhiêu
            origin_image (base64): Ảnh gốc chụp từ camera
            label_image (base64): Ảnh nhãn được cắt từ ảnh gốc
        """

        label_detect = "None Labels"
        pallet_detect = "Pallet F"
        conf = 0.00

        image_origin, label_image, rect_label, label_detect, confidence = self._get_image_and_classify()
        
        if image_origin is None:
            return label_detect, pallet_detect, conf, None, None
        
        if label_image is None:
            print("No Label!")
            return label_detect, pallet_detect, conf, self._image_to_base64(image_origin), self._image_to_base64(np.ones((100, 100), dtype=np.uint8) * 255)
        
        conf = float(f"{confidence:.2f}")
        
        cv2.rectangle(image_origin, rect_label[0], rect_label[1], (0, 255, 0), thickness=6)
        cv2.putText(image_origin, f"{label_detect} - {conf}", (rect_label[0][0], rect_label[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

        return label_detect, pallet_detect, conf, self._image_to_base64(image_origin), self._image_to_base64(label_image)

    def _handle_special_labels(self, id, class_name, label_image):
        if id == 22:  # tdc
            class_name, label_image = self.classifi_tdc_with_ocr(label_image)
        elif id == 40:  # recycling
            text_class, img_new = self.classify_label_logo_recycling(label_image)
            if text_class:
                class_name = text_class
                label_image = img_new
        elif id == 38:  # halal
            text_class, img_new = self.classify_label_logo_halal(label_image)
            if text_class:
                class_name = text_class
                label_image = img_new
        elif id == 26:  # unu
            text_class, img_new = self.classify_label_logo_unu(label_image)
            if text_class:
                class_name = text_class
                label_image = img_new
        return class_name, label_image
            
    def _image_to_base64(self, image_np: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image_np)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def api_open_camera(self):
        self.open_camera()
        print("Đã mở camera." if self.is_open else "Không mở được camera.")
        
    def api_close_camera(self):
        self.close_camera()
        print("Đã tắt camera." if not self.is_open else "Đang mở camera.")
    
    def _get_image_and_classify(self):
        """
        Lấy ảnh từ camera và phân loại nhãn.
        Returns:
            label_image: Ảnh nhãn được cắt ra (hoặc None nếu không có).
            rect_label: Vị trí nhãn (hoặc None nếu không có).
            class_name: Tên nhãn phân loại (hoặc None nếu không có).
            confidence: Độ tin cậy (float, 0.0 nếu không có).
        """
        # Chụp ảnh từ camera
        image = self.get_image()
        if image is None:
            print("Không lấy được ảnh từ camera!")
            return None, None, None, None, 0.0

        # Đảm bảo ảnh là BGR
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Phát hiện nhãn
        label_image, rect_label = self.detectLabel(image)
        if label_image is None:
            print("No Label!")
            return image, None, None, None, 0.0

        # Chỉ gọi phân loại khi label_image hợp lệ
        id, class_name, confidence = self.classifiLabel(label_image)
        if label_image is None or not isinstance(label_image, np.ndarray) or label_image.size == 0:
            return image, None, None, None, 0.0

        # Xử lý nhãn đặc biệt nếu có
        class_name, label_image = self._handle_special_labels(id, class_name, label_image)
        cv2.imwrite("label_image.jpg", label_image)
        print("Class: ", class_name)
        return image, label_image, rect_label, class_name, confidence
        
    def add_label(self):
        """
        Thêm nhãn vào pallet
        """
        pass