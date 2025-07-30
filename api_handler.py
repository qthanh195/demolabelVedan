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
        
        image = None
        label_detect = "None Labels"
        pallet_detect = "Pallet F"
        conf = 0.00
        origin_image = np.ones((480, 640), dtype=np.uint8) * 255
        label_image = origin_image.copy()
        # Chụp một ảnh từ camera 
        image = self.get_image()
        # image = cv2.imread("E:/test.jpg")
        if image is None:
            print("Không lấy được ảnh từ camera!")
            return "No Image", pallet_detect, 0.0, self._image_to_base64(origin_image),self._image_to_base64(label_image)
        if len(image.shape) == 2:  # ảnh grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:  # ảnh có shape (H, W, 1)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Phát hiện có nhãn trong ảnh hay không
        label_image, rect_label = self.detectLabel(image)
        if rect_label == None:
            print("No Label!") 
            origin_image = cv2.putText(image, f"None Labels",(100,100),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
        else:
            id, class_name, confidence = self.classifiLabel(label_image)
            if id == None:
                return "No Image", pallet_detect, float(f"{0:.2f}"), self._image_to_base64(origin_image),self._image_to_base64(label_image)
            # Nếu là nhãn tdc thì kiểm tra xem là nhãn tdc nào
            elif id == 22: # 22: 'image30_1' -> tdc
               class_name, label_image = self.classifi_tdc_with_ocr(label_image)
            elif id == 40: # 40 : 'image47_1' -> recycling
                text_class, img_new = self.classify_label_logo_recycling(label_image)
                if text_class != "":
                    class_name = text_class
                    label_image = img_new
            elif id == 38: # 38 : 'image47'
                text_class, img_new = self.classify_label_logo_halal(label_image)
                if text_class != "":
                    class_name = text_class
                    label_image = img_new
            elif id == 26: # 26 : 'image34'
                text_class, img_new = self.classify_label_logo_unu(label_image)
                if text_class != "":
                    class_name = text_class
                    label_image = img_new
            cv2.imwrite("label_image.jpg", label_image)
            print(class_name)
            print(rect_label)
            origin_image = cv2.rectangle(image, rect_label[0], rect_label[1], (0,255,0), thickness= 6)
            cv2.putText(
    origin_image,
    f"Label {re.search(r'image(\d+)_1', class_name).group(1)} - {confidence:.2f}",
    (rect_label[0][0], rect_label[0][1]-30),
    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6
)
            conf = float(f"{confidence:.2f}")
            print(conf)
            # kiểm tra xem là palet nào?
            match_class = re.search(r"image(\d+)_1", class_name)
            if match_class:
                number_classname = int(match_class.group(1))
                label_detect = f"label{number_classname}"
            
            label_names = ["A", "B", "C", "D", "E"]
            list_name_label = [name_a, name_b, name_c, name_d, name_e]
            list_threshold = [thresh_a, thresh_b, thresh_c, thresh_d, thresh_e]
            for idx, name_label in enumerate(list_name_label):
                match_label = re.search(r"label(\d+)", name_label)
                if match_label:
                    number_name_label = int(match_label.group(1))
                    if number_name_label == number_classname:
                        # label_detect = f"Label{number_classname}"
                        if conf >= list_threshold[idx]:
                            pallet_detect = f"Pallet {label_names[idx]}"
                        return label_detect, pallet_detect,conf, self._image_to_base64(origin_image),self._image_to_base64(label_image)
            for idx, name_label in enumerate(list_name_label):
                if name_label == "other label":
                    pallet_detect = f"Pallet {label_names[idx]}"
                    return label_detect, pallet_detect,conf, self._image_to_base64(origin_image),self._image_to_base64(label_image)
        
        return label_detect, pallet_detect,conf, self._image_to_base64(origin_image),self._image_to_base64(label_image)
            
    def _image_to_base64(self, image_np: np.ndarray) -> str:
        # Chuyển ndarray sang buffer ảnh (dạng JPEG)
        _, buffer = cv2.imencode('.jpg', image_np)
        # Mã hóa sang base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def api_open_camera(self):
        self.open_camera()
        if self.is_open:
            print("Đã mở camera.")
        else:
            print("Không mở được camera.")
        
    def api_close_camera(self):
        self.close_camera()
        if not self.is_open:
            print("Đã tắt camera.")
        else:
            print("Đang mở camera.")