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
    22: "Label-30",
    23: "Label-31",
    24: "Label-32",
    25: "Label-33",
    26: "Label-34",
    27: "Label-35",
    28: "Label-36",
    29: "Label-37",
    30: "Label-38",
    31: "Label-39",
    32: "Label-3",
    33: "Label-40",
    34: "Label-41",
    35: "Label-42",
    36: "Label-43",
    37: "Label-44",
    38: "Label-45",
    39: "Label-46",
    40: "Label-47",
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
            return None, None, 0.00
        
        for _, result in enumerate(results):
            for i, seg in enumerate(result.masks.xy):
                
                polygon = np.array(seg, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(polygon)
                rect_label = ((x, y), (x + w, y + h))
                image_cop = image.copy()
                
                crop_image = self.crop_image_with_contour(image_cop, polygon, offset_weight = 20, offset_height = 20)
                cv2.imwrite("crop_image.jpg", crop_image)
                # Tìm contours Áp dụng threshold
                gray_img = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite("threshold.jpg", thresh)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                filtered_contours = [c for c in contours if cv2.contourArea(c) < ((crop_image.shape[0]+10) * (crop_image.shape[1]+10))]
                print(len(filtered_contours))

                if filtered_contours:
                    # cv2.drawContours(crop_image, [max(filtered_contours, key=cv2.contourArea)], 0, (0,0,255), 4)
                    # cv2.imwrite("crop_image.jpg", crop_image)
                    cv2.imwrite("label1.jpg", self.crop_image_with_contour(crop_image, max(filtered_contours, key=cv2.contourArea)))
                    return self.rotate_with_ocr(self.crop_image_with_contour(crop_image, max(filtered_contours, key=cv2.contourArea))), rect_label, result.boxes.conf[i].item()
                
        return None, None, 0.00
        
    def classifiLabel(self, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None, None, 0.0
        id, class_name, confidence = None, None, None
        results = self.model_classifi_label.predict(image)
        logging.debug(f"Classified label: {class_name} with confidence: {results[0].probs.top1conf.item()}")
        if results[0].probs.top1conf.item() >= 0.6:
            id = results[0].probs.top1
            class_name = custom_class_names_model_classifi.get(id, results[0].names[id])
            # print(f"Classified label: {class_name} with confidence: {results[0].probs.top1conf.item()}")
            confidence = results[0].probs.top1conf.item()
        return id, class_name, confidence

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
        try:
            osd_data = pytesseract.image_to_osd(image, config='--oem 3 --psm 0')
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