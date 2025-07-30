from ultralytics import YOLO
import numpy as np
import cv2

model_logo_unu = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_unu_v1.pt")
model_segment_label = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\detect_label_segment.pt")
model_classifi_label = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_label_classification.pt")
model_detect_logo_tdc = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_tdc.pt")
model_detect_khoiluong_tdc = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_khoiluong.pt")
model_logo_recycling = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_recycling_v1.pt")
model_logo_halal = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_halal_v2.pt")

class AiHander:
    # def __init__(self):
    #     pass
        # self.model_segment_label = model_segment_label
        # self.model_classifi_label = model_classifi_label
        # self.model_detect_logo_tdc = model_detect_logo_tdc
        # self.model_detect_khoiluong_tdc = model_detect_khoiluong_tdc
        # self.model_logo_recycling = model_logo_recycling
        # self.model_logo_halal = model_logo_halal
        # self.model_logo_unu = model_logo_unu
    
    def detectLabel(self, image):
        # print(image)
        crop, rect_label = np.ones((480, 640), dtype=np.uint8) * 255, None
        results = model_segment_label.predict(image, conf=0.5)
        # print("conf detect")
        for idx, result in enumerate(results):
            if result.masks is None:
                continue
            for i, (seg, cls) in enumerate(zip(result.masks.xy, result.boxes.cls)):
                print("Conf detect", result.boxes.conf[i].item())
                polygon = np.array(seg, dtype=np.int32)
                
                x, y, w, h = cv2.boundingRect(polygon)
                rect_label = ((x, y), (x + w, y + h))
                
                # 1. Tìm hình chữ nhật xoay bao quanh polygon
                rect = cv2.minAreaRect(polygon)
                box = cv2.boxPoints(rect)
                box = np.int8(box)
                # 2. Lấy ma trận xoay
                center, size, angle = rect
                size = tuple([int(s) for s in size])
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # 3. Xoay toàn ảnh
                rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                # 4. Crop vùng rectangle đã xoay
                x, y = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)
                w, h = size
                print("confident: ", result.boxes.conf[i].item())
                crop = rotated[y:y+h, x:x+w]
        return crop, rect_label 
        
    def classifiLabel(self, image):
        id, class_name, confidence = None, None, None
        results = model_classifi_label.predict(image, conf=0.5)
        if results[0].probs.top1conf.item() >= 0.8:
            id = results[0].probs.top1
            class_name = results[0].names[id]
            confidence = results[0].probs.top1conf.item()
        return id, class_name, confidence