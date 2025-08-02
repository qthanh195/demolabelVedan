from ultralytics import YOLO
import numpy as np
import cv2

model_logo_unu = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_unu_v1.pt")
model_segment_label = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\detect_label_segment_v2.pt")
model_classifi_label = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_label_classification.pt")
model_detect_logo_tdc = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_tdc.pt")
model_detect_khoiluong_tdc = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_khoiluong.pt")
model_logo_recycling = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_recycling_v1.pt")
model_logo_halal = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_halal_v2.pt")
model_obb_label = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\obb_detect_label.pt")

custom_class_names_model_classifi = {
    0: "Label10",
    1: "Label11",
    2: "Label12",
    3: "Label13",
    4: "Label14",
    5: "Label15",
    6: "Label16",
    7: "Label17",
    8: "Label18",
    9: "Label19",
    10: "Label1",
    11: "Label20",
    12: "Label21",
    13: "Label22",
    14: "Label23",
    15: "Label24",
    16: "Label25",
    17: "Label26",
    18: "Label27",
    19: "Label28",
    20: "Label29",
    21: "Label2",
    22: "Label30",
    23: "Label31",
    24: "Label32",
    25: "Label33",
    26: "Label34",
    27: "Label35",
    28: "Label36",
    29: "Label37",
    30: "Label38",
    31: "Label39",
    32: "Label3",
    33: "Label40",
    34: "Label41",
    35: "Label42",
    36: "Label43",
    37: "Label44",
    38: "Label45",
    39: "Label46",
    40: "Label47",
    41: "Label48",
    42: "Label4",
    43: "Label5",
    44: "Label6",
    45: "Label7",
    46: "Label8",
    47: "Label9",
}

class AiHander:
    
    def detectLabel(self, image):
        # print(image)
        crop, rect_label = None, None
        results = model_segment_label.predict(image, conf=0.8, retina_masks=True)
        # # print("conf detect")
        # results = model_obb_label(image, conf=0.5)
        if len(results[0].masks.xy) == 0:
            print("No Label detected!")
            return None, None
        
        for idx, result in enumerate(results):
            for i, seg in enumerate(result.masks.xy):

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
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None, None, 0.0
        id, class_name, confidence = None, None, None
        results = model_classifi_label.predict(image, conf=0.5)
        if results[0].probs.top1conf.item() >= 0.8:
            id = results[0].probs.top1
            class_name = custom_class_names_model_classifi.get(id, results[0].names[id])
            print(f"Classified label: {class_name} with confidence: {results[0].probs.top1conf.item()}")
            confidence = results[0].probs.top1conf.item()
        return id, class_name, confidence