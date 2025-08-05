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

class AiHander:
    
    def detectLabel(self, image):
        # print(image)
        crop, rect_label, confident_detect = None, None, 0.00
        results = model_segment_label.predict(image, conf=0.9, retina_masks=True)
        if not results or results[0].masks is None or len(results[0].masks.xy) == 0:
            print("No Label detected!")
            return crop, rect_label, confident_detect
        
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
                confident_detect = result.boxes.conf[i].item()
        return crop, rect_label, confident_detect
        
    def classifiLabel(self, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return None, None, 0.0
        id, class_name, confidence = None, None, None
        results = model_classifi_label.predict(image)
        # if results[0].probs.top1conf.item() >= 0.8:
        id = results[0].probs.top1
        class_name = custom_class_names_model_classifi.get(id, results[0].names[id])
        print(f"Classified label: {class_name} with confidence: {results[0].probs.top1conf.item()}")
        confidence = results[0].probs.top1conf.item()
        return id, class_name, confidence