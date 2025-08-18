from ultralytics import YOLO

class ModelYolo:
    def __init__(self):
        super().__init__()
        print("[INFO] Loading all YOLO models...")

        self.model_logo_unu = YOLO("model/model_detect_logo_unu_v1.pt")
        self.model_segment_label = YOLO("model/detect_label_segment_v2.pt")
        self.model_classifi_label = YOLO("model/model_label_classification.pt")
        self.model_detect_logo_tdc = YOLO("model/model_detect_logo_tdc.pt")
        self.model_detect_khoiluong_tdc = YOLO("model/model_detect_khoiluong.pt")
        self.model_logo_recycling = YOLO("model/model_detect_logo_recycling_v1.pt")
        self.model_logo_halal = YOLO("model/model_detect_logo_halal_v2.pt")
        self.model_obb_label = YOLO("model/obb_detect_label.pt")

        print("[INFO] All YOLO models loaded successfully ✅")