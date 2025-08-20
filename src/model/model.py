from ultralytics import YOLO

class ModelYolo:
    def __init__(self):
        super().__init__()
        print("[INFO] Loading all YOLO models...")

        self.model_segment_label = YOLO("model/detect_label_segment_v2.pt")
        self.model_classifi_label = YOLO("model/model_label_classification.pt")

        print("[INFO] All YOLO models loaded successfully ✅")