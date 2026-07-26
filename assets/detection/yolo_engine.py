"""
Detection Engine: currently wraps Ultralytics YOLO. Swapping in RT-DETR,
SAM, or a custom model later means adding a new file in this same folder
with the same detect() -> list[dict] interface, without touching the GUI.
"""


class YoloEngine:
    def __init__(self, weight_path):
        from ultralytics import YOLO  # imported lazily: app still runs without ultralytics installed
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = self.model.names[int(box.cls[0])]
            detections.append({"box": (x1, y1, x2, y2), "conf": conf, "label": label})
        return detections