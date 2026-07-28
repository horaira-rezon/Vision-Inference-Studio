"""
Detection Engine: wraps Ultralytics YOLO. YOLO detection models always
provide real bounding-box width/height via box.xyxy (the corner
coordinates) - your weed.pt included - so that's used directly, which is
why boxes come out as the object's actual (non-square) proportions rather
than a fixed square. The square-box fallback below only triggers if a
model ever returns a degenerate box (zero or negative width/height),
which standard YOLO detection output won't produce, but is handled
defensively rather than assumed away.
"""

MIN_BOX_SIZE = 40
MAX_BOX_SIZE = 220
FALLBACK_BOX_SIZE = 80  # used only if the model gives no usable width/height


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

            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                # Fallback: model didn't provide a usable width/height -
                # draw a square box of a fixed, clamped size around
                # whatever center point is available.
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half = max(MIN_BOX_SIZE, min(FALLBACK_BOX_SIZE, MAX_BOX_SIZE)) // 2
                x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

            detections.append({"box": (x1, y1, x2, y2), "conf": conf, "label": label})
        return detections