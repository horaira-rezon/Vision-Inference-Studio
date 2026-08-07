"""
Detection Engine: wraps Ultralytics YOLO. YOLO detection models always
provide real bounding-box width/height via box.xyxy (the corner
coordinates) - your weed.pt included - so that's used directly, which is
why boxes come out as the object's actual (non-square) proportions rather
than a fixed square. The square-box fallback below only triggers if a
model ever returns a degenerate box (zero or negative width/height),
which standard YOLO detection output won't produce, but is handled
defensively rather than assumed away.

Confidence filtering happens HERE, not in the GUI layer, so it's a single
choke point: whatever detect() returns is exactly what gets drawn in the
streaming window AND exactly what's eligible to be sent to the Arduino -
there's no separate filtering logic to keep in sync between the two.

Tracking (ByteTrack/BotSORT) uses Ultralytics' built-in model.track()
instead of a plain forward pass when a tracker is selected, which is what
provides the per-box track_id.
"""

MIN_BOX_SIZE = 40
MAX_BOX_SIZE = 220
FALLBACK_BOX_SIZE = 80  # used only if the model gives no usable width/height

TRACKER_YAML = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


class YoloEngine:
    def __init__(self, weight_path):
        from ultralytics import YOLO  # imported lazily: app still runs without ultralytics installed
        self.model = YOLO(weight_path)

    def detect(self, frame, tracker=None, confidence_threshold=0.0):
        """
        tracker: None/"none" for a plain detection pass, or "bytetrack" /
            "botsort" to run Ultralytics' persistent tracker instead - each
            detection then also carries a "track_id".
        confidence_threshold: 0.0-1.0. Detections below this are dropped
            entirely (not just hidden from display).
        """
        tracker_yaml = TRACKER_YAML.get(tracker)
        if tracker_yaml is not None:
            results = self.model.track(frame, persist=True, tracker=tracker_yaml, verbose=False)
        else:
            results = self.model(frame, verbose=False)

        boxes = results[0].boxes
        detections = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[int(box.cls[0])]
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None

            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                # Fallback: model didn't provide a usable width/height -
                # draw a square box of a fixed, clamped size around
                # whatever center point is available.
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half = max(MIN_BOX_SIZE, min(FALLBACK_BOX_SIZE, MAX_BOX_SIZE)) // 2
                x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

            detections.append({"box": (x1, y1, x2, y2), "conf": conf, "label": label, "track_id": track_id})
        return detections