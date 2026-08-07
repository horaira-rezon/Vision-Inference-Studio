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

Tracking (ByteTrack/BotSORT/OC-SORT/DeepSORT) uses Ultralytics' built-in model.track()
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

# "ocsort" / "deepsort" aren't Ultralytics-native trackers (no built-in
# yaml for either), so they run through custom_trackers.py instead - see
# YoloEngine.detect() below.
CUSTOM_TRACKERS = ("ocsort", "deepsort")


class YoloEngine:
    def __init__(self, weight_path):
        from ultralytics import YOLO  # imported lazily: app still runs without ultralytics installed
        self.model = YOLO(weight_path)
        self._last_tracker_param = "__unset__"

        # a fresh model load means any previous OC-SORT/DeepSORT instance
        # (module-level, in custom_trackers.py) belongs to the old model -
        # drop it so its track IDs/appearance state don't leak forward
        from assets.detection import custom_trackers
        custom_trackers.reset()

    def detect(self, frame, tracker=None, confidence_threshold=0.0):
        """
        tracker: None/"none" for a plain detection pass, "bytetrack" /
            "botsort" to run Ultralytics' own persistent tracker, or
            "ocsort" / "deepsort" to run the boxmot-backed tracker in
            custom_trackers.py instead - every option gives each detection
            a "track_id".
        confidence_threshold: 0.0-1.0. Detections below this are dropped
            entirely (not just hidden from display).
        """
        if tracker != self._last_tracker_param:
            # Ultralytics' model.track() does two things the first time
            # it's called: builds self.model.predictor, AND permanently
            # registers on_predict_start/on_predict_postprocess_end
            # callbacks on self.model itself (register_tracker() in
            # ultralytics/trackers/track.py calls model.add_callback(...),
            # not predictor.add_callback(...)). Those callbacks are what
            # actually assign box.id - and because they live on the MODEL,
            # not the predictor, resetting predictor alone doesn't remove
            # them: they keep firing on every later plain self.model(frame)
            # call too, which is why "bytetrack/botsort -> No Tracking"
            # kept showing IDs while "ocsort/deepsort -> No Tracking"
            # (which never call .track() at all, so never register these)
            # never had the problem. model.reset_callbacks() clears them
            # back to Ultralytics' defaults.
            self.model.predictor = None
            self.model.reset_callbacks()
            from assets.detection import custom_trackers
            custom_trackers.reset()
            self._last_tracker_param = tracker

        if tracker in CUSTOM_TRACKERS:
            return self._detect_custom_tracker(frame, tracker, confidence_threshold)

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

    def _detect_custom_tracker(self, frame, tracker, confidence_threshold):
        """OC-SORT/DeepSORT path: a plain (untracked) forward pass, same as
        the "no tracker" branch above, then boxmot assigns track IDs on top.
        Box-fallback and confidence filtering both mirror the bytetrack/
        botsort path above exactly, so behavior is identical either way."""
        from assets.detection import custom_trackers

        results = self.model(frame, verbose=False)
        boxes = results[0].boxes

        raw_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half = max(MIN_BOX_SIZE, min(FALLBACK_BOX_SIZE, MAX_BOX_SIZE)) // 2
                x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

            raw_boxes.append((x1, y1, x2, y2, conf, cls_id))

        tracked = custom_trackers.track(tracker, frame, raw_boxes)

        detections = []
        for t in tracked:
            if t["conf"] < confidence_threshold:
                continue
            detections.append({
                "box": t["box"],
                "conf": t["conf"],
                "label": self.model.names[t["cls_id"]],
                "track_id": t["track_id"],
            })
        return detections