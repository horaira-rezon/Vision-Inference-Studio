import numpy as np
from models.base import VisionModel

class YOLOModel(VisionModel):
    def __init__(self, weight_path, task):
        from ultralytics import YOLO
        self.model = YOLO(weight_path)
        self.task = task
        self.weight_path = weight_path
        self.names = self.model.names
        self._last_tracker = "__unset__"
        from assets.detection import custom_trackers
        custom_trackers.reset()

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        if tracker != self._last_tracker:
            self.model.predictor = None
            self.model.reset_callbacks()
            from assets.detection import custom_trackers
            custom_trackers.reset()
            self._last_tracker = tracker
        if self.task == "classification":
            results = self.model(frame, verbose=False)
            probs = getattr(results[0], "probs", None)
            if probs is None:
                return {"type": "classification", "classes": []}
            items = []
            top = min(5, len(probs.top5))
            for i in range(top):
                cls_id = int(probs.top5[i])
                conf = float(probs.top5conf[i])
                if conf >= confidence_threshold:
                    items.append({"label": self.names[cls_id], "conf": conf})
            return {"type": "classification", "classes": items}
        if self.task == "pose":
            results = self.model.track(frame, persist=True, tracker=tracker + ".yaml", verbose=False) if tracker in ("bytetrack","botsort") else self.model(frame, verbose=False)
            return self._pose_results(results, confidence_threshold, tracker)
        if self.task == "instance_segmentation":
            results = self.model.track(frame, persist=True, tracker=tracker + ".yaml", verbose=False) if tracker in ("bytetrack","botsort") else self.model(frame, verbose=False)
            return self._segmentation_results(results, confidence_threshold, tracker)
        results = self.model.track(frame, persist=True, tracker=tracker + ".yaml", verbose=False) if tracker in ("bytetrack","botsort") else self.model(frame, verbose=False)
        return self._detection_results(results, confidence_threshold, tracker)

    def _detection_results(self, results, threshold, tracker):
        boxes = getattr(results[0], "boxes", None)
        detections = []
        if boxes is None:
            return {"type": "detection", "detections": detections}
        for box in boxes:
            conf = float(box.conf[0])
            if conf < threshold:
                continue
            xyxy = [int(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
            detections.append({"box": tuple(xyxy), "conf": conf, "label": self.names[cls_id], "track_id": track_id})
        return {"type": "detection", "detections": detections}

    def _segmentation_results(self, results, threshold, tracker):
        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        items = []
        if boxes is None:
            return {"type": "instance_segmentation", "segments": []}
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
            mask = None
            if masks is not None and i < len(masks.data):
                mask = masks.data[i].detach().cpu().numpy()
            items.append({"box": (x1,y1,x2,y2), "conf": conf, "label": self.names[cls_id], "track_id": track_id, "mask": mask})
        return {"type": "instance_segmentation", "segments": items}

    def _pose_results(self, results, threshold, tracker):
        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        items = []
        if boxes is None:
            return {"type": "pose", "poses": []}
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if getattr(box, "id", None) is not None else None
            points = None
            if keypoints is not None and i < len(keypoints.xy):
                points = keypoints.xy[i].detach().cpu().numpy().tolist()
            items.append({"box": (x1,y1,x2,y2), "conf": conf, "label": self.names[cls_id], "track_id": track_id, "keypoints": points})
        return {"type": "pose", "poses": items}

    def _name_to_id(self, name):
        for k, v in self.names.items():
            if v == name:
                return int(k)
        return 0

    def infer_with_tracking(self, frame, confidence_threshold=0.0, tracker=None):
        return self.infer(frame, confidence_threshold, tracker)

    def close(self):
        self.model = None
