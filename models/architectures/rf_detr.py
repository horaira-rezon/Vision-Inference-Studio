from models.base import VisionModel

_TASK_TOKEN = {
    "instance_segmentation": "Seg",
    "pose": "Keypoint",
    "detection": None,
}


class RFDETRModel(VisionModel):
    def __init__(self, weight_path, task="detection"):
        try:
            from rfdetr.detr import RFDETR
        except ImportError as exc:
            raise RuntimeError("RF-DETR requires rfdetr. Install it with: pip install rfdetr==1.8.3") from exc
        self.task = task
        self.weight_path = weight_path
        # from_checkpoint reads the checkpoint's own model_name/args to pick the
        # right RF-DETR subclass and size (Nano/Small/Medium/Large/...) instead of
        # assuming a fixed size - a size mismatch would otherwise fail to load.
        self.model = RFDETR.from_checkpoint(weight_path)
        cls_name = type(self.model).__name__
        expected_token = _TASK_TOKEN.get(task)
        if task == "detection" and ("Seg" in cls_name or "Keypoint" in cls_name):
            raise RuntimeError(
                f"This weight is an RF-DETR {cls_name} checkpoint, not a plain "
                f"detection model. Select the matching vision task instead."
            )
        if expected_token and expected_token not in cls_name:
            raise RuntimeError(
                f"This weight is an RF-DETR {cls_name} checkpoint, which doesn't "
                f"match the selected vision task. Pick the vision task that matches "
                f"what this weight was trained for."
            )
        self.names = getattr(self.model, "class_names", None) or {}

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        detections = self.model.predict(frame, threshold=confidence_threshold)
        boxes = getattr(detections, "xyxy", None)
        scores = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)
        masks = getattr(detections, "mask", None)
        keypoints = getattr(detections, "keypoints", None)
        data = getattr(detections, "data", {}) or {}
        class_names_per_item = data.get("class_name")
        if boxes is None or len(boxes) == 0:
            if self.task == "instance_segmentation":
                return {"type": "instance_segmentation", "segments": []}
            if self.task == "pose":
                return {"type": "pose", "poses": []}
            return {"type": "detection", "detections": []}
        items = []
        for i, box in enumerate(boxes):
            conf = float(scores[i]) if scores is not None else 1.0
            cls_id = int(class_ids[i]) if class_ids is not None else 0
            if class_names_per_item is not None:
                label = str(class_names_per_item[i])
            elif isinstance(self.names, dict) and self.names:
                label = self.names.get(cls_id, str(cls_id))
            else:
                label = str(cls_id)
            item = {"box": tuple(int(v) for v in box), "conf": conf, "label": label, "class_id": cls_id, "track_id": None}
            if self.task == "instance_segmentation" and masks is not None:
                item["mask"] = masks[i]
            if self.task == "pose" and keypoints is not None:
                item["keypoints"] = keypoints[i]
            items.append(item)
        if self.task == "instance_segmentation":
            return {"type": "instance_segmentation", "segments": items}
        if self.task == "pose":
            return {"type": "pose", "poses": items}
        return {"type": "detection", "detections": items}
