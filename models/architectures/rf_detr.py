from models.base import VisionModel

class RFDETRModel(VisionModel):
    def __init__(self, weight_path, task="detection"):
        try:
            if task == "instance_segmentation":
                from rfdetr import RFDETRSegMedium as ModelClass
            elif task == "pose":
                from rfdetr import RFDETRKeypointPreview as ModelClass
            else:
                from rfdetr import RFDETRMedium as ModelClass
        except ImportError as exc:
            raise RuntimeError("RF-DETR requires rfdetr. Install it with: pip install rfdetr==1.8.3") from exc
        self.task = task
        self.weight_path = weight_path
        self.model = ModelClass(pretrain_weights=weight_path)
        self.names = getattr(self.model, "class_names", {})

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        detections = self.model.predict(frame, threshold=confidence_threshold)
        boxes = getattr(detections, "xyxy", None)
        scores = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)
        masks = getattr(detections, "mask", None)
        keypoints = getattr(detections, "keypoints", None)
        items = []
        if boxes is None:
            return {"type": "instance_segmentation" if self.task == "instance_segmentation" else "detection", "segments": [] if self.task == "instance_segmentation" else None, "detections": []}
        for i, box in enumerate(boxes):
            conf = float(scores[i]) if scores is not None else 1.0
            cls_id = int(class_ids[i]) if class_ids is not None else 0
            label = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else str(cls_id)
            item = {"box": tuple(int(v) for v in box), "conf": conf, "label": label, "track_id": None}
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
