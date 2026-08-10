from models.base import VisionModel

class SAMModel(VisionModel):
    def __init__(self, weight_path, task="instance_segmentation"):
        from ultralytics import SAM
        self.model = SAM(weight_path)
        self.task = task
        self.weight_path = weight_path
        self.names = {0: "object"}

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        try:
            results = self.model(frame, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                "SAM failed to run prompt-free automatic segmentation on this frame. "
                "This SAM checkpoint/version may require explicit point or box prompts, "
                "which this live-camera pipeline doesn't provide."
            ) from exc
        result = results[0]
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        segments = []
        if masks is None:
            return {"type":"instance_segmentation","segments":[]}
        for i in range(len(masks.data)):
            mask = masks.data[i].detach().cpu().numpy()
            if boxes is not None and i < len(boxes):
                conf = float(boxes.conf[i])
                box = tuple(int(v) for v in boxes.xyxy[i])
            else:
                conf = 1.0
                ys, xs = mask.nonzero()
                box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else (0,0,0,0)
            if conf >= confidence_threshold:
                segments.append({"box":box,"conf":conf,"label":"object","class_id":0,"track_id":None,"mask":mask})
        return {"type":"instance_segmentation","segments":segments}
