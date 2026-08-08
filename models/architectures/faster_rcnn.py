import torch
from models.base import VisionModel

class FasterRCNNModel(VisionModel):
    def __init__(self, weight_path, task="detection"):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        self.task = "instance_segmentation" if task == "instance_segmentation" else "detection"
        self.weight_path = weight_path
        if self.task == "instance_segmentation":
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            self.model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k,v in state.items()}
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        metadata_names = checkpoint.get("class_names", checkpoint.get("names")) if isinstance(checkpoint, dict) else None
        if isinstance(metadata_names, (list, tuple)):
            self.names = {i: str(v) for i, v in enumerate(metadata_names)}
        elif isinstance(metadata_names, dict):
            self.names = {int(k): str(v) for k, v in metadata_names.items()}
        else:
            self.names = {}

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        import cv2, numpy as np
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2,0,1).to(self.device)
        with torch.no_grad():
            result = self.model([tensor])[0]
        items = []
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")
        masks = result.get("masks")
        for i in range(len(boxes)):
            conf = float(scores[i])
            if conf < confidence_threshold:
                continue
            box = tuple(int(v) for v in boxes[i].detach().cpu().tolist())
            label = self.names.get(int(labels[i]), str(int(labels[i])))
            item = {"box":box,"conf":conf,"label":label,"track_id":None}
            if masks is not None:
                item["mask"] = masks[i,0].detach().cpu().numpy()
            items.append(item)
        kind = "instance_segmentation" if masks is not None else "detection"
        return {"type":kind, "detections":items, "segments":items}
