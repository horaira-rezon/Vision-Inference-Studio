import torch
from models.base import VisionModel
from models.architectures._weights import safe_torch_load, extract_state_and_names


class FasterRCNNModel(VisionModel):
    def __init__(self, weight_path, task="detection"):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        self.task = "instance_segmentation" if task == "instance_segmentation" else "detection"
        self.weight_path = weight_path
        checkpoint = safe_torch_load(weight_path)
        state, metadata_names = extract_state_and_names(checkpoint)
        box_key = "roi_heads.box_predictor.cls_score.weight"
        if not isinstance(state, dict) or box_key not in state:
            raise RuntimeError(
                "This weight file doesn't look like a torchvision Faster R-CNN / "
                "Mask R-CNN state_dict (missing 'roi_heads.box_predictor.cls_score.weight'). "
                "Make sure the selected weight was trained with this architecture."
            )
        num_classes = state[box_key].shape[0]

        if self.task == "instance_segmentation":
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            self.model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            mask_key = "roi_heads.mask_predictor.mask_fcn_logits.weight"
            if mask_key in state:
                mask_in = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
                self.model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_classes)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{'Mask R-CNN' if self.task == 'instance_segmentation' else 'Faster R-CNN'} "
                f"weight did not fully match the model architecture (missing={len(missing)}, "
                f"unexpected={len(unexpected)}). The weight file may not match this architecture."
            )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if metadata_names:
            self.names = metadata_names
        else:
            self.names = {i: str(i) for i in range(num_classes)}

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        import cv2, numpy as np
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
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
            cls_id = int(labels[i])
            label = self.names.get(cls_id, str(cls_id))
            item = {"box": box, "conf": conf, "label": label, "class_id": cls_id, "track_id": None}
            if masks is not None:
                item["mask"] = masks[i, 0].detach().cpu().numpy()
            items.append(item)
        kind = "instance_segmentation" if masks is not None else "detection"
        return {"type": kind, "detections": items, "segments": items}
