import torch
from models.base import VisionModel


class UNetModel(VisionModel):
    def __init__(self, weight_path, task="semantic_segmentation"):
        self.task = task
        self.weight_path = weight_path
        self.model = None
        metadata_names = None
        try:
            self.model = torch.jit.load(weight_path, map_location="cpu")
        except Exception:
            self.model = None
        if self.model is None:
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            if hasattr(checkpoint, "eval") and hasattr(checkpoint, "parameters"):
                self.model = checkpoint
            elif isinstance(checkpoint, dict) and hasattr(checkpoint.get("model"), "eval"):
                self.model = checkpoint["model"]
                metadata_names = checkpoint.get("class_names", checkpoint.get("names"))
            else:
                raise RuntimeError(
                    "U-Net weights must be a TorchScript module (torch.jit.save) or "
                    "a checkpoint containing a full serialized PyTorch model object "
                    "(torch.save(model, path) or {'model': model, ...}). A bare "
                    "state_dict can't be loaded here because U-Net has no fixed "
                    "torchvision architecture to reconstruct it against - export your "
                    "trained model with torch.jit.script(model).save(path) (recommended, "
                    "safer to load) or torch.save(model, path) instead of "
                    "torch.save(model.state_dict(), path)."
                )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        if isinstance(metadata_names, (list, tuple)):
            self.names = {i: str(v) for i, v in enumerate(metadata_names)}
        elif isinstance(metadata_names, dict):
            self.names = {int(k): str(v) for k, v in metadata_names.items()}
        else:
            self.names = {}

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        import cv2, numpy as np
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.dim() == 4 and output.shape[1] == 1:
            mask = (torch.sigmoid(output)[0, 0] >= 0.5).long().detach().cpu().numpy()
        else:
            mask = output.argmax(1)[0].detach().cpu().numpy()
        return {"type": "semantic_segmentation", "mask": mask, "class_names": self.names}
