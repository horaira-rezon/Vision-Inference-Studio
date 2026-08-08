import torch
from models.base import VisionModel

class UNetModel(VisionModel):
    def __init__(self, weight_path, task="semantic_segmentation"):
        self.task = task
        self.weight_path = weight_path
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        metadata_names = checkpoint.get("class_names", checkpoint.get("names")) if isinstance(checkpoint, dict) else None
        if hasattr(checkpoint, "eval") and hasattr(checkpoint, "parameters"):
            self.model = checkpoint
        elif isinstance(checkpoint, dict) and hasattr(checkpoint.get("model"), "eval"):
            self.model = checkpoint["model"]
        else:
            raise RuntimeError("The U-Net weight must contain a serialized PyTorch model or a checkpoint with a 'model' object.")
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
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (tuple,list)):
            output = output[0]
        mask = output.argmax(1)[0].detach().cpu().numpy()
        return {"type":"semantic_segmentation","mask":mask,"class_names":self.names}
