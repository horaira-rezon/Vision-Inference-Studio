import torch
from models.base import VisionModel

class EfficientNetModel(VisionModel):
    def __init__(self, weight_path, task="classification"):
        from torchvision.models import efficientnet_b0
        self.task = task
        self.weight_path = weight_path
        self.model = efficientnet_b0(weights=None)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        metadata_names = checkpoint.get("class_names", checkpoint.get("names")) if isinstance(checkpoint, dict) else None
        if isinstance(metadata_names, (list, tuple)):
            self.names = {i: str(v) for i, v in enumerate(metadata_names)}
        elif isinstance(metadata_names, dict):
            self.names = {int(k): str(v) for k, v in metadata_names.items()}
        else:
            self.names = {i: str(i) for i in range(1000)}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        import cv2, numpy as np
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(5, probs.numel()))
        classes = [{"label": self.names.get(int(i), str(int(i))), "conf": float(v)} for v,i in zip(values,indices) if float(v) >= confidence_threshold]
        return {"type":"classification","classes":classes}
