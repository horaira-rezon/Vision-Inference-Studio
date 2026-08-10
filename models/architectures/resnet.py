import torch
from models.base import VisionModel
from models.architectures._weights import safe_torch_load, extract_state_and_names

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetModel(VisionModel):
    def __init__(self, weight_path, task="classification"):
        from torchvision.models import resnet50
        self.task = task
        self.weight_path = weight_path
        checkpoint = safe_torch_load(weight_path)
        state, metadata_names = extract_state_and_names(checkpoint)
        if not isinstance(state, dict) or "fc.weight" not in state:
            raise RuntimeError(
                "This weight file doesn't look like a ResNet-50 state_dict "
                "(missing 'fc.weight'). Make sure the selected weight was "
                "trained with this architecture."
            )
        num_classes = state["fc.weight"].shape[0]
        self.model = resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"ResNet weight did not fully match the model architecture "
                f"(missing={len(missing)}, unexpected={len(unexpected)}). The weight "
                f"file may not be a ResNet-50 checkpoint."
            )
        self.model.eval()
        if metadata_names:
            self.names = metadata_names
        else:
            self.names = {i: str(i) for i in range(num_classes)}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def infer(self, frame, confidence_threshold=0.0, tracker=None):
        import cv2, numpy as np
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(5, probs.numel()))
        classes = [{"label": self.names.get(int(i), str(int(i))), "conf": float(v)} for v, i in zip(values, indices) if float(v) >= confidence_threshold]
        return {"type": "classification", "classes": classes}
