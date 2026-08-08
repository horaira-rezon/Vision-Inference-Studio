from models.architectures.yolo import YOLOModel
from models.architectures.sam import SAMModel
from models.architectures.rf_detr import RFDETRModel
from models.architectures.efficientnet import EfficientNetModel
from models.architectures.resnet import ResNetModel
from models.architectures.rcnn import RCNNModel
from models.architectures.faster_rcnn import FasterRCNNModel
from models.architectures.unet import UNetModel

def create_model(task, architecture, weight_path):
    if architecture == "yolo":
        return YOLOModel(weight_path, task)
    if architecture == "sam":
        return SAMModel(weight_path, task)
    if architecture == "rf_detr":
        return RFDETRModel(weight_path, task)
    if architecture == "efficientnet":
        return EfficientNetModel(weight_path, task)
    if architecture == "resnet":
        return ResNetModel(weight_path, task)
    if architecture == "rcnn":
        return RCNNModel(weight_path, task)
    if architecture == "faster_rcnn":
        return FasterRCNNModel(weight_path, task)
    if architecture == "unet":
        return UNetModel(weight_path, task)
    raise ValueError(f"Unsupported model architecture: {architecture}")
