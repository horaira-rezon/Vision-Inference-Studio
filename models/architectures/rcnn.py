from models.architectures.faster_rcnn import FasterRCNNModel


class RCNNModel(FasterRCNNModel):
    """torchvision has no standalone 'R-CNN' checkpoint format (the original
    R-CNN pipeline used external region proposals + separate per-region SVM
    classifiers, not one end-to-end .pt/.pth file), so this loads the same
    Faster R-CNN detector as the 'Faster R-CNN' option. Weights trained with
    a genuinely different R-CNN implementation will not load here."""

    def __init__(self, weight_path, task="detection"):
        super().__init__(weight_path, task="detection")
