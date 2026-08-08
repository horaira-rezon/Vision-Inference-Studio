from models.architectures.faster_rcnn import FasterRCNNModel

class RCNNModel(FasterRCNNModel):
    def __init__(self, weight_path, task="detection"):
        super().__init__(weight_path, task="detection")
