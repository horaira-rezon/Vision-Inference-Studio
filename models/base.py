from abc import ABC, abstractmethod

class VisionModel(ABC):
    task = None

    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.names = {}

    @abstractmethod
    def infer(self, frame, confidence_threshold=0.0):
        raise NotImplementedError

    def close(self):
        pass
