import cv2
from .base import CameraSource


class WebcamSource(CameraSource):
    """A plain RGB-only camera (laptop webcam or generic USB camera)."""

    def __init__(self, index):
        self.index = index
        self.cap = None

    def start(self):
        # Forcing the V4L2 backend explicitly (instead of letting OpenCV try
        # every backend in turn) removes most of the slow, noisy fallback
        # attempts you saw in the terminal.
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.index}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None
        h, w = frame.shape[:2]
        return frame, None, w // 2, h // 2

    def stop(self):
        if self.cap:
            self.cap.release()

    @property
    def has_depth(self):
        return False