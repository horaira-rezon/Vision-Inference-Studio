import threading
import cv2
from .webcam import WebcamSource
from .realsense import RealSenseSource, REALSENSE_AVAILABLE

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

class CameraManager:
    def __init__(self, max_index=4):
        self.max_index = max_index

    def scan_async(self, on_complete):
        threading.Thread(target=self._scan, args=(on_complete,), daemon=True).start()

    def _scan(self, on_complete):
        options = []
        if REALSENSE_AVAILABLE and rs is not None:
            try:
                if len(rs.context().query_devices()) > 0:
                    options.append("RealSense Depth Camera")
            except Exception:
                pass
        for idx in range(self.max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                if width and width > 0:
                    options.append(f"Webcam {idx}")
            cap.release()
        on_complete(options)

    def build(self, choice):
        if choice == "RealSense Depth Camera":
            source = RealSenseSource()
        else:
            idx = int(choice.split(" ")[-1])
            source = WebcamSource(idx)
        source.start()
        return source
