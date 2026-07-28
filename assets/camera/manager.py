"""
Camera Manager: finds what's actually plugged in, and builds the right
CameraSource for whatever the user picks. Camera selection behavior is
unchanged - only WHERE RealSenseSource's implementation lives changed
(private.realsense_camera instead of a public assets/camera/realsense.py).
"""

import threading
import cv2

from .webcam import WebcamSource

# private/ is .gitignored - fails gracefully so the camera picker just
# won't offer RealSense on a machine that doesn't have it
try:
    from my_version.realsense_camera import RealSenseSource, REALSENSE_AVAILABLE
except ImportError:
    RealSenseSource = None
    REALSENSE_AVAILABLE = False

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class CameraManager:
    def __init__(self, max_index=4):
        self.max_index = max_index

    def scan_async(self, on_complete):
        """Runs the hardware probe off the main thread.
        on_complete(options: list[str]) fires when done."""
        threading.Thread(target=self._scan, args=(on_complete,), daemon=True).start()

    def _scan(self, on_complete):
        options = []

        if REALSENSE_AVAILABLE:
            ctx = rs.context()
            if len(ctx.query_devices()) > 0:
                options.append("RealSense Depth Camera")

        for idx in range(self.max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                if width and width > 0:
                    options.append(f"Webcam {idx}")
            cap.release()

        on_complete(options)

    def build(self, choice):
        """Instantiates and starts the CameraSource for the given menu choice."""
        if choice == "RealSense Depth Camera":
            source = RealSenseSource()
        else:
            idx = int(choice.split(" ")[-1])
            source = WebcamSource(idx)
        source.start()
        return source