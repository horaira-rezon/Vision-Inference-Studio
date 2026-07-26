"""
Camera Manager: finds what's actually plugged in, and builds the right
CameraSource for whatever the user picks.

Fixes from your terminal error:
  1. Scanning runs on a background thread (scan_async), so the Tkinter
     window no longer freezes while OpenCV probes each index.
  2. cv2.CAP_V4L2 is passed explicitly, skipping OpenCV's slower multi-
     backend fallback attempts (this is most of the noisy output you saw).
  3. Nodes that open but report zero frame width (like the metadata-only
     /dev/video1 companion node many webcams expose) are filtered out,
     since they aren't real capture streams.
"""

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