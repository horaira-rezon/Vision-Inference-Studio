import cv2
import threading
import time
from .base import CameraSource, LatestFrameBuffer

class WebcamSource(CameraSource):
    def __init__(self, index):
        self.index = index
        self.cap = None
        self._buffer = LatestFrameBuffer()
        self._running = False
        self._thread = None
        self.capture_fps = 0.0
        self.requested_fps = 60
        self._frame_counter = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)) or "?"
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[WebcamSource] negotiated: fourcc={fourcc_str} fps={actual_fps} resolution={actual_w}x{actual_h}")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        last_time = None
        fps = 0.0
        while self._running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue
            now = time.perf_counter()
            if last_time is not None:
                dt = now - last_time
                if dt > 0:
                    inst = 1.0 / dt
                    fps = inst if fps == 0.0 else fps * 0.85 + inst * 0.15
            last_time = now
            self.capture_fps = fps
            self._frame_counter += 1
            h, w = frame.shape[:2]
            self._buffer.publish((frame, None, w // 2, h // 2))

    def read(self):
        value = self._buffer.read()
        if value is None:
            return None, None, None, None
        return value

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()
        self.cap = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._buffer.clear()
        self.capture_fps = 0.0
        self._frame_counter = 0

    @property
    def has_depth(self):
        return False
