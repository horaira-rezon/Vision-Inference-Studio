import numpy as np
import threading
from assets.camera.base import CameraSource, LatestFrameBuffer

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class RealSenseSource(CameraSource):
    def __init__(self):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed")
        self.pipeline = None
        self.align = None
        self.intrinsics = None
        self._buffer = LatestFrameBuffer()
        self._running = False
        self._thread = None

    def start(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(cfg)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        self.align = rs.align(rs.stream.color)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running and self.pipeline is not None:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data()).copy()
                cx, cy = int(self.intrinsics.ppx), int(self.intrinsics.ppy)
                self._buffer.publish((image, depth_frame, cx, cy))
            except Exception:
                if self._running:
                    continue
                break

    def read(self):
        value = self._buffer.read()
        if value is None:
            return None, None, None, None
        return value

    def get_depth_meters(self, x=None, y=None, depth_frame=None):
        if depth_frame is None:
            return None
        if x is None:
            x = int(self.intrinsics.ppx)
        if y is None:
            y = int(self.intrinsics.ppy)
        x = max(0, min(int(x), self.intrinsics.width - 1))
        y = max(0, min(int(y), self.intrinsics.height - 1))
        depth = float(depth_frame.get_distance(x, y))
        return depth if depth > 0 else None

    def stop(self):
        self._running = False
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._buffer.clear()

    @property
    def has_depth(self):
        return True
