import numpy as np
from assets.camera.base import CameraSource

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

    def start(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(cfg)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        self.align = rs.align(rs.stream.color)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None, None
        image = np.asanyarray(color_frame.get_data())
        cx, cy = int(self.intrinsics.ppx), int(self.intrinsics.ppy)
        return image, depth_frame, cx, cy

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
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None

    @property
    def has_depth(self):
        return True
