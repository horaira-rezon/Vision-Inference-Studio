"""
PRIVATE - do not publish / .gitignore this folder.

RealSense-specific camera handling: streaming setup, intrinsics, and the
pixel + depth -> real-world (X, Y, Z) deprojection that private/nozzle_targeting.py
consumes. Moved out of assets/camera/ unchanged, so the fact that a
RealSense depth camera and its intrinsics are being used for real-world
targeting isn't part of the published codebase.
"""

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
        # Center dot/axes use the true optical center (camera intrinsic),
        # not the geometric middle of the frame.
        cx, cy = int(self.intrinsics.ppx), int(self.intrinsics.ppy)
        return image, depth_frame, cx, cy

    def deproject(self, x, y, depth_frame):
        """Pixel (x, y) + depth -> real-world (X, Y, Z) in meters, or None
        if there's no valid depth reading at that pixel."""
        z = depth_frame.get_distance(x, y)
        if z <= 0:
            return None
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], z)

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

    @property
    def has_depth(self):
        return True