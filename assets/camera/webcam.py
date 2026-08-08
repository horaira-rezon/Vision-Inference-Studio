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

        # Most USB webcams default to an uncompressed (YUYV) capture mode,
        # which is USB-bandwidth-limited and commonly caps out around
        # 10-15fps once you're above a small resolution - this matches
        # "stuck around 10fps" exactly. MJPG is still a real per-frame
        # JPEG (no quality loss beyond normal JPEG compression), just far
        # smaller over the wire, which is what actually lets the same
        # hardware run at its higher advertised framerates. FOURCC has to
        # be set before FPS/resolution: V4L2 only advertises the higher
        # rates for the modes that support them (this one), so setting it
        # after can end up silently ignored by the driver.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Only ever keep the newest frame waiting rather than letting a
        # backlog build up if something downstream is briefly slow - same
        # "always work with the latest thing" principle DetectionWorker
        # already follows for inference.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap.set() calls above are best-effort: on hardware/drivers that
        # don't support a given mode they simply fail and leave whatever
        # was already negotiated in place - never raises, never crashes.

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