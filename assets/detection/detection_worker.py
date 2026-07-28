"""
Detection Worker: runs model inference on a background thread. The GUI's
frame loop calls submit_frame() (non-blocking - just hands off the latest
frame) and get_latest_detections() (non-blocking - reads whatever the
worker most recently finished). If inference is slower than the video
feed, frames are naturally skipped (only the newest one is ever waiting)
rather than queuing up and making everything progressively laggier.
"""

import threading


class DetectionWorker:
    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_detections = []
        self._running = True
        self._new_frame = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame):
        with self._lock:
            self._latest_frame = frame
        self._new_frame.set()

    def get_latest_detections(self):
        with self._lock:
            return self._latest_detections

    def _run(self):
        while self._running:
            got_frame = self._new_frame.wait(timeout=0.5)
            if not got_frame:
                continue
            self._new_frame.clear()

            with self._lock:
                frame = self._latest_frame

            if frame is None:
                continue

            try:
                detections = self.model.detect(frame)
            except Exception:
                detections = []

            with self._lock:
                self._latest_detections = detections

    def stop(self):
        self._running = False