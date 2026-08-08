"""
Detection Worker: runs model inference on a background thread. The GUI's
frame loop calls submit_frame() (non-blocking - just hands off the latest
frame, tracker mode, and confidence threshold together), get_latest_
detections() (non-blocking - reads whatever the worker most recently
finished), and get_latest_error() (non-blocking - reads the message from
the most recent failed detect() call, or None). If inference is slower
than the video feed, frames are naturally skipped (only the newest one is
ever waiting) rather than queuing up and making everything progressively
laggier.

tracker/confidence_threshold are bundled with the frame (rather than set
separately) so a given detect() call always uses the values that were
current at the moment that specific frame was submitted - no separate
shared state to race against.

Errors from detect() (e.g. a tracker failing to construct, like DeepOC-SORT's
ReID weights not loading) used to be swallowed silently here, which meant
a broken tracker just showed nothing forever with zero explanation.
get_latest_error() exists specifically so app.py can surface that as a
notice instead.
"""

import threading


class DetectionWorker:
    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_tracker = None
        self._latest_confidence_threshold = 0.0
        self._latest_detections = []
        self._latest_error = None
        self._running = True
        self._new_frame = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame, tracker=None, confidence_threshold=0.0):
        with self._lock:
            self._latest_frame = frame
            self._latest_tracker = tracker
            self._latest_confidence_threshold = confidence_threshold
        self._new_frame.set()

    def get_latest_detections(self):
        with self._lock:
            return self._latest_detections

    def get_latest_error(self):
        with self._lock:
            return self._latest_error

    def _run(self):
        while self._running:
            got_frame = self._new_frame.wait(timeout=0.5)
            if not got_frame:
                continue
            self._new_frame.clear()

            with self._lock:
                frame = self._latest_frame
                tracker = self._latest_tracker
                confidence_threshold = self._latest_confidence_threshold

            if frame is None:
                continue

            try:
                detections = self.model.detect(frame, tracker=tracker, confidence_threshold=confidence_threshold)
                error = None
            except Exception as e:
                detections = []
                error = str(e)

            with self._lock:
                self._latest_detections = detections
                self._latest_error = error

    def stop(self):
        self._running = False