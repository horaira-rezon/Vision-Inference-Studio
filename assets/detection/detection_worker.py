import threading

class DetectionWorker:
    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_tracker = "none"
        self._latest_confidence_threshold = 0.0
        self._latest_result = {}
        self._latest_error = None
        self._running = True
        self._new_frame = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame, tracker="none", confidence_threshold=0.0):
        with self._lock:
            self._latest_frame = frame
            self._latest_tracker = tracker
            self._latest_confidence_threshold = confidence_threshold
        self._new_frame.set()

    def get_latest_result(self):
        with self._lock:
            return self._latest_result

    def get_latest_error(self):
        with self._lock:
            return self._latest_error

    def _apply_generic_tracker(self, frame, result, tracker):
        if tracker not in ("ocsort", "deepocsort"):
            return result
        if result.get("type") == "detection":
            items = result.get("detections", [])
            if not items:
                return result
            from assets.detection import generic_trackers
            result["detections"] = generic_trackers.apply(tracker, frame, items)
        elif result.get("type") == "instance_segmentation":
            items = result.get("segments", [])
            if not items:
                return result
            from assets.detection import generic_trackers
            result["segments"] = generic_trackers.apply(tracker, frame, items)
        elif result.get("type") == "pose":
            items = result.get("poses", [])
            if not items:
                return result
            from assets.detection import generic_trackers
            result["poses"] = generic_trackers.apply(tracker, frame, items)
        return result

    def _run(self):
        while self._running:
            if not self._new_frame.wait(timeout=0.5):
                continue
            self._new_frame.clear()
            with self._lock:
                frame = self._latest_frame
                tracker = self._latest_tracker
                threshold = self._latest_confidence_threshold
            if frame is None:
                continue
            try:
                if hasattr(self.model, "infer_with_tracking"):
                    result = self.model.infer_with_tracking(frame, threshold, tracker)
                else:
                    result = self.model.infer(frame, threshold, tracker)
                result = self._apply_generic_tracker(frame, result, tracker)
                self._latest_error = None
            except Exception as exc:
                result = {}
                self._latest_error = str(exc)
            with self._lock:
                self._latest_result = result

    def stop(self):
        self._running = False
        self._new_frame.set()
