import cv2

class VideoFileSource:
    def __init__(self, path):
        self.path = path
        self.capture = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_index = -1
        self.paused = False

    def start(self):
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video file: {self.path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 0:
            self.fps = 30.0

    def read(self):
        if self.capture is None:
            return None, None, None, None
        if self.paused and self.current_index >= 0:
            pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            if int(pos) != self.current_index:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_index)
            ok, frame = self.capture.read()
            if not ok:
                return None, None, None, None
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_index)
            return frame, None, None, None
        ok, frame = self.capture.read()
        if not ok:
            self.paused = True
            if self.frame_count:
                self.current_index = self.frame_count - 1
            return None, None, None, None
        self.current_index = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        return frame, None, None, None

    def seek(self, index):
        if self.capture is None:
            return None
        index = max(0, min(int(index), max(0, self.frame_count - 1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        self.current_index = index
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.current_index = index
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index + 1)
        return frame

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @property
    def has_depth(self):
        return False

    @property
    def is_file(self):
        return True
