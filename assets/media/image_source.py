import cv2

class ImageFileSource:
    def __init__(self, path):
        self.path = path
        self.image = None
        self.current_index = 0
        self.frame_count = 1
        self.paused = True

    def start(self):
        self.image = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Could not open image file: {self.path}")

    def read(self):
        if self.image is None:
            return None, None, None, None
        return self.image.copy(), None, None, None

    def seek(self, index):
        return self.image.copy() if self.image is not None else None

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def stop(self):
        self.image = None

    @property
    def has_depth(self):
        return False

    @property
    def is_file(self):
        return True
