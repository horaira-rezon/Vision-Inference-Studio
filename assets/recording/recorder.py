"""
Recording: video capture (cv2.VideoWriter) and screenshots. Folder selection
itself is handled by the GUI (it owns the file dialogs); this class just
does the actual writing once a folder has been chosen, and never mixes
video output with screenshot output.
"""

import os
import cv2
from datetime import datetime


class Recorder:
    def __init__(self, fps=20):
        self.writer = None
        self.recording = False
        self.fps = fps

    def start_recording(self, frame, video_dir):
        h, w = frame.shape[:2]
        filename = datetime.now().strftime("recording_%Y%m%d_%H%M%S.mp4")
        path = os.path.join(video_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        self.recording = True
        return path

    def write_frame(self, frame):
        if self.recording and self.writer is not None:
            self.writer.write(frame)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.recording = False

    def save_screenshot(self, frame, screenshot_dir):
        filename = datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
        path = os.path.join(screenshot_dir, filename)
        cv2.imwrite(path, frame)
        return path