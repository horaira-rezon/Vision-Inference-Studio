from abc import ABC, abstractmethod
import threading

class LatestFrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._sequence = 0
        self._read_sequence = 0

    def publish(self, value):
        with self._lock:
            self._frame = value
            self._sequence += 1

    def read(self):
        with self._lock:
            if self._frame is None or self._sequence == self._read_sequence:
                return None
            value = self._frame
            self._read_sequence = self._sequence
            return value

    def clear(self):
        with self._lock:
            self._frame = None
            self._sequence = 0
            self._read_sequence = 0

class CameraSource(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @property
    @abstractmethod
    def has_depth(self):
        pass
