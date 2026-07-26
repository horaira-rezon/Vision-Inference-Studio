"""
Common interface every camera type must implement, so the GUI never needs
to know or care whether it's talking to a webcam or a RealSense unit.
"""

from abc import ABC, abstractmethod


class CameraSource(ABC):

    @abstractmethod
    def start(self):
        """Open the physical device / pipeline."""

    @abstractmethod
    def read(self):
        """Returns (color_image, depth_frame_or_None, center_x, center_y)."""

    @abstractmethod
    def stop(self):
        """Release the device cleanly."""

    @property
    @abstractmethod
    def has_depth(self):
        """True if this source can provide real depth data."""