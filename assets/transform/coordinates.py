"""
Coordinate Transformation: pixel-only math, used for the plain-RGB
(no depth) case. The real-world angle/distance/step math for RealSense
lives in private/nozzle_targeting.py.
"""

import numpy as np


def pixel_distance(cx, cy, mx, my):
    """Straight-line pixel distance from center to a clicked/detected point."""
    return float(np.sqrt((mx - cx) ** 2 + (my - cy) ** 2))