"""
Coordinate Transformation: pure math, no camera or GUI dependencies, so it's
independently testable.
"""

import numpy as np


def pixel_distance(cx, cy, mx, my):
    """Straight-line pixel distance from center to a clicked point (RGB-only case)."""
    return float(np.sqrt((mx - cx) ** 2 + (my - cy) ** 2))


def point_to_angle_and_distance(point_3d):
    """Real-world (X, Y, Z) -> (diagonal_distance_m, angle_deg).
    angle_deg > 0 means the target is to the right, < 0 means left."""
    X, Y, Z = point_3d
    diagonal_distance = float(np.sqrt(X**2 + Y**2 + Z**2))
    angle_deg = float(np.degrees(np.arctan(X / Z)))
    return diagonal_distance, angle_deg