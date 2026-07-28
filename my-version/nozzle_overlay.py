"""
PRIVATE - do not publish / .gitignore this folder.

The full RealSense targeting overlay: computing the target from a pixel,
sending the nozzle command, and drawing the marker/line/readout text.
gui/app.py only calls render() with a target pixel - it no longer knows
what gets drawn or computed for the depth-camera + nozzle case at all.
"""

import cv2
from assets.visualization import overlay


def render(image, camera_source, depth_frame, cx, cy, target_x, target_y, nozzle, arduino_connection):
    """
    image            - the frame to draw on (mutated in place)
    camera_source    - the active CameraSource (must have .deproject)
    depth_frame      - current RealSense depth frame
    cx, cy           - intrinsic center (axes origin)
    target_x, target_y - pixel being targeted (mouse click OR detection centroid)
    nozzle           - a NozzleTargeting instance (from nozzle_targeting.py)
    arduino_connection - raw pyserial connection, or None if not connected
    """
    point = camera_source.deproject(target_x, target_y, depth_frame)

    if point is None:
        overlay.draw_click_marker(image, cx, cy, target_x, target_y, color=(0, 0, 255))
        cv2.putText(image, "No Depth Data", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return

    diag, angle, direction, steps = nozzle.compute(point)
    nozzle.send(arduino_connection, steps)

    overlay.draw_click_marker(image, cx, cy, target_x, target_y)
    lines = [
        (f"Diag Dist: {diag:.3f} m", (0, 255, 0)),
        (f"Target Ang: {angle:.1f} deg ({direction})", (0, 255, 0)),
        (f"Steps to Move: {steps}", (0, 255, 0)),
        (f"Nozzle At: {nozzle.current_nozzle_angle:.1f} deg", (0, 255, 0)),
    ]
    overlay.draw_text_lines(image, lines)