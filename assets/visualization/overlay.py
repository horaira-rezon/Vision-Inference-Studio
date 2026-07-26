"""
Visualization: every OpenCV drawing call lives here, shared by both the
depth-camera and plain-webcam code paths so the drawing logic exists once.
"""

import cv2


def draw_axes(image, cx, cy):
    h, w = image.shape[:2]
    cv2.line(image, (0, cy), (w, cy), (100, 100, 100), 1)
    cv2.line(image, (cx, 0), (cx, h), (100, 100, 100), 1)


def draw_click_marker(image, cx, cy, mx, my, color=(0, 255, 0)):
    cv2.line(image, (cx, cy), (mx, my), color, 1)
    cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
    cv2.circle(image, (mx, my), 3, (0, 0, 255), -1)


def draw_text_lines(image, lines, origin=(10, 20), gap=20):
    """lines: list of (text, bgr_color) tuples, stacked top to bottom."""
    x, y = origin
    for i, (text, color) in enumerate(lines):
        cv2.putText(image, text, (x, y + i * gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_detection_box(image, box, label, conf):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)