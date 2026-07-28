"""
Visualization: every OpenCV drawing call lives here, shared by both the
depth-camera and plain-webcam code paths. All functions accept an optional
`scale` so the SAME call can draw crisply at native camera resolution
(for recording) or at a larger display resolution (for on-screen viewing) -
the app draws each frame twice, once per resolution, rather than drawing
once and stretching a rasterized result.
"""

import cv2

TECH_ACCENT = (255, 210, 40)   # cyan-ish (BGR) - boxes, axes, center point
TECH_TARGET = (0, 165, 255)    # amber (BGR) - target/centroid reticles


def draw_axes(image, cx, cy, color=(100, 100, 100), thickness=1):
    h, w = image.shape[:2]
    cv2.line(image, (0, cy), (w, cy), color, thickness)
    cv2.line(image, (cx, 0), (cx, h), color, thickness)


def draw_reticle(image, x, y, color, scale=1.0):
    """Small circle + four corner ticks - a targeting-scope look, used for
    both detection centroids and click/target points."""
    r = max(4, round(7 * scale))
    tick = max(3, round(5 * scale))
    thickness = max(1, round(2 * scale))
    cv2.circle(image, (x, y), r, color, thickness, cv2.LINE_AA)
    cv2.line(image, (x - r - tick, y), (x - r + 2, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x + r - 2, y), (x + r + tick, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - r - tick), (x, y - r + 2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y + r - 2), (x, y + r + tick), color, thickness, cv2.LINE_AA)


def draw_click_marker(image, cx, cy, mx, my, color=None, scale=1.0):
    """Center point + line + target reticle. `color` overrides both (used
    for the red 'no depth data' state); otherwise uses the default
    accent/target color pair."""
    line_color = color if color is not None else TECH_ACCENT
    center_color = color if color is not None else TECH_ACCENT
    target_color = color if color is not None else TECH_TARGET

    thickness = max(1, round(1 * scale))
    r = max(3, round(4 * scale))

    cv2.line(image, (cx, cy), (mx, my), line_color, thickness, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), r, center_color, -1, cv2.LINE_AA)
    draw_reticle(image, mx, my, target_color, scale)


def draw_centroid_marker(image, cx, cy, color=None, scale=1.0):
    """Reticle marker for EVERY detected object's centroid (not just the
    primary target)."""
    draw_reticle(image, cx, cy, color if color is not None else TECH_TARGET, scale)


def draw_text_lines(image, lines, origin=(10, 24), gap=None, scale=1.0):
    """lines: list of (text, bgr_color) tuples, stacked top to bottom.
    Font size, thickness, and line spacing all scale with `scale` so text
    stays legible (not tiny) when drawn on a larger display frame."""
    font_scale = 0.5 * scale
    thickness = max(1, round(2 * scale))
    line_gap = gap if gap is not None else int(22 * scale)
    x, y = int(origin[0] * scale), int(origin[1] * scale)
    for i, (text, color) in enumerate(lines):
        cv2.putText(image, text, (x, y + i * line_gap), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


def draw_detection_box(image, box, label, conf, scale=1.0):
    """Techy corner-bracket style instead of a full rectangle outline, plus
    a filled label pill (like a targeting HUD) instead of plain text."""
    x1, y1, x2, y2 = box
    color = TECH_ACCENT
    thickness = max(1, round(2 * scale))
    corner_len = max(8, int(min(x2 - x1, y2 - y1) * 0.2))

    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(image, (px, py), (px + dx * corner_len, py), color, thickness, cv2.LINE_AA)
        cv2.line(image, (px, py), (px, py + dy * corner_len), color, thickness, cv2.LINE_AA)

    label_text = f"{label} {conf:.2f}"
    font_scale = 0.5 * scale
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = max(3, round(4 * scale))
    cv2.rectangle(image, (x1, y1 - th - 2 * pad), (x1 + tw + 2 * pad, y1), color, -1)
    cv2.putText(image, label_text, (x1 + pad, y1 - pad), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (20, 20, 20), thickness, cv2.LINE_AA)