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
TARGET_DOT = (0, 0, 255)       # simple red dot (BGR) - click targets & centroids


def draw_axes(image, cx, axis_y, thickness=1, color=(100, 100, 100)):
    """Full-width/height crosshair. `cx` is always the true camera/frame
    center column and never moves. `axis_y` is the row the horizontal line
    is drawn on - normally the true center row, but the X-Axis slider (see
    config_window.py) can move it up/down independently, which is why this
    takes its own parameter instead of reusing a single (cx, cy) pair."""
    h, w = image.shape[:2]
    cv2.line(image, (0, axis_y), (w, axis_y), color, thickness)
    cv2.line(image, (cx, 0), (cx, h), color, thickness)


def draw_fixed_center_marker(image, cx, cy, scale=1.0, color=(0, 165, 255)):
    """The true camera-center dot, drawn in a distinct color and never
    moved - only shown once the X-Axis slider has actually shifted the
    crosshair's horizontal line away from this point, so the user can
    still see where the untouched center was."""
    r = max(3, round(4 * scale))
    cv2.circle(image, (cx, cy), r, color, -1, cv2.LINE_AA)


def draw_click_marker(image, cx, cy, mx, my, box=None, color=None, scale=1.0):
    """Center point + line + a simple filled dot at the target - no
    hollow reticle, just a plain dot like the original version.

    When `box` is given, the line always points toward the target/centroid,
    but only the portion from the image center up to the box border is
    drawn - the portion that would fall INSIDE the box (border to centroid)
    is skipped entirely rather than drawn-then-covered. Combined with the
    fact that this is called BEFORE draw_detection_box in the render plan
    (see app.py), the box's corner brackets and label pill always end up
    on top of whatever tiny sliver of line remains near the border, so the
    line reads as sitting behind/blending with the box and its label.
    """
    line_color = color if color is not None else TECH_ACCENT
    center_color = color if color is not None else TECH_ACCENT
    target_color = color if color is not None else TARGET_DOT
    thickness = max(1, round(1 * scale))
    r = max(3, round(4 * scale))
    end_x, end_y = mx, my

    if box is not None:
        x1, y1, x2, y2 = box

        # cv2.clipLine clips the (cx,cy)->(mx,my) segment against the box
        # rectangle and returns whatever portion of it lies INSIDE that
        # rectangle - i.e. exactly the part we want to omit. Whichever of
        # the two returned points is nearer to (cx, cy) is where the line
        # first enters the box; that's where the visible line should stop.
        # Using the library's clip instead of hand-rolled per-edge math
        # also avoids missed/degenerate cases (vertical or horizontal
        # lines, the center point already sitting inside the box, etc.).
        inside, p1, p2 = cv2.clipLine((x1, y1, x2 - x1, y2 - y1), (cx, cy), (mx, my))

        if inside:
            d1 = (p1[0] - cx) ** 2 + (p1[1] - cy) ** 2
            d2 = (p2[0] - cx) ** 2 + (p2[1] - cy) ** 2
            entry_x, entry_y = p1 if d1 <= d2 else p2

            gap = max(4, round(5 * scale))
            vx = entry_x - cx
            vy = entry_y - cy
            length = (vx * vx + vy * vy) ** 0.5

            if length > 0:
                # pull the endpoint back toward the center by `gap`, but
                # never past the center itself (can happen when the box
                # border sits very close to (cx, cy))
                pull_back = min(gap, length)
                entry_x -= int(vx / length * pull_back)
                entry_y -= int(vy / length * pull_back)

            end_x, end_y = entry_x, entry_y
        # if not `inside`, the segment never touches the box at all (rare
        # given the box is built from the target itself) - draw the full
        # line as a harmless fallback.

    cv2.line(image, (cx, cy), (end_x, end_y), line_color, thickness, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), r, center_color, -1, cv2.LINE_AA)
    cv2.circle(image, (mx, my), r, target_color, -1, cv2.LINE_AA)


def draw_centroid_marker(image, cx, cy, color=None, scale=1.0):
    """Simple filled dot for EVERY detected object's centroid."""
    r = max(3, round(4 * scale))
    cv2.circle(image, (cx, cy), r, color if color is not None else TARGET_DOT, -1, cv2.LINE_AA)


def draw_fps(image, fps, scale=1.0):
    """Same font, color, and scale-with-resolution behavior as the other
    streaming-window overlay text (draw_text_lines): green, HERSHEY_SIMPLEX,
    0.5*scale font size, thickness scaling with it too. Drawn top-right
    (rather than reusing draw_text_lines' top-left origin) so it never
    overlaps the Pixel Dist / depth / nozzle text there."""
    text = f"FPS: {fps:.1f}"
    font_scale = 0.5 * scale
    thickness = max(1, round(2 * scale))
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    h, w = image.shape[:2]
    x = w - tw - int(10 * scale)
    y = int(24 * scale)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)


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


def draw_detection_box(image, box, label, conf, scale=1.0, track_id=None):
    """Techy corner-bracket style instead of a full rectangle outline, plus
    a filled label pill (like a targeting HUD) instead of plain text.
    When track_id is given (ByteTrack/BotSORT active), it's appended on
    the same line as "Class 0.85", e.g. "Class  0.85   ID 3" - extra
    spaces between the three fields are intentional, giving each a little
    visual breathing room since cv2 has no real letter/word-spacing
    control beyond the text content itself."""
    x1, y1, x2, y2 = box
    color = TECH_ACCENT
    thickness = max(1, round(2 * scale))
    box_w = x2 - x1
    box_h = y2 - y1
    corner_len = int(min(box_w, box_h) * 0.28)
    corner_len = max(8, corner_len)
    max_corner = max(4, min((box_w - 6) // 2, (box_h - 6) // 2))
    corner_len = min(corner_len, max_corner)

    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(image, (px, py), (px + dx * corner_len, py), color, thickness, cv2.LINE_AA)
        cv2.line(image, (px, py), (px, py + dy * corner_len), color, thickness, cv2.LINE_AA)

    label_text = f"{label}  {conf:.2f}"
    if track_id is not None:
        label_text += f"   ID {track_id}"
    font_scale = 0.5 * scale
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = max(3, round(4 * scale))
    label_gap = max(2, round(3 * scale))
    # cv2 draws a line of a given thickness centered on its coordinate, so
    # the corner bracket's outer left edge actually sits `thickness // 2`
    # pixels to the left of x1 - not exactly at x1. Anchor the label pill
    # there (rather than at x1 plus an arbitrary offset) so its left edge
    # lines up with the bracket's outer edge instead of sitting off by a
    # couple of pixels.
    label_x = x1 - (thickness // 2)
    cv2.rectangle(image, (label_x, y1 - th - 2 * pad - label_gap), (label_x + tw + 2 * pad, y1 - label_gap), color, -1)
    cv2.putText(image, label_text, (label_x + pad, y1 - pad - label_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)