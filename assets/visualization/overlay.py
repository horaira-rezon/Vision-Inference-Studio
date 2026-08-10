import cv2
import numpy as np

TECH_ACCENT = (255, 210, 40)
TARGET_DOT = (0, 0, 255)

PALETTE = [
    (66, 133, 244),
    (52, 168, 83),
    (251, 188, 5),
    (234, 67, 53),
    (154, 88, 219),
    (255, 138, 0),
    (0, 188, 212),
    (233, 30, 99),
    (139, 195, 74),
    (121, 85, 72),
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def _color_for(key):
    idx = key % len(PALETTE) if isinstance(key, int) else abs(hash(key)) % len(PALETTE)
    return PALETTE[idx]

def _text_scale(scale):
    return min(scale, 1.3)

def draw_axes(image, cx, axis_y, thickness=1, color=(100, 100, 100)):
    h, w = image.shape[:2]
    cv2.line(image, (0, axis_y), (w, axis_y), color, thickness)
    cv2.line(image, (cx, 0), (cx, h), color, thickness)

def draw_fps(image, fps, scale=1.0):
    text = f"FPS: {fps:.1f}"
    ts = _text_scale(scale)
    font_scale = 0.5 * ts
    thickness = max(1, round(1.7 * ts))
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    h, w = image.shape[:2]
    x = w - tw - int(10 * ts)
    y = int(24 * ts)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

def draw_left_text(image, lines, scale=1.0):
    ts = _text_scale(scale)
    font_scale = 0.5 * ts
    thickness = max(1, round(1.7 * ts))
    x = int(10 * ts)
    y = int(24 * ts)
    for i, (text, color) in enumerate(lines):
        cv2.putText(image, text, (x, y + i * int(22 * ts)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_detection_box(image, box, label, conf, scale=1.0, track_id=None, color=None, draw_label=True):
    x1, y1, x2, y2 = box
    color = color if color is not None else TECH_ACCENT
    thickness = max(1, round(2 * scale))
    box_w = x2 - x1
    box_h = y2 - y1
    corner_len = int(min(box_w, box_h) * 0.28)
    corner_len = max(8, corner_len)
    max_corner = max(4, min((box_w - 6) // 2, (box_h - 6) // 2))
    corner_len = min(corner_len, max_corner)
    for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(image, (px, py), (px + dx * corner_len, py), color, thickness, cv2.LINE_AA)
        cv2.line(image, (px, py), (px, py + dy * corner_len), color, thickness, cv2.LINE_AA)
    if not draw_label:
        return
    label_text = f"{track_id} {label} {conf:.2f}" if track_id is not None else f"{label} {conf:.2f}"
    ts = _text_scale(scale)
    font_scale = 0.5 * ts
    text_thickness = max(1, round(1.7 * ts))
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    pad = max(3, round(4 * ts))
    label_gap = max(2, round(3 * ts))
    label_x = x1 - (thickness // 2)
    y_top = max(0, y1 - th - 2 * pad - label_gap)
    y_bottom = max(th + pad, y1 - label_gap)
    cv2.rectangle(image, (label_x, y_top), (label_x + tw + 2 * pad, y_bottom), color, -1)
    cv2.putText(image, label_text, (label_x + pad, y_bottom - pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), text_thickness, cv2.LINE_AA)

def draw_instance_masks(image, segments, alpha=0.45, scale=1.0):
    """Draws each segment in its own color with a label, matching the
    detection-box label style. Returns a list of {box, label, conf, track_id,
    color} dicts - one per drawn segment - with box computed as the tight
    (leftmost/rightmost/topmost/bottommost pixel) extent of that segment's
    mask, in the same pixel coordinates as `image`. Callers use this to draw
    a tracking bounding box without re-deriving it from the raw model box."""
    boxes = []
    for idx, item in enumerate(segments):
        mask = item.get("mask")
        if mask is None:
            continue
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = mask > 0.5
        if not np.any(binary):
            continue
        track_id = item.get("track_id")
        key = track_id if track_id is not None else idx
        color = _color_for(key)
        color_layer = np.zeros_like(image)
        color_layer[:, :] = color
        image[binary] = cv2.addWeighted(image[binary], 1.0 - alpha, color_layer[binary], alpha, 0)
        ys, xs = np.nonzero(binary)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        label = item.get("label", "object")
        conf = item.get("conf", 0.0)
        _draw_segment_label(image, (x1, y1, x2, y2), label, conf, color, scale, track_id)
        boxes.append({"box": (x1, y1, x2, y2), "label": label, "conf": conf, "track_id": track_id, "color": color})
    return boxes

def _draw_segment_label(image, box, label, conf, color, scale, track_id=None):
    x1, y1, x2, y2 = box
    label_text = f"{track_id} {label} {conf:.2f}" if track_id is not None else f"{label} {conf:.2f}"
    ts = _text_scale(scale)
    font_scale = 0.5 * ts
    text_thickness = max(1, round(1.7 * ts))
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    pad = max(3, round(4 * ts))
    label_gap = max(2, round(3 * ts))
    y_top = max(0, y1 - th - 2 * pad - label_gap)
    y_bottom = max(th + pad, y1 - label_gap)
    cv2.rectangle(image, (x1, y_top), (x1 + tw + 2 * pad, y_bottom), color, -1)
    cv2.putText(image, label_text, (x1 + pad, y_bottom - pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), text_thickness, cv2.LINE_AA)

def draw_semantic_mask(image, mask, alpha=0.35):
    if mask is None:
        return
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = mask >= 0
    if not np.any(valid):
        return
    color = np.zeros_like(image)
    color[:, :] = TECH_ACCENT
    image[valid] = cv2.addWeighted(image[valid], 1.0 - alpha, color[valid], alpha, 0)

def draw_pose(image, points, scale=1.0):
    """Draws keypoints and their COCO-skeleton connecting lines in the same
    accent color and line/text styling used for object detection. Returns
    the tight (min/max over all valid keypoints) bounding box in the same
    already-scaled pixel coordinates used to draw the points, or None if
    there were no valid keypoints - used by callers to draw a tracking box
    for pose without relying on the model's own box output."""
    if not points:
        return None
    radius = max(2, round(3 * scale))
    line_thickness = max(1, round(2 * scale))
    pts = []
    for point in points:
        if len(point) < 2:
            pts.append(None)
            continue
        x, y = int(point[0] * scale), int(point[1] * scale)
        if x <= 0 and y <= 0:
            pts.append(None)
        else:
            pts.append((x, y))
    for a, b in COCO_SKELETON:
        if a < len(pts) and b < len(pts) and pts[a] is not None and pts[b] is not None:
            cv2.line(image, pts[a], pts[b], TECH_ACCENT, line_thickness, cv2.LINE_AA)
    valid_pts = [p for p in pts if p is not None]
    for p in valid_pts:
        cv2.circle(image, p, radius, TECH_ACCENT, -1, cv2.LINE_AA)
    if not valid_pts:
        return None
    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    return (min(xs), min(ys), max(xs), max(ys))
