import cv2
import numpy as np

TECH_ACCENT = (255, 210, 40)
TARGET_DOT = (0, 0, 255)

def _text_scale(scale):
    return min(scale, 1.3)

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

def draw_detection_box(image, box, label, conf, scale=1.0, track_id=None):
    x1, y1, x2, y2 = box
    color = TECH_ACCENT
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

def draw_instance_masks(image, segments, alpha=0.35):
    for item in segments:
        mask = item.get("mask")
        if mask is None:
            continue
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = mask > 0.5
        if not np.any(binary):
            continue
        color = np.zeros_like(image)
        color[:, :] = TECH_ACCENT
        image[binary] = cv2.addWeighted(image[binary], 1.0 - alpha, color[binary], alpha, 0)

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
    if points is None:
        return
    radius = max(2, round(3 * scale))
    for point in points:
        if len(point) < 2:
            continue
        x, y = int(point[0] * scale), int(point[1] * scale)
        if x <= 0 and y <= 0:
            continue
        cv2.circle(image, (x, y), radius, TECH_ACCENT, -1, cv2.LINE_AA)
