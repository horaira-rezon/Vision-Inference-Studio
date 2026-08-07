"""
OC-SORT and DeepSORT tracking, added alongside Ultralytics' built-in
ByteTrack/BotSORT support. Ultralytics' model.track() only ships
bytetrack.yaml and botsort.yaml internally - it has no native OC-SORT or
DeepSORT config - so those two run through the `boxmot` package instead:
yolo_engine.py runs a plain (non-tracking) YOLO forward pass to get raw
detections, hands them to track() below, and a persistent boxmot tracker
instance assigns track IDs to them frame-to-frame - same end result (a
track_id per box) as the Ultralytics trackers, and the same output shape
(box/conf/label/track_id) so yolo_engine.detect() can return it unchanged.

Requires the `boxmot` package (pip install boxmot).
"""

import numpy as np

_TRACKER_INSTANCES = {}


def _get_tracker(tracker_key):
    """One persistent tracker instance per algorithm, created lazily and
    reused every frame so identity is preserved across frames - the same
    role Ultralytics' persist=True plays for bytetrack/botsort."""
    if tracker_key in _TRACKER_INSTANCES:
        return _TRACKER_INSTANCES[tracker_key]

    from pathlib import Path
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if tracker_key == "ocsort":
        from boxmot.trackers.bbox import OcSort
        tracker = OcSort()
    elif tracker_key == "deepsort":
        from boxmot.trackers.bbox import DeepOcSort
        tracker = DeepOcSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device=device,
            half=False,
        )
    else:
        raise ValueError(f"Unknown custom tracker: {tracker_key}")

    _TRACKER_INSTANCES[tracker_key] = tracker
    return tracker


def track(tracker_key, frame, boxes):
    """boxes: list of (x1, y1, x2, y2, conf, cls_id) tuples from a plain
    (non-tracking) YOLO pass. Returns a list of dicts - box/conf/cls_id/
    track_id - one per still-tracked box, in boxmot's own output order."""
    tracker = _get_tracker(tracker_key)

    dets = np.array(boxes, dtype=float) if boxes else np.empty((0, 6))
    tracked = tracker.update(dets, frame)

    results = []
    for row in tracked:
        x1, y1, x2, y2, track_id, conf, cls_id = row[:7]
        results.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "conf": float(conf),
            "cls_id": int(cls_id),
            "track_id": int(track_id),
        })
    return results


def reset(tracker_key=None):
    """Drops the persistent tracker instance(s) - called whenever a new
    model is loaded (see YoloEngine.__init__) so an old tracker's internal
    state never leaks into a new model/session. tracker_key=None clears
    every custom tracker at once."""
    if tracker_key is None:
        _TRACKER_INSTANCES.clear()
    else:
        _TRACKER_INSTANCES.pop(tracker_key, None)