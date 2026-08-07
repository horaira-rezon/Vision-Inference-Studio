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
_TRACKER_ERRORS = {}  # tracker_key -> error message, so a known-broken
                       # tracker (e.g. DeepSORT with a failed ReID download)
                       # doesn't retry construction on every single frame


def _get_tracker(tracker_key):
    """One persistent tracker instance per algorithm, created lazily and
    reused every frame so identity is preserved across frames - the same
    role Ultralytics' persist=True plays for bytetrack/botsort."""
    if tracker_key in _TRACKER_INSTANCES:
        return _TRACKER_INSTANCES[tracker_key]

    if tracker_key in _TRACKER_ERRORS:
        # already failed once this session - re-raise the same message
        # instead of re-attempting a potentially expensive (or network-
        # downloading) construction again on every frame
        raise RuntimeError(_TRACKER_ERRORS[tracker_key])

    from pathlib import Path
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # boxmot's own auto-downloader targets its package-relative WEIGHTS
    # folder - that's what its example/CLI code points reid_weights at.
    # A bare relative filename (no resolved directory) is what silently
    # produced reid_model=None before: nothing on disk at that path, and
    # apparently no download triggered from it either.
    try:
        from boxmot.utils import WEIGHTS
        reid_weights_path = WEIGHTS / "osnet_x0_25_msmt17.pt"
    except Exception:
        reid_weights_path = Path("osnet_x0_25_msmt17.pt")

    try:
        if tracker_key == "ocsort":
            from boxmot.trackers.bbox import OcSort
            tracker = OcSort()
        elif tracker_key == "deepsort":
            from boxmot.trackers.bbox import DeepOcSort
            tracker = DeepOcSort(
                reid_weights=reid_weights_path,
                device=device,
                half=False,
            )

            # DeepOcSort's constructor does NOT raise if the ReID weights
            # fail to load or auto-download (e.g. no internet access at
            # runtime, or this boxmot version expects the weights placed
            # somewhere other than reid_weights_path) - it silently falls
            # back to reid_model=None and then produces empty tracking
            # output on every single frame, which is exactly what "DeepSORT
            # shows nothing at all" looked like, with no error anywhere.
            # Failing loudly here, the moment it happens, turns that into
            # a visible notice instead.
            if getattr(tracker, "reid_model", None) is None:
                raise RuntimeError(
                    f"DeepSORT's ReID weights did not load from "
                    f"{reid_weights_path} - boxmot silently falls back to a "
                    f"non-functional tracker instead of raising, which is why "
                    f"nothing was appearing on screen. This usually means the "
                    f"auto-download failed (check internet access on this "
                    f"machine) or your installed boxmot version expects the "
                    f"weights somewhere else. Try downloading "
                    f"osnet_x0_25_msmt17.pt yourself and placing it at exactly "
                    f"that path, or run `python3 -c \"from boxmot import "
                    f"DeepOcSort; DeepOcSort(reid_weights='{reid_weights_path}', "
                    f"device='cpu', half=False)\"` in your terminal (with the "
                    f"venv active) to see boxmot's own warning/error output "
                    f"directly."
                )
        else:
            raise ValueError(f"Unknown custom tracker: {tracker_key}")
    except Exception as e:
        _TRACKER_ERRORS[tracker_key] = str(e)
        raise

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
    """Drops the persistent tracker instance(s) AND any cached error for
    them - called whenever a new model is loaded (see YoloEngine.__init__)
    or the tracker selection changes (see YoloEngine.detect()), so old
    state/errors never leak into a new model/session/attempt.
    tracker_key=None clears everything at once."""
    if tracker_key is None:
        _TRACKER_INSTANCES.clear()
        _TRACKER_ERRORS.clear()
    else:
        _TRACKER_INSTANCES.pop(tracker_key, None)
        _TRACKER_ERRORS.pop(tracker_key, None)