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

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        if tracker_key == "ocsort":
            from boxmot.trackers.bbox import OcSort
            tracker = OcSort()
        elif tracker_key == "deepsort":
            tracker = _build_deepsort(device)
        else:
            raise ValueError(f"Unknown custom tracker: {tracker_key}")
    except Exception as e:
        _TRACKER_ERRORS[tracker_key] = str(e)
        raise

    _TRACKER_INSTANCES[tracker_key] = tracker
    return tracker


def _build_deepsort(device):
    """Tries boxmot's ReID weight loading two different ways - both are
    real, documented usages that just vary by boxmot version - and only
    falls back to embedding_off=True (boxmot's own fully-supported
    motion-only mode, same association logic as OC-SORT) if NEITHER
    actually produces a loaded reid_model. This means DeepSORT always ends
    up usable, even on a machine/boxmot version where the appearance
    ReID weights can't be resolved automatically - degraded re-id is far
    better than a tracker that silently produces nothing at all."""
    from pathlib import Path
    from boxmot.trackers.bbox import DeepOcSort

    attempts = []
    try:
        from boxmot.utils import WEIGHTS
        attempts.append(WEIGHTS / "osnet_x0_25_msmt17.pt")
    except Exception:
        pass
    # Bare filename (no directory) - this is the form boxmot's own CLI/
    # example scripts pass; some versions only recognize the auto-download
    # name as an exact string match and won't trigger on a Path object
    # pointing at the same file, which looks like what happened here: a
    # stable connection, but still reid_model=None with the Path form.
    attempts.append("osnet_x0_25_msmt17.pt")

    for weights in attempts:
        tracker = DeepOcSort(reid_weights=weights, device=device, half=False)
        if getattr(tracker, "reid_model", None) is not None:
            return tracker

    print(
        "[custom_trackers] DeepSORT's ReID weights (osnet_x0_25_msmt17.pt) "
        f"could not be loaded automatically (tried: {[str(a) for a in attempts]}). "
        "Falling back to DeepSORT's motion-only mode (embedding_off=True) - "
        "tracking still works, just without appearance-based re-identification. "
        "For full DeepSORT, download osnet_x0_25_msmt17.pt yourself and place "
        f"it at {attempts[0]}."
    )
    try:
        return DeepOcSort(reid_weights=Path("osnet_x0_25_msmt17.pt"), device=device, half=False, embedding_off=True)
    except Exception as e:
        raise RuntimeError(
            "DeepSORT could not be constructed at all, even in motion-only "
            f"mode (embedding_off=True): {e}. This points to a boxmot "
            "installation/version issue beyond just the ReID weights - check "
            "`pip show boxmot` and that DeepOcSort is available in your "
            "installed version."
        ) from e


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