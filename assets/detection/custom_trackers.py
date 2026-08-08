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

    # boxmot's own registered download source for this exact file has a
    # long, well-documented history of dead/rate-limited links (it was
    # originally Google-Drive-hosted, going back to the torchreid project
    # this is descended from - see e.g. github.com/mikel-brostrom/boxmot
    # issues #781, #1154, #944 for the same "silently fails to load"
    # pattern with different weights files). A stable connection but still
    # reid_model=None points at that, not at anything local. Rather than
    # keep guessing at boxmot's internal download mechanism, fetch a known
    # copy ourselves from a couple of verified-to-exist mirrors. This is
    # safe even if a mirror turns out to be the wrong file/architecture:
    # boxmot's own loader below is what actually validates it (matching
    # state_dict keys/shapes), so a bad download just fails to load and
    # falls through to the same motion-only fallback as before - it can
    # never silently produce corrupt/wrong embeddings.
    target = attempts[0] if attempts and isinstance(attempts[0], Path) else Path("osnet_x0_25_msmt17.pt")
    if _try_download_reid_weights(target):
        try:
            tracker = DeepOcSort(reid_weights=target, device=device, half=False)
            if getattr(tracker, "reid_model", None) is not None:
                return tracker
            print(f"[custom_trackers] downloaded {target} but DeepOcSort still reports reid_model=None after loading it")
        except Exception as e:
            print(f"[custom_trackers] downloaded {target} but DeepOcSort raised loading it: {type(e).__name__}: {e}")

    print(
        "[custom_trackers] DeepSORT's ReID weights (osnet_x0_25_msmt17.pt) "
        f"could not be loaded automatically (tried boxmot's own resolution: "
        f"{[str(a) for a in attempts]}, and a direct download from a couple "
        "of known mirrors). Falling back to DeepSORT's motion-only mode "
        "(embedding_off=True) - tracking still works, just without "
        "appearance-based re-identification. See this session's reply for "
        "how to check exactly why boxmot's own download is failing, and "
        f"where to place a manually-downloaded copy ({target})."
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


def _try_download_reid_weights(target_path):
    """Best-effort direct download of osnet_x0_25_msmt17.pt. The primary
    mirror is a HuggingFace repo dedicated specifically to this file
    (not bundled inside someone's unrelated demo project); its SHA256
    matches a second, independently-uploaded copy found separately, which
    is strong evidence it's the genuine, correct checkpoint rather than a
    mismatched/corrupted one - so it's verified against that hash before
    ever being accepted. The secondary mirror has no independently-
    confirmed hash, so it only gets a size sanity check; either way,
    boxmot's own loader (the caller) is the final, real correctness
    check - a bad file simply fails to load and falls through to the
    existing motion-only fallback, never silently produces bad tracking."""
    import hashlib
    import urllib.request

    KNOWN_GOOD_SHA256 = "6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99"
    mirrors = [
        ("https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt", KNOWN_GOOD_SHA256),
        ("https://huggingface.co/spaces/xfys/yolov5_tracking/resolve/main/weights/osnet_x0_25_msmt17.pt", None),
    ]

    for url, expected_sha256 in mirrors:
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, target_path)
            if not target_path.exists() or target_path.stat().st_size < 1_000_000:
                size = target_path.stat().st_size if target_path.exists() else 0
                print(f"[custom_trackers] download from {url} produced only {size} bytes (expected >1MB) - discarding")
                target_path.unlink(missing_ok=True)
                continue
            if expected_sha256:
                actual = hashlib.sha256(target_path.read_bytes()).hexdigest()
                if actual != expected_sha256:
                    print(f"[custom_trackers] downloaded {url} but its SHA256 didn't match the known-good hash (got {actual}) - discarding")
                    target_path.unlink(missing_ok=True)
                    continue
            print(f"[custom_trackers] downloaded and verified {url} -> {target_path}")
            return True
        except Exception as e:
            print(f"[custom_trackers] download attempt failed for {url}: {type(e).__name__}: {e}")
            target_path.unlink(missing_ok=True)
            continue
    return False


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