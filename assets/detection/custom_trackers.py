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
    """Builds a real ReID backend via boxmot's ReID class and hands the
    resulting *backend object* to DeepOcSort as reid_model=. DeepOcSort's
    constructor has no reid_weights/device/half parameters at all (those
    only exist on boxmot's higher-level create_tracker() registry helper,
    which is what internally turns reid_weights into a ReID(...).model and
    passes THAT in as reid_model) - so passing reid_weights=/device=/half=
    straight into DeepOcSort(), like every previous attempt here did,
    silently vanishes into **kwargs and is never used. That's the whole
    story: reid_model was staying None because no ReID backend was ever
    being constructed in the first place, not because of a bad download,
    a filename-registry collision, or a wrapped checkpoint - the weights
    file itself was fine the whole time.

    Falls back to embedding_off=True (boxmot's own fully-supported
    motion-only mode, same association logic as OC-SORT) only if a real
    ReID backend genuinely can't be built - so DeepSORT always ends up
    usable, even on a machine where the appearance weights can't be
    resolved at all."""
    from boxmot.reid.core import ReID
    from boxmot.trackers.bbox import DeepOcSort
    from boxmot.utils import WEIGHTS

    weights_path = WEIGHTS / "osnet_x0_25_msmt17.pt"

    if not weights_path.exists():
        _try_download_reid_weights(weights_path)

    if weights_path.exists():
        try:
            # Build the backend ourselves, then hand the *object* (not the
            # path) to DeepOcSort - this is the "reid_model: Pre-built ReID
            # backend model (e.g. ReID(...).model)" usage documented on
            # DeepOcSort itself.
            reid_backend = ReID(path=weights_path, device=device, half=False).model
            tracker = DeepOcSort(reid_model=reid_backend)
            if getattr(tracker, "model", None) is not None:
                return tracker
            print(f"[custom_trackers] built a ReID backend from {weights_path} but DeepOcSort.model is still None after passing it in as reid_model - inspecting the checkpoint")
            _inspect_checkpoint(weights_path)
        except Exception as e:
            print(f"[custom_trackers] failed to build/load a ReID backend from {weights_path}: {type(e).__name__}: {e}")

    print(
        "[custom_trackers] DeepSORT's ReID weights (osnet_x0_25_msmt17.pt) "
        f"could not be loaded (expected at {weights_path}). Falling back to "
        "DeepSORT's motion-only mode (embedding_off=True) - tracking still "
        "works, just without appearance-based re-identification."
    )
    try:
        return DeepOcSort(embedding_off=True)
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


def _inspect_checkpoint(path):
    """Loads the .pt file ourselves, independent of boxmot, and prints
    what's actually inside it - top-level type/keys, and a few state_dict
    key names if there's something state_dict-shaped. This is diagnostic
    only (never raises out to the caller): if DeepOcSort still won't use
    a SHA256-verified-correct file even under a non-registry filename,
    the next real question is whether the checkpoint is wrapped in an
    extra layer (e.g. {"state_dict": {...}} or {"model": ...} instead of
    a bare state_dict) or uses different key naming than this boxmot
    version's OSNet class expects - either of which would explain
    reid_model staying None without boxmot necessarily raising."""
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            print(f"[custom_trackers] checkpoint is a dict with {len(keys)} top-level keys, first few: {keys[:10]}")
            # a bare state_dict looks like {"conv1.conv.weight": tensor, ...} -
            # lots of dotted param-name keys. A wrapped checkpoint looks like
            # {"state_dict": {...}} or {"model": ..., "epoch": ...} - few
            # top-level keys, none of them tensors.
            sample = keys[0] if keys else None
            if sample is not None and hasattr(ckpt[sample], "shape"):
                print("[custom_trackers] looks like a bare state_dict (values are tensors)")
            else:
                print(f"[custom_trackers] does NOT look like a bare state_dict - value type for '{sample}' is {type(ckpt.get(sample)).__name__ if sample else '?'}. This is likely why loading silently produced nothing: boxmot's OSNet loader may expect the state_dict directly, not wrapped inside this key.")
        else:
            print(f"[custom_trackers] checkpoint top-level type is {type(ckpt).__name__}, not a dict at all")
    except Exception as e:
        print(f"[custom_trackers] could not even torch.load {path} ourselves: {type(e).__name__}: {e}")


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