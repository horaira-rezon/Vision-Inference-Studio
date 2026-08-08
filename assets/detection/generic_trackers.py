import numpy as np

_INSTANCES = {}

def reset():
    _INSTANCES.clear()

def _build(name):
    if name in _INSTANCES:
        return _INSTANCES[name]
    from boxmot.trackers.bbox import BYTETracker, BotSort, OcSort, DeepOcSort
    mapping = {
        "bytetrack": BYTETracker,
        "botsort": BotSort,
        "ocsort": OcSort,
        "deepocsort": DeepOcSort,
    }
    cls = mapping[name]
    try:
        tracker = cls()
    except TypeError:
        tracker = cls()
    _INSTANCES[name] = tracker
    return tracker

def apply(name, frame, items):
    if name == "none" or not items:
        return items
    if name in ("ocsort", "deepocsort"):
        from assets.detection import custom_trackers
        raw = [(d["box"][0], d["box"][1], d["box"][2], d["box"][3], d["conf"], d.get("class_id", 0)) for d in items]
        tracked = custom_trackers.track(name, frame, raw)
        result = []
        for row in tracked:
            best = min(items, key=lambda d: abs(d["box"][0]-row["box"][0])+abs(d["box"][1]-row["box"][1])+abs(d["box"][2]-row["box"][2])+abs(d["box"][3]-row["box"][3]))
            copy = dict(best)
            copy["box"] = row["box"]
            copy["conf"] = row["conf"]
            copy["track_id"] = row["track_id"]
            result.append(copy)
        return result
    tracker = _build(name)
    dets = np.array([(d["box"][0],d["box"][1],d["box"][2],d["box"][3],d["conf"],d.get("class_id",0)) for d in items], dtype=float)
    if dets.size == 0:
        return items
    tracked = tracker.update(dets, frame)
    result = []
    for row in tracked:
        x1,y1,x2,y2,tid,conf,cls_id = row[:7]
        best = min(items, key=lambda d: abs(d["box"][0]-x1)+abs(d["box"][1]-y1)+abs(d["box"][2]-x2)+abs(d["box"][3]-y2))
        copy = dict(best)
        copy["box"]=(int(x1),int(y1),int(x2),int(y2))
        copy["conf"]=float(conf)
        copy["track_id"]=int(tid)
        result.append(copy)
    return result
