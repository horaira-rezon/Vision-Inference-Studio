import json
import os

DEFAULT_SETTINGS = {
    "video_output_dir": None,
    "screenshot_output_dir": None,
    "tracker_mode": "none",
    "confidence_threshold": 50,
    "fps_viewer_on": True,
    "axis_lines_on": False,
    "axis_line_slider": 0.5,
    "coco_class_filter": None,  # list of COCO class_ids, or None/empty = show all
}

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "assets", "settings.json")

class Settings:
    def __init__(self, path=SETTINGS_FILE):
        self.path = path
        self.data = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key in DEFAULT_SETTINGS:
                if key in saved:
                    self.data[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, persist=True):
        if key not in DEFAULT_SETTINGS:
            return
        self.data[key] = value
        if persist:
            self.save()
