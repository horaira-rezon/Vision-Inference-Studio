"""
Settings Manager

Stores everything that used to be hardcoded (Arduino port, stepper hardware
constants, and — new — the chosen video/screenshot folders) in one JSON file
under assets/, so every other module reads from here instead of from
scattered constants.
"""

import json
import os

DEFAULT_SETTINGS = {
    "video_output_dir": None,
    "screenshot_output_dir": None,
    "arduino_port": "/dev/ttyACM0",
    "arduino_baud": 115200,
    "steps_per_degree": 400.0 / 360.0,   # NEMA17 @ 400 steps/rev
    "command_delay": 0.1,

    # --- Configuration window (gui/config_window.py) ---
    # "none" | "external" | "diagonal_only" - None until the user picks one
    "actuation_mode": None,
    # whether the diagonal line + its overlay text (pixel/real-world dist,
    # nozzle angle, steps, etc.) are drawn at all
    "diagonal_distance_on": False,
    # 0.0..1.0, 0.5 = centered/no shift. Only applied when actuation_mode
    # is "external" - moves the horizontal crosshair line (and therefore
    # where the diagonal line/its cyan dot originate) up or down without
    # moving the true camera-center dot or any real-world calculation.
    "x_axis_slider": 0.5,
    # Object Tracking (Configuration window, section 4) - "none" | "bytetrack"
    # | "botsort". Independent of actuation_mode/diagonal_distance_on.
    "tracker_mode": "none",
    # Confidence Threshold (section 5), 0-100. Boxes below this are dropped
    # by YoloEngine.detect() itself - the SAME filtered list is what gets
    # drawn in the streaming window AND what's eligible for Arduino
    # actuation, so this one setting governs both.
    "confidence_threshold": 50,
    # FPS viewer - independent of every other Configuration-window option
    # above (External Actuation mode, Diagonal Distance, etc.) and of the
    # Reset button, which only touches the other four sections. On by
    # default; visible in the live view and the full "Take Screenshot",
    # but not the clean or boxes-only screenshot variants.
    "fps_viewer_on": True,
}

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "..", "assets", "settings.json")


class Settings:
    def __init__(self, path=SETTINGS_FILE):
        self.path = path
        self.data = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    saved = json.load(f)
                self.data.update(saved)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt or unreadable file -> fall back to defaults quietly

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, persist=True):
        self.data[key] = value
        if persist:
            self.save()