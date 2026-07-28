"""
File Dialogs: Tkinter's built-in filedialog is the plain Tcl/Tk picker -
no typing a path directly, slow navigation, no search. On Linux, zenity
wraps the actual GTK file chooser (fast, supports typing/pasting a path,
bookmarks, recent locations), so we shell out to it when available and
fall back to tkinter's dialog otherwise (e.g. zenity not installed, or
running on Windows/macOS).

Install zenity if you don't have it: sudo apt install zenity
"""

import shutil
import subprocess
from tkinter import filedialog

ZENITY_AVAILABLE = shutil.which("zenity") is not None


def choose_file(title, pattern="*.*", pattern_label="Files"):
    """Returns a selected file path, or None if cancelled."""
    if ZENITY_AVAILABLE:
        try:
            result = subprocess.run(
                ["zenity", "--file-selection", "--title", title,
                 "--file-filter", f"{pattern_label} | {pattern}"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                return path or None
            return None  # user cancelled
        except Exception:
            pass  # zenity present but failed for some reason - fall through

    ext = pattern.lstrip("*")
    return filedialog.askopenfilename(title=title, filetypes=[(pattern_label, pattern)]) or None


def choose_directory(title):
    """Returns a selected directory path, or None if cancelled."""
    if ZENITY_AVAILABLE:
        try:
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title", title],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                return path or None
            return None
        except Exception:
            pass

    return filedialog.askdirectory(title=title) or None