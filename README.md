# Vision Inference Studio

Vision Inference Studio is a desktop app for running computer vision models against a live camera, a video file, or a still image, and watching the results overlaid on the stream in real time. It's built around a simple idea: pick a **task** (classification, detection, segmentation, pose), pick an **architecture** for that task, point it at a **model weight file**, and the app handles the rest — decoding frames, running inference on a background thread so the UI never freezes, drawing the results, optionally tracking objects across frames, and recording or screenshotting whatever you're looking at.

It's written in Python with a [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) interface and OpenCV underneath, and is designed to run comfortably on a laptop as well as lower-power devices like a Raspberry Pi.

## What you can do with it

- **Run five kinds of vision tasks** — image classification, object detection, instance segmentation, semantic segmentation, and 2D pose estimation — each with a choice of model architecture (see the table below).
- **Feed it from three kinds of input** — a live camera (a plain USB/laptop webcam, or an Intel RealSense depth camera), a video file, or a static image file.
- **Track objects across frames** with a choice of algorithm — ByteTrack and BotSORT (built into Ultralytics), or OC-SORT and DeepOC-SORT (via the `boxmot` package) — each detection gets a persistent ID once tracking is on.
- **Filter results by confidence** with a single threshold that governs both what's drawn on screen and what a tracker is allowed to consider.
- **Reshape the live view** without touching the underlying frame data: flip horizontally/vertically, rotate in 90° steps, or switch the display to an isolated Red/Green/Blue channel, grayscale, a binary threshold, an HSV or HSL mask, or (on a depth camera) the raw depth view.
- **Overlay reference axis lines** on the stream — a fixed vertical line and a horizontal line you can nudge up or down — useful for visually judging where something sits relative to center.
- **Record video or take screenshots** of what's on screen, in three flavors: the full overlay, just the clean camera frame, or just the detection boxes/labels with nothing else.
- **See live FPS** as an on-screen readout, toggleable independently of everything else.

| Task | Available architectures |
|---|---|
| Image Classification | EfficientNet, ResNet |
| Object Detection | YOLO, RF-DETR, R-CNN, Faster R-CNN |
| Instance Segmentation | YOLO, SAM, RF-DETR, Mask R-CNN |
| Semantic Segmentation | U-Net |
| 2D Pose Estimation | YOLO, RF-DETR |

## How the app is put together

At a high level, the app has four layers that stay deliberately separate from each other:

1. **Input layer** (`assets/camera/`, `assets/media/`) — anything that can hand the app a raw video frame. A `CameraManager` scans for a webcam or a RealSense camera; `ImageFileSource`/`VideoFileSource` do the same job for a still image or a video file. Every source exposes the same `start()`/`read()` shape, so the rest of the app doesn't need to know or care which one is active.

2. **Model layer** (`models/`) — a small plugin system for vision models. `models/task_registry.py` is the single source of truth for which architectures are offered for which task; `models/factory.py` turns a `(task, architecture, weight_path)` triple into a concrete model instance; each file under `models/architectures/` wraps one underlying library (Ultralytics for YOLO, SAM's own package, torchvision for ResNet/Faster R-CNN, etc.) behind the same `predict(frame)` interface. Adding a new architecture means adding one file here and one registry entry — nothing else in the app needs to change.

3. **Inference + tracking** (`assets/detection/`) — `DetectionWorker` runs the model on a dedicated background thread so a slow model never freezes the video feed; the main thread just asks it for whatever the latest result is. `custom_trackers.py` and `generic_trackers.py` wrap the tracking algorithms (Ultralytics' own for ByteTrack/BotSORT, `boxmot` for OC-SORT/DeepOC-SORT) behind one shared interface so the rest of the app treats every tracker identically.

4. **GUI layer** (`gui/`) — `MainApp` (`gui/app.py`) is the composition root: it owns the camera/model/tracker state, runs the frame loop, and wires together the **left sidebar** (input, vision task, recording, screenshots), the **right sidebar** / `ViewControls` (orientation, channel/color-space view, depth), the central `MediaView` (the actual video panel), and the **Configuration** window (tracking algorithm, confidence threshold, axis lines, FPS viewer). Settings are read fresh every frame from a shared `Settings` object (`assets/config/settings.py`), which persists to a small JSON file — so a slider you drag in the Configuration window takes effect on the very next frame, no explicit "apply" step needed.

The overlay drawing itself (`assets/visualization/overlay.py`) is kept separate from all of the above — it only knows how to draw boxes, masks, keypoints, text, and axis lines onto an image array, and has no idea where that data came from.

## Project tree

```
.
├── main.py                          # entry point - run this
├── requirements.txt
├── assets/
│   ├── camera/
│   │   ├── base.py                  # shared camera-source interface
│   │   ├── manager.py               # scans for/creates a webcam or RealSense source
│   │   ├── realsense.py             # Intel RealSense depth camera support
│   │   └── webcam.py                # plain USB/laptop webcam support
│   ├── communication/
│   │   └── network.py               # placeholder for the "Remote Setup" input mode
│   ├── config/
│   │   └── settings.py              # persisted app settings (JSON-backed)
│   ├── detection/
│   │   ├── detection_worker.py      # background-thread inference runner
│   │   ├── custom_trackers.py       # OC-SORT / DeepOC-SORT via boxmot
│   │   └── generic_trackers.py      # shared tracker interface/helpers
│   ├── media/
│   │   ├── image_source.py          # static image file input
│   │   └── video_source.py          # video file input
│   ├── recording/
│   │   └── recorder.py              # video/screenshot writers
│   ├── ui/
│   │   └── file_dialogs.py          # native file/folder picker helpers
│   ├── visualization/
│   │   └── overlay.py               # all drawing: boxes, masks, keypoints, axis lines, FPS
│   └── assets/settings.json         # gitignored - your local runtime settings
├── gui/
│   ├── setup_screen.py              # first screen: Local vs Remote input
│   ├── app.py                       # MainApp - the composition root & frame loop
│   ├── left_sidebar.py              # input / vision task / recording / screenshots
│   ├── view_controls.py             # right sidebar: orientation & stream view mode
│   ├── media_view.py                # the central video panel widget
│   ├── vision_task_window.py        # task + architecture picker dialogs
│   ├── config_window.py             # tracking, confidence, axis lines, FPS viewer
│   └── input_dialogs.py             # misc small input dialogs
└── models/
    ├── task_registry.py             # which architectures are offered for which task
    ├── factory.py                   # (task, architecture, weight_path) -> model instance
    ├── base.py                      # shared model interface
    └── architectures/
        ├── yolo.py
        ├── rf_detr.py
        ├── sam.py
        ├── rcnn.py
        ├── faster_rcnn.py
        ├── efficientnet.py
        ├── resnet.py
        ├── unet.py
        └── _weights.py              # shared weight-loading helpers
```

## Quick tutorial

1. **Launch the app** — `python3 main.py`. You'll land on the setup screen with two choices; click **Local Setup** (remote streaming is a placeholder for now).

2. **Pick an input**, from the left sidebar's "1. Inference Input" section → **Select Input**. Choose a live camera (it's auto-detected — a RealSense camera if one's plugged in, otherwise your webcam), a video file, or an image file.

3. **Pick a vision task**, "2. Vision Tasks" → **Vision Task** — choose Classification, Detection, Instance Segmentation, Semantic Segmentation, or Pose. You'll then be asked to pick which architecture to run for that task (e.g. YOLO vs Faster R-CNN for detection).

4. **Load a model weight** — click **Model Weight (.pt)** and point it at your trained weights file for the architecture you picked.

5. **(Optional) Open Configuration** at the bottom of the left sidebar to:
   - turn on **Object Tracking** (ByteTrack, BotSORT, OC-SORT, or DeepOC-SORT) so detections get a persistent ID across frames,
   - raise or lower the **Confidence Threshold** — this governs what's drawn *and* what a tracker sees,
   - switch on **X-Y Axis Lines** for a center reference cross, with a slider to nudge the horizontal line,
   - toggle the **FPS Viewer** on or off.

6. **(Optional) Open View Controls** on the right sidebar to flip or rotate the stream, or switch the display to a single color channel, grayscale, a binary threshold, HSV/HSL mask, or depth view (depth only if a RealSense camera is active) — none of this changes what the model actually sees, only what you see.

7. **Record or screenshot** from "3. Recording" / "4. Screenshot" in the left sidebar — pick an output folder once, then use **Start Recording** / **Window Screenshot** (full overlay) / **Capture Cam. Frame** (clean, no overlay) / **Detection UI Only** (boxes/labels, nothing else) as needed.

## Installing it on your own machine

**1. Clone the repository:**

```bash
git clone <your-repository-url>
cd <repository-folder>
```

**2. Create and activate a virtual environment** (recommended, keeps dependencies isolated from the rest of your system):

```bash
python3 -m venv venv
source venv/bin/activate        # on Linux/macOS
venv\Scripts\activate           # on Windows
```

**3. Install the dependencies:**

```bash
pip install -r requirements.txt
```

A couple of things worth knowing about that install:

- **`torch`/`torchvision`** are pinned to CPU-generic versions here; if you have an NVIDIA GPU and want CUDA acceleration, install the matching CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) *before* running `pip install -r requirements.txt`, so it doesn't get overwritten with the CPU-only build.
- **`pyrealsense2`** is only needed if you're actually using an Intel RealSense camera. It's a native-code package and occasionally lags behind the newest Python versions — if it fails to install, you can comment that line out of `requirements.txt` and the app will simply fall back to webcam-only input (it detects at runtime whether the package is available).
- **`boxmot`** powers the OC-SORT/DeepOC-SORT tracking options specifically; ByteTrack/BotSORT come from `ultralytics` and don't need it.

**4. Run it:**

```bash
python3 main.py
```

That's it — the app creates its own local `assets/assets/settings.json` on first run to remember your last-used folders and Configuration preferences (it's gitignored, so it stays local to your machine and won't get committed).
