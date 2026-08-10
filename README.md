# Vision Inference Studio

Vision Inference Studio is a desktop app for running computer vision models against a live camera, a video file, or a still image, and watching the results overlaid on the stream in real time. It's built around a simple idea: pick a **Vision Task** (Classification, Detection, Segmentation, Pose Estimation), pick an **Architecture** for that task, point it at a **Model Weight**, and the app handles the rest.

It's written in Python with a `CustomTkinter` interface and OpenCV underneath, and is designed to run comfortably on a laptop as well as lower-power devices like a Raspberry Pi.

## ✨ Current Features

- Run 5 kinds of vision tasks: **Image Classification**, **Object Detection**, **Instance Segmentation**, **Semantic Segmentation**, and **2D Pose Estimation**; each with a choice of model architecture (see the table below).
- Feed it from 3 kinds of input (live USB camera/webcam, depth camera, video file, or image file).
- Track objects across frames with a choice of algorithm among ByteTrack and BotSORT (built into Ultralytics), or OC-SORT and DeepOC-SORT (via the `boxmot` package); each detection gets a persistent ID.
- Filter results by a customizable confidence threshold that governs what's drawn on screen.
- Reshape the live view: flip horizontally/vertically, rotate in 90° steps, or switch to an isolated R/G/B channel, grayscale, different thresholding techniques, an HSV or HSL mask, or a raw depth view (depth camera).
- Overlay reference axis lines on the stream (user can nudge the horizontal line up or down).
- Record video or take screenshots of what's on screen, in three flavors (the full overlay, just the clean camera frame, or just the detection boxes with their corresponding labels).
- See live FPS as an on-screen readout that is toggleable independently of everything else.

| Task | Available architectures |
|---|---|
| Image Classification | EfficientNet, ResNet |
| Object Detection | YOLO, RF-DETR, R-CNN, Faster R-CNN |
| Instance Segmentation | YOLO, SAM, RF-DETR, Mask R-CNN |
| Semantic Segmentation | U-Net |
| 2D Pose Estimation | YOLO, RF-DETR |

## 🛠️ Project Structure

```
Vision Inference Studio
├── assets/
│   ├── camera/
│   │   ├── base.py                  # Shared camera-source interface
│   │   ├── manager.py               # Scans for/creates a webcam or RealSense source
│   │   ├── realsense.py             # Intel RealSense depth camera support
│   │   └── webcam.py                # Plain USB/laptop webcam support
│   ├── communication/
│   │   └── network.py               # Placeholder for the "Remote Setup" input mode
│   ├── config/
│   │   └── settings.py              # Persisted app settings (JSON-backed)
│   ├── detection/
│   │   ├── detection_worker.py      # Background-thread inference runner
│   │   ├── custom_trackers.py       # OC-SORT / DeepOC-SORT via boxmot
│   │   └── generic_trackers.py      # Shared tracker interface/helpers
│   ├── media/
│   │   ├── image_source.py          # Static image file input
│   │   └── video_source.py          # Video file input
│   ├── recording/
│   │   └── recorder.py              # Video/screenshot writers
│   ├── ui/
│   │   └── file_dialogs.py          # Native file/folder picker helpers
│   ├── visualization/
│   │   └── overlay.py               # All drawing: boxes, masks, keypoints, axis lines, FPS
│   └── assets/settings.json         # Gitignored - your local runtime settings
├── gui/
│   ├── setup_screen.py              # First screen: Local vs Remote input
│   ├── app.py                       # MainApp - the composition root & frame loop
│   ├── left_sidebar.py              # Input / vision task / recording / screenshots
│   ├── view_controls.py             # Right sidebar: orientation & stream view mode
│   ├── media_view.py                # The central video panel widget
│   ├── vision_task_window.py        # Task + architecture picker dialogs
│   ├── config_window.py             # Tracking, confidence, axis lines, FPS viewer
│   └── input_dialogs.py             # Misc small input dialogs
├── models/
│   ├── task_registry.py             # Which architectures are offered for which task
│   ├── factory.py                   # (task, architecture, weight_path) -> model instance
│   ├── base.py                      # Shared model interface
│   └── architectures/
│       ├── yolo.py
│       ├── rf_detr.py
│       ├── sam.py
│       ├── rcnn.py
│       ├── faster_rcnn.py
│       ├── efficientnet.py
│       ├── resnet.py
│       ├── unet.py
│       └── weights.py               # Shared weight-loading helpers
├── main.py                          # Entry point - run this
└── requirements.txt                 # Python dependencies
```

## 📖 Quick Tutorial

1. **Launch the app:** `python3 main.py`. You'll land on the setup screen with two choices.

2. **Pick an input:** From the left sidebar's "1. Inference Input" section → **Select Input**. Choose a live camera (it's auto-detected if only one camera is available), a video file, or an image file.

3. **Pick a vision task:** From "2. Vision Tasks", choose Classification, Detection, Instance Segmentation, Semantic Segmentation, or Pose Estimation. You'll then be asked to pick an architecture.

4. **Load a model weight:** Click "Model Weight (.pt)" and point it at your trained weights file. Note: Pick an architecture based on which your model was trained otherwise, you'll get an error.

5. **Open Configuration** at the bottom of the left sidebar to:
   - turn on Object Tracking so detections get a persistent ID across frames,
   - raise or lower the Confidence Threshold; this governs what a tracker sees,
   - switch on X-Y Axis Lines for a center reference cross,
   - Slide the X-Line slider to nudge the horizontal line up and down,
   - toggle the FPS Viewer** on or off as an on-screen readout.

6. **Open View Controls** on the right sidebar to flip or rotate the stream, or switch the display to a single color channel, grayscale, different thresholding methods, HSV/HSL mask, or depth view (depth camera).

7. **Record or Screenshot:** From "3. Recording" / "4. Screenshot" in the left sidebar — pick an output folder once, then use Start Recording / Window Screenshot / Capture Cam. Frame / Detection UI Only as needed.

## ⚙️ Installation and Local Setup

**Prerequisits:** Python 3.10+, Git, a compatible camera, and sufficient CPU/GPU resources for the selected AI models, and the model weights corresponding to the selected architecture and vision task.

**1. Clone the repository:**

```bash
git clone <your-repository-url>
cd <repository-folder>
```

**2. Create a Virtual Environment: (recommended)**

```bash
python3 -m venv venv
source venv/bin/activate        # on Linux/macOS
venv\Scripts\activate           # on Windows
```

**3. Install the Dependencies:**

```bash
pip install -r requirements.txt
```

Things worth knowing about:

- **`torch`/`torchvision`** are pinned to CPU-generic versions; if you have an NVIDIA GPU and want CUDA acceleration, install the matching CUDA build (should be available at the official pytorch website) before running `pip install -r requirements.txt`, so it doesn't get overwritten with the CPU-only build.
- **`pyrealsense2`** is only needed if you're actually using an Intel RealSense camera. It's a native-code package and occasionally lags behind the newest Python versions — if it fails to install, you can comment that line out of `requirements.txt` and the app will simply fall back to webcam-only input.
- **`boxmot`** powers the OC-SORT/DeepOC-SORT tracking options specifically; ByteTrack/BotSORT come from `ultralytics` and don't need the boxmot package for this.

**4. Run It:**

```bash
python3 main.py
```

That's it. The app creates its own local `assets/assets/settings.json` on first run to remember your last-used folders and Configuration preferences (it's gitignored).

## ⭐ Citation

If you use this project in your academic research or other works, please cite this repository. And, if this software helps you with your project, dropping a star on the repo is always appreciated!

---

## 📄 License

This project is distributed under the MIT License and is fully open for reference and learning purposes. See LICENSE file for details. You are free to clone it and adapt the structural code for your own project.

---

## 🌐 Support

For issues and feature requests, please open a GitHub issue or contact the development team.