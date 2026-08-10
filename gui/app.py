import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk, ImageGrab
import cv2
import time
import threading
import os
from collections import deque

from assets.camera.manager import CameraManager
from assets.detection.detection_worker import DetectionWorker
from assets.recording.recorder import Recorder
from assets.visualization import overlay
from assets.ui.file_dialogs import choose_file, choose_directory
from assets.media.video_source import VideoFileSource
from assets.media.image_source import ImageFileSource
from gui.config_window import ConfigWindow
from gui.view_controls import ViewControls
from gui.left_sidebar import LeftSidebar
from gui.media_view import MediaView
from gui.input_dialogs import ChoiceWindow
from gui.vision_task_window import VisionTaskWindow, ArchitectureWindow
from models.factory import create_model

NOTICE_COLORS = {
    "idle": "gray70",
    "ok": "#4ade80",
    "warn": "#fbbf24",
    "error": "#f87171",
}

class MainApp(ctk.CTkFrame):
    def __init__(self, master, settings):
        super().__init__(master, fg_color="transparent")
        self.master = master
        self.settings = settings
        self.camera_manager = CameraManager()
        self.camera_source = None
        self.media_kind = None
        self.media_source = None
        self.media_frame = None
        self.media_finished = False
        self.model = None
        self.detection_worker = None
        self._last_detection_error = None
        self.recorder = Recorder()
        self.current_frame = None
        self._raw_frame = None
        self._display_frame = None
        self._detection_frame = None
        self.fps = 0.0
        self._last_frame_time = None
        self._frame_after_id = None
        self._display_timestamps = deque(maxlen=120)
        self.config_window = None
        self.display_scale = 1.0
        self.flip_vertical_enabled = False
        self.flip_horizontal_enabled = False
        self.rotation_angle = 0
        self.view_mode = "rgb"
        self.threshold_method = None
        self.binary_thresh, self.binary_maxval = 127, 255
        self.otsu_maxval, self.otsu_invert = 255, False
        self.adaptive_method = "mean"
        self.adaptive_maxval, self.adaptive_block_size, self.adaptive_c = 255, 11, 2
        self.hsv_h_min, self.hsv_h_max = 0, 179
        self.hsv_s_min, self.hsv_s_max = 0, 255
        self.hsv_v_min, self.hsv_v_max = 0, 255
        self.hsl_h_min, self.hsl_h_max = 0, 179
        self.hsl_l_min, self.hsl_l_max = 0, 255
        self.hsl_s_min, self.hsl_s_max = 0, 255
        self.vision_task = None
        self.model_architecture = None
        self.model_path = None
        self._classification_text = []
        self._depth_text = []
        self._last_result = {}
        self.recording_start_time = None
        self._recording_after_id = None
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.left_sidebar = LeftSidebar(self, self)
        self._build_video_area()

    def _build_video_area(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.media_view = MediaView(self, self)
        self.video_frame = self.media_view.outer
        self.video_label = self.media_view.label
        self.NOTICE_MARGIN = 40
        self.notice_bar = ctk.CTkLabel(
            self.video_frame,
            text="",
            font=ctk.CTkFont(size=20),
            fg_color="transparent",
            text_color=NOTICE_COLORS["idle"],
            justify="center",
        )
        self.notice_bar.place(
            relx=0.5,
            rely=1.0,
            anchor="center",
            y=-(self.media_view.bottom_gap // 2),
        )
        self.video_frame.bind("<Configure>", self._update_notice_wraplength)
        self.after(50, self._update_notice_wraplength)
        self.right_sidebar_container = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color="#1a1d23")
        self.right_sidebar_container.grid(row=0, column=2, sticky="nse")
        self.right_sidebar_container.grid_propagate(False)
        self.right_sidebar_container.grid_rowconfigure(0, weight=1)
        self.right_sidebar_container.grid_columnconfigure(0, weight=1)
        self.right_sidebar = ViewControls(self.right_sidebar_container, self)
        self.right_sidebar.grid(row=0, column=0, sticky="nsew")

    def _set_dot(self, dot, color_key):
        self.left_sidebar.set_dot(dot, color_key)

    def _short_path(self, path, max_len=30):
        if not path:
            return None
        return path if len(path) <= max_len else "..." + path[-(max_len - 3):]

    def _update_notice_wraplength(self, event=None):
        width = self.video_frame.winfo_width()
        if width > 1:
            self.notice_bar.configure(wraplength=max(200, width - self.NOTICE_MARGIN * 2))

    def _show_notice(self, text, color_key="ok"):
        self._notice_token = getattr(self, "_notice_token", 0) + 1
        token = self._notice_token
        self.notice_bar.configure(
            text=text,
            text_color=NOTICE_COLORS.get(color_key, NOTICE_COLORS["idle"]),
        )
        self.notice_bar.lift()
        self._notice_priority_until = time.perf_counter() + 4.0
        def _clear():
            if self._notice_token == token:
                self.notice_bar.configure(text="")
        self.after(4000, _clear)

    def _update_classification_notice(self, text):
        # Low-priority, continuously-refreshed readout - never steals the
        # notice bar away from an explicit _show_notice still in its
        # display window (camera/recording/error messages, etc).
        if time.perf_counter() < getattr(self, "_notice_priority_until", 0):
            return
        self.notice_bar.configure(text=text, text_color=NOTICE_COLORS["idle"])
        self.notice_bar.lift()

    def select_camera(self):
        picker = ChoiceWindow(
            self.master,
            "Inference Input",
            "Select Inference Input",
            [
                ("live", "Live Camera"),
                ("video", "Video File"),
                ("image", "Image File"),
            ],
            self._select_input,
            width=360,
            height=290,
        )
        picker.grab_set()

    def _select_input(self, choice, picker):
        picker.destroy()
        if choice == "live":
            self._select_live_camera()
        elif choice == "video":
            self._select_video_file()
        else:
            self._select_image_file()

    def _select_live_camera(self):
        self.media_kind = "live"
        self._show_notice("Scanning for cameras...", "warn")
        self.camera_manager.scan_async(self._on_scan_complete)

    def _on_scan_complete(self, options):
        if not options:
            self._set_dot(self.left_sidebar.camera_dot, "error")
            self._show_notice("No camera detected", "error")
            return
        if len(options) == 1:
            self._start_camera(options[0])
            return
        picker = ctk.CTkToplevel(self.master)
        picker.title("Select Camera")
        picker.geometry("300x240")
        ctk.CTkLabel(picker, text="Multiple cameras found", font=ctk.CTkFont(size=14, weight="bold")).pack(padx=12, pady=(16,8))
        for opt in options:
            ctk.CTkButton(picker, text=opt, command=lambda o=opt: (self._start_camera(o), picker.destroy())).pack(fill="x", padx=16, pady=4)

    def _start_camera(self, choice):
        try:
            self._stop_current_source()
            self.camera_source = self.camera_manager.build(choice)
        except Exception as e:
            self._set_dot(self.left_sidebar.camera_dot, "error")
            self._show_notice(str(e), "error")
            return
        self.media_kind = "live"
        self.left_sidebar.camera_btn.configure(state="normal", text="Select input")
        self._set_dot(self.left_sidebar.camera_dot, "ok")
        self._show_notice(f"Camera ready: {choice}", "ok")
        self.media_view.show_file_controls(False)
        self.media_finished = False
        self._display_timestamps.clear()
        self.fps = 0.0
        self.update_frame()

    def _select_video_file(self):
        path = choose_file("Select Video File", pattern="*.mp4 *.avi *.mov *.mkv *.webm", pattern_label="Video Files")
        if not path:
            return
        try:
            self._stop_current_source()
            source = VideoFileSource(path)
            source.start()
            self.media_source = source
            self.media_kind = "video"
            self.media_frame = None
            self.media_finished = False
            self.left_sidebar.camera_btn.configure(state="normal", text="Select input")
            self._set_dot(self.left_sidebar.camera_dot, "ok")
            self._show_notice(f"Video loaded: {os.path.basename(path)}", "ok")
            self.media_view.show_file_controls(True)
            self.media_view.set_paused(False)
            self.media_view.set_slider(0, source.frame_count)
            self._display_timestamps.clear()
            self.fps = 0.0
            self.update_frame()
        except Exception as e:
            self._set_dot(self.left_sidebar.camera_dot, "error")
            self._show_notice(str(e), "error")

    def _select_image_file(self):
        path = choose_file("Select Image File", pattern="*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp", pattern_label="Image Files")
        if not path:
            return
        try:
            self._stop_current_source()
            source = ImageFileSource(path)
            source.start()
            self.media_source = source
            self.media_kind = "image"
            self.media_frame = source.image.copy()
            self.media_finished = False
            self.left_sidebar.camera_btn.configure(state="normal", text="Select input")
            self._set_dot(self.left_sidebar.camera_dot, "ok")
            self._show_notice(f"Image loaded: {os.path.basename(path)}", "ok")
            self.media_view.show_file_controls(False)
            self.media_view.play_button.configure(state="disabled")
            self.media_view.slider.configure(state="disabled")
            self._display_timestamps.clear()
            self.fps = 0.0
            self.update_frame()
        except Exception as e:
            self._set_dot(self.left_sidebar.camera_dot, "error")
            self._show_notice(str(e), "error")

    def _stop_current_source(self):
        if self._frame_after_id is not None:
            try:
                self.after_cancel(self._frame_after_id)
            except Exception:
                pass
            self._frame_after_id = None
        if self.camera_source is not None:
            try:
                self.camera_source.stop()
            except Exception:
                pass
        if self.media_source is not None:
            try:
                self.media_source.stop()
            except Exception:
                pass
        self.camera_source = None
        self.media_source = None

    def select_vision_task(self):
        VisionTaskWindow(self.master, self._on_task_selected)

    def _on_task_selected(self, task):
        if self.vision_task != task and self.model is not None:
            self._unload_model()
        self.vision_task = task
        self.model_architecture = None
        self.model_path = None
        self._last_detection_error = None
        self._classification_text = []
        self._set_dot(self.left_sidebar.model_dot, "warn")
        self._show_notice(f"Vision task selected: {task}", "warn")
        self.left_sidebar.model_weight_btn.configure(state="normal")
        self.left_sidebar.unload_tasks_btn.configure(state="disabled")
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window._refresh_from_settings()

    def select_model_weight(self):
        if self.vision_task is None:
            self._show_notice("Select a vision task first", "warn")
            VisionTaskWindow(self.master, self._on_task_selected)
            return
        ArchitectureWindow(self.master, self.vision_task, self._on_architecture_selected)

    def _on_architecture_selected(self, task, architecture):
        if task != self.vision_task:
            return
        path = choose_file("Select Model Weight", pattern="*.pt *.pth *.onnx *.safetensors", pattern_label="Model Weights")
        if not path:
            return
        self._set_dot(self.left_sidebar.model_dot, "warn")
        self._show_notice(f"Loading {architecture} model...", "warn")
        self.left_sidebar.model_weight_btn.configure(state="disabled")
        threading.Thread(target=self._load_model_async, args=(task, architecture, path), daemon=True).start()

    def _load_model_async(self, task, architecture, path):
        try:
            model = create_model(task, architecture, path)
            self.after(0, lambda: self._on_model_loaded(model, task, architecture, path))
        except Exception:
            self.after(0, lambda: self._on_model_load_failed())

    def _on_model_loaded(self, model, task, architecture, path):
        self._unload_model()
        self.model = model
        self.detection_worker = DetectionWorker(model)
        self.vision_task = task
        self.model_architecture = architecture
        self.model_path = path
        if task == "classification":
            self.settings.set("tracker_mode", "none")
        self._last_detection_error = None
        self._set_dot(self.left_sidebar.model_dot, "ok")
        self._show_notice(f"Model loaded: {os.path.basename(path)}", "ok")
        self.left_sidebar.model_weight_btn.configure(state="normal")
        self.left_sidebar.unload_tasks_btn.configure(state="normal")
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window._refresh_from_settings()

    def _on_model_load_failed(self):
        self._show_notice("Model loading failed", "error")
        self.left_sidebar.model_weight_btn.configure(state="normal")
        self.left_sidebar.unload_tasks_btn.configure(state="disabled")
        self._set_dot(self.left_sidebar.model_dot, "error")

    def _unload_model(self):
        if self.detection_worker:
            self.detection_worker.stop()
            self.detection_worker = None
        if self.model is not None:
            try:
                self.model.close()
            except Exception:
                pass
        self.model = None
        self.model_architecture = None
        self.model_path = None
        self._last_result = {}
        self._classification_text = []

    def unload_all_tasks(self):
        self._unload_model()
        self.vision_task = None
        self._last_detection_error = None
        self._set_dot(self.left_sidebar.model_dot, "idle")
        self.left_sidebar.model_weight_btn.configure(state="disabled")
        self.left_sidebar.unload_tasks_btn.configure(state="disabled")
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window._refresh_from_settings()
        self._show_notice("All vision tasks unloaded", "idle")

    def open_configuration(self):
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window.lift()
            self.config_window.focus_force()
            return
        self.config_window = ConfigWindow(self.master, self.settings, get_task_fn=lambda: self.vision_task)

    def _current_tracker(self):
        if self.vision_task == "classification":
            return "none"
        return self.settings.get("tracker_mode") or "none"

    def select_video_folder(self):
        folder = choose_directory("Select folder to save VIDEO recordings")
        if not folder:
            return
        self.settings.set("video_output_dir", folder)
        self.left_sidebar.video_folder_label.configure(text=self._short_path(folder))
        self._show_notice("Video folder set", "ok")

    def select_screenshot_folder(self):
        folder = choose_directory("Select folder to save SCREENSHOTS")
        if not folder:
            return
        self.settings.set("screenshot_output_dir", folder)
        self.left_sidebar.screenshot_folder_label.configure(text=self._short_path(folder))
        self._show_notice("Screenshot folder set", "ok")

    def toggle_recording(self):
        if not self.recorder.recording:
            video_dir = self.settings.get("video_output_dir")
            if not video_dir:
                self._show_notice("Select a video folder first", "error")
                return
            if self._raw_frame is None:
                self._show_notice("No camera frame available to record", "error")
                return
            frame = self._detection_frame.copy() if self._detection_frame is not None else self._raw_frame.copy()
            self.recorder.start_recording(frame, video_dir)
            self.left_sidebar.record_btn.configure(text="Stop Recording")
            self.recording_start_time = time.time()
            self._show_notice("Recording started", "warn")
            self._update_recording_notice()
        else:
            self.recorder.stop_recording()
            self.left_sidebar.record_btn.configure(text="Start Recording")
            if self._recording_after_id is not None:
                self.after_cancel(self._recording_after_id)
                self._recording_after_id = None
            self.recording_start_time = None
            self._show_notice("Recording saved", "ok")

    def _update_recording_notice(self):
        if self.recording_start_time is None:
            return
        elapsed = int(time.time() - self.recording_start_time)
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs else f"{mins:02d}:{secs:02d}"
        self.left_sidebar.record_btn.configure(text=f"Recording: {time_str}")
        self.notice_bar.configure(text=f"Recording: {time_str}", text_color=NOTICE_COLORS["warn"])
        self.notice_bar.lift()
        self._recording_after_id = self.after(1000, self._update_recording_notice)

    def take_screenshot(self):
        screenshot_dir = self.settings.get("screenshot_output_dir")
        if not screenshot_dir:
            self._show_notice("Select a screenshot folder first", "error")
            return
        try:
            target = self.media_view.label
            target.update_idletasks()
            x = target.winfo_rootx()
            y = target.winfo_rooty()
            w = target.winfo_width()
            h = target.winfo_height()
            if w <= 1 or h <= 1:
                return
            frame = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            path = self._next_screenshot_path(screenshot_dir)
            frame.save(path)
            self._show_notice(f"Screenshot saved: {os.path.basename(path)}", "ok")
        except Exception:
            frame = self._detection_frame
            if frame is not None:
                path = self.recorder.save_screenshot(frame, screenshot_dir)
                self._show_notice(f"Screenshot saved: {os.path.basename(path)}", "ok")
            else:
                self._show_notice("No camera frame available", "error")

    def take_screenshot_clean(self):
        screenshot_dir = self.settings.get("screenshot_output_dir")
        if not screenshot_dir:
            self._show_notice("Select a screenshot folder first", "error")
            return
        if self._raw_frame is None:
            self._show_notice("No camera frame available", "error")
            return
        path = self.recorder.save_screenshot(self._transform_display_image(self._raw_frame.copy()), screenshot_dir)
        self._show_notice(f"Camera frame saved: {os.path.basename(path)}", "ok")

    def take_screenshot_boxes_only(self):
        screenshot_dir = self.settings.get("screenshot_output_dir")
        if not screenshot_dir:
            self._show_notice("Select a screenshot folder first", "error")
            return
        if self._raw_frame is None:
            self._show_notice("No camera frame available", "error")
            return
        frame = self._transform_display_image(self._raw_frame.copy())
        self._draw_result(frame, self._last_result or {}, scale=1.0, boxes_only=True)
        path = self.recorder.save_screenshot(frame, screenshot_dir)
        self._show_notice(f"Detection UI screenshot saved: {os.path.basename(path)}", "ok")

    def _next_screenshot_path(self, folder):
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png")

    def toggle_media_pause(self):
        if self.media_kind != "video" or self.media_source is None:
            return
        paused = self.media_source.toggle_pause()
        self.media_view.set_paused(paused)
        self._show_notice("Video paused" if paused else "Video resumed", "warn" if paused else "ok")

    def seek_media(self, value):
        if self.media_kind != "video" or self.media_source is None:
            return
        if not self.media_source.paused:
            return
        frame = self.media_source.seek(int(float(value)))
        if frame is not None:
            self.media_frame = frame
            self._process_and_display(frame, None, None, None)
            self._show_notice(f"Video position: {int(float(value)) + 1}", "idle")

    def _read_source(self):
        if self.media_kind == "live":
            if self.camera_source is None:
                return None, None, None, None
            return self.camera_source.read()
        if self.media_kind == "video":
            if self.media_source is None:
                return None, None, None, None
            if self.media_source.paused and self.media_frame is not None:
                return self.media_frame.copy(), None, None, None
            frame, depth, cx, cy = self.media_source.read()
            if frame is None:
                try:
                    self.media_source.seek(0)
                    self.media_source.paused = False
                    self.media_finished = False
                    self.media_view.set_paused(False)
                    frame, depth, cx, cy = self.media_source.read()
                except Exception:
                    frame = None
            if frame is None:
                return self.media_frame, depth, cx, cy
            self.media_finished = False
            self.media_frame = frame.copy()
            self.media_view.set_slider(self.media_source.current_index, self.media_source.frame_count)
            return frame, depth, cx, cy
        if self.media_kind == "image":
            if self.media_source is None:
                return None, None, None, None
            return self.media_source.read()
        return None, None, None, None

    def update_frame(self):
        if self.media_kind is None:
            return
        start = time.perf_counter()
        raw_image, depth_frame, cx, cy = self._read_source()
        if raw_image is not None:
            self._process_and_display(raw_image, depth_frame, cx, cy)
            now = time.perf_counter()
            self._display_timestamps.append(now)
            cutoff = now - 1.0
            while self._display_timestamps and self._display_timestamps[0] < cutoff:
                self._display_timestamps.popleft()
            if len(self._display_timestamps) >= 2:
                elapsed = self._display_timestamps[-1] - self._display_timestamps[0]
                self.fps = (len(self._display_timestamps) - 1) / elapsed if elapsed > 0 else 0.0
            elif len(self._display_timestamps) == 1:
                self.fps = 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if self.media_kind == "video" and self.media_source is not None and not self.media_source.paused:
            source_fps = max(float(getattr(self.media_source, "fps", 30.0) or 30.0), 1.0)
            delay = max(1, int(1000.0 / source_fps - elapsed_ms))
        else:
            delay = 1
        self._frame_after_id = self.after(delay, self.update_frame)

    def _process_and_display(self, raw_image, depth_frame, cx, cy):
        self._raw_frame = raw_image
        if self.model is None or self.detection_worker is None:
            result = {}
        else:
            result = self._infer(raw_image)
        self._last_result = result
        display_image = self._transform_display_image(raw_image.copy())
        scale = self._display_scale_for(display_image, raw_image)
        if scale != 1.0:
            display_image = cv2.resize(display_image, (max(1, int(display_image.shape[1] * scale)), max(1, int(display_image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        self.display_scale = scale
        self._draw_result(display_image, result, scale=scale, depth_frame=depth_frame, cx=cx, cy=cy)
        self._display_frame = display_image
        self.media_view.set_image(display_image)
        if self.recorder.recording:
            native = self._transform_display_image(raw_image.copy())
            self._draw_result(native, result, scale=1.0, depth_frame=depth_frame, cx=cx, cy=cy)
            self._detection_frame = native
            self.recorder.write_frame(self._detection_frame)

    def _infer(self, frame):
        if self.model is None or self.detection_worker is None:
            self._classification_text = []
            return {}
        self.detection_worker.submit_frame(frame, tracker=self._current_tracker(), confidence_threshold=(self.settings.get("confidence_threshold") or 0) / 100.0)
        result = self.detection_worker.get_latest_result()
        error = self.detection_worker.get_latest_error()
        if error != self._last_detection_error:
            self._last_detection_error = error
            if error:
                self._show_notice(f"Tracking error: {error}", "error")
        return result or {}

    def _display_scale_for(self, image, raw_image):
        panel_w = self.media_view.outer.winfo_width() - self.media_view.side_margin * 2
        panel_h = self.media_view.outer.winfo_height() - self.media_view.top_margin - self.media_view.bottom_gap
        h,w=image.shape[:2]
        if panel_w < 10 or panel_h < 10:
            return 1.0
        return max(min(panel_w / w, panel_h / h) * self.media_view.size_shrink, 0.01)

    def _compute_axis_row(self, cy_disp, h_disp):
        """Row (in already-transformed+scaled DISPLAY coordinates) the
        horizontal axis line is drawn on. slider=0.5 keeps it exactly at
        cy_disp (true center); >0.5 moves it up toward row 0, <0.5 moves
        it down toward the bottom edge. The vertical line's column is
        never touched by this - see _draw_result."""
        slider = self.settings.get("axis_line_slider")
        if slider is None:
            slider = 0.5
        t = (slider - 0.5) * 2
        if t >= 0:
            row = cy_disp - t * cy_disp
        else:
            row = cy_disp + (-t) * (h_disp - 1 - cy_disp)
        return int(round(row))

    def _draw_result(self, image, result, scale=1.0, depth_frame=None, cx=None, cy=None, boxes_only=False):
        kind = result.get("type")
        tracking = self._current_tracker() != "none"
        if kind == "classification":
            classes = result.get("classes", [])
            if classes and not boxes_only:
                label = classes[0]["label"]
                self._update_classification_notice(f"Class: {label}")
        elif kind == "detection":
            for item in result.get("detections", []):
                if tracking or self.vision_task == "detection":
                    self._draw_box_transformed(image, item, scale)
        elif kind == "instance_segmentation":
            segments = result.get("segments", [])
            transformed = self._transform_segments(segments)
            computed = overlay.draw_instance_masks(image, transformed, scale=scale)
            if tracking:
                for item in computed:
                    # box/color already come from the mask's own pixel extent
                    # (leftmost/rightmost/topmost/bottommost) in final image
                    # coordinates - draw the box only, the label is already
                    # shown by draw_instance_masks in the segment's own color.
                    self._draw_box_cosmetic(image, item, scale, draw_label=False)
        elif kind == "semantic_segmentation":
            mask = result.get("mask")
            mask = self._transform_mask(mask) if mask is not None else None
            overlay.draw_semantic_mask(image, mask)
        elif kind == "pose":
            poses = self._transform_poses(result.get("poses", []))
            for item in poses:
                bbox = overlay.draw_pose(image, item.get("keypoints"), scale)
                if tracking and bbox is not None:
                    copy = dict(item)
                    copy["box"] = bbox
                    self._draw_box_cosmetic(image, copy, scale, draw_label=True)
        if not boxes_only:
            if self.settings.get("axis_lines_on") and cx is not None and cy is not None and self._raw_frame is not None:
                raw_h, raw_w = self._raw_frame.shape[:2]
                disp_cx, disp_cy = self._transform_point(cx, cy, raw_w, raw_h)
                s_cx, s_cy = int(disp_cx * scale), int(disp_cy * scale)
                axis_row = self._compute_axis_row(s_cy, image.shape[0])
                overlay.draw_axes(image, s_cx, axis_row, thickness=max(1, round(scale)))
            left=[]
            if self.camera_source is not None and getattr(self.camera_source, "has_depth", False) and depth_frame is not None and cx is not None and cy is not None:
                try:
                    depth = self.camera_source.get_depth_meters(cx, cy, depth_frame)
                    if depth is not None:
                        left.append((f"Depth: {depth:.2f} Meter", (0,255,0)))
                except Exception:
                    pass
            if left:
                overlay.draw_left_text(image, left, scale=1.0)
            if self.settings.get("fps_viewer_on"):
                overlay.draw_fps(image, self.fps, scale=1.0)

    def _draw_box_transformed(self, image, item, scale):
        x1,y1,x2,y2=self._transform_box(*item["box"],self._raw_frame.shape[1],self._raw_frame.shape[0])
        copy=dict(item)
        copy["box"]=(int(x1*scale),int(y1*scale),int(x2*scale),int(y2*scale))
        self._draw_box_direct(image,copy,1.0)

    def _draw_box_direct(self,image,item,scale):
        box=item["box"]
        if scale != 1.0:
            box=tuple(int(v*scale) for v in box)
        overlay.draw_detection_box(image,box,item.get("label","object"),item.get("conf",0.0),scale=scale,track_id=item.get("track_id"))

    def _draw_box_cosmetic(self, image, item, cosmetic_scale, draw_label=True):
        # Box coordinates are already in final display-image pixel space
        # (derived from a mask's or pose's own pixel extent) - only the
        # stroke/font sizing should follow the display scale, not the box.
        overlay.draw_detection_box(
            image, item["box"], item.get("label", "object"), item.get("conf", 0.0),
            scale=cosmetic_scale, track_id=item.get("track_id"),
            color=item.get("color"), draw_label=draw_label,
        )

    def _transform_segments(self,segments):
        out=[]
        for item in segments:
            d=dict(item)
            d["box"]=self._transform_box(*item["box"],self._raw_frame.shape[1],self._raw_frame.shape[0])
            d["mask"]=self._transform_mask(item.get("mask"))
            out.append(d)
        return out

    def _transform_poses(self,poses):
        out=[]
        for item in poses:
            d=dict(item)
            d["box"]=self._transform_box(*item["box"],self._raw_frame.shape[1],self._raw_frame.shape[0])
            pts=[]
            for p in item.get("keypoints") or []:
                x,y=self._transform_point(p[0],p[1],self._raw_frame.shape[1],self._raw_frame.shape[0])
                pts.append((x,y))
            d["keypoints"]=pts
            out.append(d)
        return out

    def _transform_mask(self,mask):
        if mask is None:
            return None
        arr=(mask.astype("uint8")*255) if mask.dtype != "uint8" else mask
        if self.flip_horizontal_enabled:
            arr=cv2.flip(arr,1)
        if self.flip_vertical_enabled:
            arr=cv2.flip(arr,0)
        if self.rotation_angle==90:
            arr=cv2.rotate(arr,cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle==180:
            arr=cv2.rotate(arr,cv2.ROTATE_180)
        elif self.rotation_angle==270:
            arr=cv2.rotate(arr,cv2.ROTATE_90_COUNTERCLOCKWISE)
        return arr>127

    def _transform_display_image(self,image):
        if self.flip_horizontal_enabled:
            image=cv2.flip(image,1)
        if self.flip_vertical_enabled:
            image=cv2.flip(image,0)
        if self.rotation_angle==90:
            image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle==180:
            image=cv2.rotate(image,cv2.ROTATE_180)
        elif self.rotation_angle==270:
            image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.view_mode=="red":
            image=image.copy(); image[:,:,0]=0; image[:,:,1]=0
        elif self.view_mode=="green":
            image=image.copy(); image[:,:,0]=0; image[:,:,2]=0
        elif self.view_mode=="blue":
            image=image.copy(); image[:,:,1]=0; image[:,:,2]=0
        elif self.view_mode=="gray":
            image=cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        elif self.view_mode=="threshold":
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if self.threshold_method=="otsu":
                flag=cv2.THRESH_BINARY_INV if self.otsu_invert else cv2.THRESH_BINARY
                _,result=cv2.threshold(gray,0,self.otsu_maxval,flag+cv2.THRESH_OTSU)
            elif self.threshold_method=="adaptive":
                flag=cv2.ADAPTIVE_THRESH_GAUSSIAN_C if self.adaptive_method=="gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
                result=cv2.adaptiveThreshold(gray,self.adaptive_maxval,flag,cv2.THRESH_BINARY,self.adaptive_block_size,self.adaptive_c)
            else:
                _,result=cv2.threshold(gray,self.binary_thresh,self.binary_maxval,cv2.THRESH_BINARY)
            image=cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
        elif self.view_mode=="hsv":
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,(self.hsv_h_min,self.hsv_s_min,self.hsv_v_min),(self.hsv_h_max,self.hsv_s_max,self.hsv_v_max))
            image=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        elif self.view_mode=="hsl":
            hls=cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
            mask=cv2.inRange(hls,(self.hsl_h_min,self.hsl_l_min,self.hsl_s_min),(self.hsl_h_max,self.hsl_l_max,self.hsl_s_max))
            image=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        elif self.view_mode=="depth":
            if self.camera_source is None or not getattr(self.camera_source,"has_depth",False):
                self.view_mode="rgb"
                self._show_notice("No depth camera detected. Showing RGB stream.", "error")
        return image

    def _transform_point(self,x,y,width,height):
        if self.flip_horizontal_enabled:
            x=width-1-x
        if self.flip_vertical_enabled:
            y=height-1-y
        if self.rotation_angle==90:
            x,y=height-1-y,x
        elif self.rotation_angle==180:
            x,y=width-1-x,height-1-y
        elif self.rotation_angle==270:
            x,y=y,width-1-x
        return x,y

    def _transform_box(self,x1,y1,x2,y2,width,height):
        corners=[self._transform_point(x1,y1,width,height),self._transform_point(x2,y1,width,height),self._transform_point(x1,y2,width,height),self._transform_point(x2,y2,width,height)]
        xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
        return min(xs),min(ys),max(xs),max(ys)

    def _on_click(self,event):
        return

    def flip_vertical(self):
        self.flip_vertical_enabled=not self.flip_vertical_enabled
        self._show_notice(f"Vertical Flip: {'ON' if self.flip_vertical_enabled else 'OFF'}", "ok")

    def flip_horizontal(self):
        self.flip_horizontal_enabled=not self.flip_horizontal_enabled
        self._show_notice(f"Horizontal Flip: {'ON' if self.flip_horizontal_enabled else 'OFF'}", "ok")

    def rotate_cw(self):
        self.rotation_angle=(self.rotation_angle+90)%360
        self._show_notice(f"Rotation: {self.rotation_angle}°", "ok")

    def rotate_ccw(self):
        self.rotation_angle=(self.rotation_angle-90)%360
        self._show_notice(f"Rotation: {self.rotation_angle}°", "ok")

    def show_rgb_channel(self):
        self.view_mode="rgb"
        self._show_notice("RGB view enabled", "ok")

    def show_red_channel(self):
        self.view_mode="red"
        self._show_notice("Red channel enabled", "ok")

    def show_green_channel(self):
        self.view_mode="green"
        self._show_notice("Green channel enabled", "ok")

    def show_blue_channel(self):
        self.view_mode="blue"
        self._show_notice("Blue channel enabled", "ok")

    def show_grayscale(self):
        self.view_mode="gray"
        self._show_notice("Grayscale view enabled", "ok")

    def show_depth_channel(self):
        self.view_mode="depth"
        self._show_notice("Depth view enabled", "ok")

    def _open_singleton_window(self,attr,builder):
        win=getattr(self,attr,None)
        if win is not None and win.winfo_exists():
            win.lift(); win.focus_force(); return win
        win=builder(); setattr(self,attr,win); return win

    def _labeled_slider(self,parent,text,minimum,maximum,current,on_change,steps=None,pady=(10,2)):
        row=ctk.CTkFrame(parent,fg_color="transparent")
        row.pack(fill="x",padx=20,pady=pady)
        row.grid_columnconfigure(0,weight=1)
        ctk.CTkLabel(row,text=text,anchor="w").grid(row=0,column=0,sticky="w")
        value_label=ctk.CTkLabel(row,text=str(int(current)),width=44,anchor="e",font=ctk.CTkFont(weight="bold"),text_color="#4CC9F0")
        value_label.grid(row=0,column=1,sticky="e")
        def apply(value):
            on_change(value); value_label.configure(text=str(int(round(float(value)))))
        slider=ctk.CTkSlider(parent,from_=minimum,to=maximum,number_of_steps=steps or max(1,int(maximum-minimum)),command=apply)
        slider.set(current); slider.pack(fill="x",padx=20,pady=(0,4))
        return slider

    def open_threshold_settings(self):
        self._show_notice("Thresholding settings opened", "idle")
        def build():
            picker=ctk.CTkToplevel(self); picker.title("Select Thresholding Method"); picker.geometry("340x260"); picker.resizable(False,False)
            ctk.CTkLabel(picker,text="Choose a Thresholding Technique",font=ctk.CTkFont(size=15,weight="bold")).pack(padx=16,pady=(20,12))
            methods=[("Binary Thresholding",self._open_binary_threshold_window),("Otsu's Method",self._open_otsu_window),("Adaptive Thresholding",self._open_adaptive_window)]
            for label,opener in methods:
                ctk.CTkButton(picker,text=label,command=lambda o=opener,p=picker:(p.destroy(),o())).pack(fill="x",padx=20,pady=6)
            return picker
        self._open_singleton_window("threshold_picker_window",build)

    def _open_binary_threshold_window(self):
        self.view_mode,self.threshold_method="threshold","binary"
        self._show_notice("Binary thresholding enabled", "ok")
        def build():
            win=ctk.CTkToplevel(self); win.title("Thresholding - Binary"); win.geometry("380x260"); win.resizable(False,False)
            self._labeled_slider(win,"Threshold Value",0,255,self.binary_thresh,lambda v:setattr(self,"binary_thresh",int(v)),pady=(20,2))
            self._labeled_slider(win,"Maximum Value",0,255,self.binary_maxval,lambda v:setattr(self,"binary_maxval",int(v)))
            return win
        self._open_singleton_window("binary_window",build)

    def _open_otsu_window(self):
        self.view_mode,self.threshold_method="threshold","otsu"
        self._show_notice("Otsu thresholding enabled", "ok")
        def build():
            win=ctk.CTkToplevel(self); win.title("Thresholding - Otsu's Method"); win.geometry("380x240")
            ctk.CTkLabel(win,text="Otsu's method computes the split point automatically from the image histogram. Only the output level is configurable here.",font=ctk.CTkFont(size=12),text_color="gray60",justify="left",wraplength=340).pack(padx=20,pady=(18,4))
            self._labeled_slider(win,"Maximum Value",0,255,self.otsu_maxval,lambda v:setattr(self,"otsu_maxval",int(v)))
            invert=tk.BooleanVar(value=self.otsu_invert)
            ctk.CTkSwitch(win,text="Invert Intensities",variable=invert,onvalue=True,offvalue=False,command=lambda:setattr(self,"otsu_invert",bool(invert.get()))).pack(anchor="w",padx=20,pady=(8,4))
            return win
        self._open_singleton_window("otsu_window",build)

    def _open_adaptive_window(self):
        self.view_mode,self.threshold_method="threshold","adaptive"
        self._show_notice("Adaptive thresholding enabled", "ok")
        def build():
            win=ctk.CTkToplevel(self); win.title("Thresholding - Adaptive"); win.geometry("400x400"); win.resizable(False,False)
            ctk.CTkLabel(win,text="Local Method",anchor="w").pack(fill="x",padx=20,pady=(18,2))
            method=tk.StringVar(value=self.adaptive_method)
            ctk.CTkSegmentedButton(win,values=["Mean","Gaussian"],variable=method,command=lambda v:setattr(self,"adaptive_method",v.lower())).pack(fill="x",padx=20,pady=(0,8))
            self._labeled_slider(win,"Maximum Value",0,255,self.adaptive_maxval,lambda v:setattr(self,"adaptive_maxval",int(v)))
            self._labeled_slider(win,"Block Size (Odd, Neighborhood Width)",3,51,self.adaptive_block_size,self._adaptive_block_changed,steps=24)
            self._labeled_slider(win,"C (Constant Subtracted from Mean)",-50,50,self.adaptive_c,lambda v:setattr(self,"adaptive_c",int(v)))
            return win
        self._open_singleton_window("adaptive_window",build)

    def _adaptive_block_changed(self,value):
        block=int(round(float(value)))
        self.adaptive_block_size=block if block%2==1 else block+1

    def open_hsv_settings(self):
        self.view_mode="hsv"
        self._show_notice("HSV filter enabled", "ok")
        def build():
            win=ctk.CTkToplevel(self); win.title("HSV Filter"); win.geometry("420x560"); win.resizable(False,False)
            sliders=[("Hue Minimum",0,179,self.hsv_h_min,lambda v:setattr(self,"hsv_h_min",int(v))),("Hue Maximum",0,179,self.hsv_h_max,lambda v:setattr(self,"hsv_h_max",int(v))),("Saturation Minimum",0,255,self.hsv_s_min,lambda v:setattr(self,"hsv_s_min",int(v))),("Saturation Maximum",0,255,self.hsv_s_max,lambda v:setattr(self,"hsv_s_max",int(v))),("Value Minimum",0,255,self.hsv_v_min,lambda v:setattr(self,"hsv_v_min",int(v))),("Value Maximum",0,255,self.hsv_v_max,lambda v:setattr(self,"hsv_v_max",int(v)))]
            for args in sliders: self._labeled_slider(win,*args)
            return win
        self._open_singleton_window("hsv_window",build)

    def open_hsl_settings(self):
        self.view_mode="hsl"
        self._show_notice("HSL filter enabled", "ok")
        def build():
            win=ctk.CTkToplevel(self); win.title("HSL Filter"); win.geometry("420x560"); win.resizable(False,False)
            sliders=[("Hue Minimum",0,179,self.hsl_h_min,lambda v:setattr(self,"hsl_h_min",int(v))),("Hue Maximum",0,179,self.hsl_h_max,lambda v:setattr(self,"hsl_h_max",int(v))),("Lightness Minimum",0,255,self.hsl_l_min,lambda v:setattr(self,"hsl_l_min",int(v))),("Lightness Maximum",0,255,self.hsl_l_max,lambda v:setattr(self,"hsl_l_max",int(v))),("Saturation Minimum",0,255,self.hsl_s_min,lambda v:setattr(self,"hsl_s_min",int(v))),("Saturation Maximum",0,255,self.hsl_s_max,lambda v:setattr(self,"hsl_s_max",int(v)))]
            for args in sliders: self._labeled_slider(win,*args)
            return win
        self._open_singleton_window("hsl_window",build)

    def on_close(self):
        if self._frame_after_id is not None:
            try: self.after_cancel(self._frame_after_id)
            except Exception: pass
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window.destroy()
        self.recorder.stop_recording()
        self._stop_current_source()
        self._unload_model()

