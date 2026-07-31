import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import time
import threading
from datetime import datetime

from assets.camera.manager import CameraManager
from assets.detection.yolo_engine import YoloEngine
from assets.detection.detection_worker import DetectionWorker
from assets.communication.serial_com import ArduinoLink, SERIAL_AVAILABLE
from assets.recording.recorder import Recorder
from assets.visualization import overlay
from assets.transform.coordinates import pixel_distance
from assets.ui.file_dialogs import choose_file, choose_directory
from gui.config_window import ConfigWindow
from gui.view_controls import ViewControls

# private/ is .gitignored - these imports fail gracefully if it's missing,
# so the app still runs (RGB webcam mode) on a machine without it
try:
    from my_version.nozzle_targeting import NozzleTargeting
    from my_version.nozzle_bridge import compute_target

    PRIVATE_MODULE_AVAILABLE = True
except ImportError:
    NozzleTargeting = None
    compute_target = None
    PRIVATE_MODULE_AVAILABLE = False

# background color for the small persistent state dots (Camera/Model/Arduino)
DOT_COLORS = {
    "idle": "#6b7280",
    "ok": "#22c55e",
    "warn": "#f59e0b",
    "error": "#ef4444",
}

# text color for floating notices above the stream (no background box)
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
        self.camera_locked = False

        self.model = None
        self.detection_worker = None

        self.arduino = None
        self.nozzle = NozzleTargeting() if PRIVATE_MODULE_AVAILABLE else None
        self.recorder = Recorder()
        self.current_frame = None
        self._raw_frame = None  # clean frame (no overlay), refreshed every loop
        self.fps = 0.0
        self._last_frame_time = None
        self._last_plan = (
            None  # render plan for the same frame, for on-demand overlay variants
        )
        self.config_window = None

        self.mouse_x, self.mouse_y = 0, 0
        self.mouse_clicked = False  # True after first click in click mode
        self.display_scale = 1.0  # updated every frame by _scale_to_panel
        self.recording_start_time = None
        self._recording_after_id = None
        self.flip_vertical_enabled = False
        self.flip_horizontal_enabled = False
        self.rotation_angle = 0
        self.display_transform = True
        self.view_mode = "rgb"
        self.threshold_method = None  # "binary" | "otsu" | "adaptive"
        self.binary_thresh, self.binary_maxval = 127, 255
        self.otsu_maxval, self.otsu_invert = 255, False
        self.adaptive_method = "mean"  # "mean" | "gaussian"
        self.adaptive_maxval, self.adaptive_block_size, self.adaptive_c = 255, 11, 2
        self.hsv_h_min = 0
        self.hsv_h_max = 179
        self.hsv_s_min = 0
        self.hsv_s_max = 255
        self.hsv_v_min = 0
        self.hsv_v_max = 255
        self.hsl_h_min = 0
        self.hsl_h_max = 179
        self.hsl_l_min = 0
        self.hsl_l_max = 255
        self.hsl_s_min = 0
        self.hsl_s_max = 255
        self.has_thermal = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_video_area()

    # ------------------------------------------------------------ sidebar
    def _build_sidebar(self):
        sidebar_container = ctk.CTkFrame(
            self, width=260, corner_radius=0, fg_color="#1a1d23"
        )
        sidebar_container.grid(row=0, column=0, sticky="nsw")
        sidebar_container.grid_propagate(False)
        sidebar_container.grid_rowconfigure(0, weight=1)
        sidebar_container.grid_columnconfigure(0, weight=1)

        sidebar = ctk.CTkScrollableFrame(
            sidebar_container, corner_radius=0, fg_color="#1a1d23"
        )
        sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            sidebar, text="I/O Commands", font=ctk.CTkFont(size=20, weight="bold")
        ).pack(fill="x", padx=16, pady=(16, 12))

        # --- Camera ---
        self.camera_dot = self._section_label_with_dot(sidebar, "1. Camera")
        self.camera_btn = ctk.CTkButton(
            sidebar,
            text="Select Camera",
            command=self.select_camera,
            font=ctk.CTkFont(size=14),
        )
        self.camera_btn.pack(fill="x", padx=16, pady=(4, 2))

        # --- Model ---
        self.model_dot = self._section_label_with_dot(sidebar, "2. Detection Model")
        self.model_btn = ctk.CTkButton(
            sidebar,
            text="Select Model Weight (.pt)",
            command=self.select_model,
            font=ctk.CTkFont(size=14),
        )
        self.model_btn.pack(fill="x", padx=16, pady=(4, 2))
        self.skip_model_btn = ctk.CTkButton(
            sidebar,
            text="Skip Model (Click Mode)",
            command=self.skip_model,
            fg_color="#5e6471",
            hover_color="#4b5563",
            font=ctk.CTkFont(size=14),
        )
        self.skip_model_btn.pack(fill="x", padx=16, pady=(0, 2))

        # --- Arduino ---
        self.arduino_dot = self._section_label_with_dot(sidebar, "3. Arduino")
        self.arduino_btn = ctk.CTkButton(
            sidebar,
            text="Connect Arduino",
            command=self.toggle_arduino,
            fg_color="#065f46",
            hover_color="#047857",
            font=ctk.CTkFont(size=14),
        )
        self.arduino_btn.pack(fill="x", padx=16, pady=(4, 2))

        # --- Recording ---
        self._section_label(sidebar, "4. Recording")

        self.video_folder_btn = ctk.CTkButton(
            sidebar,
            text="Select Video Folder",
            command=self.select_video_folder,
            fg_color="transparent",
            border_width=1,
            border_color="gray40",
            font=ctk.CTkFont(size=14),
        )
        self.video_folder_btn.pack(fill="x", padx=16, pady=(4, 2))
        self.video_folder_label = ctk.CTkLabel(
            sidebar,
            text=self._short_path(self.settings.get("video_output_dir"))
            or "No folder selected",
            font=ctk.CTkFont(size=14),
            text_color="gray60",
            anchor="w",
            justify="left",
        )
        self.video_folder_label.pack(anchor="w", padx=16, pady=(0, 8), fill="x")

        self.record_btn = ctk.CTkButton(
            sidebar,
            text="Start Recording",
            command=self.toggle_recording,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            font=ctk.CTkFont(size=14),
        )
        self.record_btn.pack(fill="x", padx=16, pady=(0, 14))

        # --- Screenshot ---
        self._section_label(sidebar, "5. Screenshot")

        self.screenshot_folder_btn = ctk.CTkButton(
            sidebar,
            text="Select Image Folder",
            command=self.select_screenshot_folder,
            fg_color="transparent",
            border_width=1,
            border_color="gray40",
            font=ctk.CTkFont(size=14),
        )
        self.screenshot_folder_btn.pack(fill="x", padx=16, pady=(4, 2))
        self.screenshot_folder_label = ctk.CTkLabel(
            sidebar,
            text=self._short_path(self.settings.get("screenshot_output_dir"))
            or "No folder selected",
            font=ctk.CTkFont(size=14),
            text_color="gray60",
            anchor="w",
            justify="left",
        )
        self.screenshot_folder_label.pack(anchor="w", padx=16, pady=(0, 8), fill="x")

        self.screenshot_btn = ctk.CTkButton(
            sidebar,
            text="Window Screenshot",
            command=self.take_screenshot,
            fg_color="#065f46",
            hover_color="#047857",
            font=ctk.CTkFont(size=14),
        )
        self.screenshot_btn.pack(fill="x", padx=16, pady=(0, 2))

        self.screenshot_clean_btn = ctk.CTkButton(
            sidebar,
            text="Capture Cam. Frame",
            command=self.take_screenshot_clean,
            fg_color="#05523c",
            hover_color="#036247",
            font=ctk.CTkFont(size=14),
        )
        self.screenshot_clean_btn.pack(fill="x", padx=16, pady=(0, 2))

        self.screenshot_boxes_btn = ctk.CTkButton(
            sidebar,
            text="Detection UI Only",
            command=self.take_screenshot_boxes_only,
            fg_color="#033f2e",
            hover_color="#02553E",
            font=ctk.CTkFont(size=14),
        )
        self.screenshot_boxes_btn.pack(fill="x", padx=16, pady=(0, 14))

        # --- Configuration: pinned below the scrollable list (own footer
        # area, doesn't scroll away with everything else) ---
        footer = ctk.CTkFrame(sidebar_container, fg_color="#15181d", corner_radius=0)
        footer.grid(row=1, column=0, sticky="ew")
        self.config_btn = ctk.CTkButton(
            footer,
            text="Configuration",
            command=self.open_configuration,
            fg_color="#374151",
            hover_color="#4b5563",
            font=ctk.CTkFont(size=16),
        )
        self.config_btn.pack(fill="x", padx=16, pady=30)

    def _section_label(self, parent, text):
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="gray70",
        ).pack(anchor="w", padx=16, pady=(16, 2))

    def _section_label_with_dot(self, parent, text):
        """Create a section label with a status dot right after it."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(anchor="w", padx=16, pady=(16, 2))

        label = ctk.CTkLabel(
            container,
            text=text,
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="gray70",
        )
        label.pack(side="left")

        # Use a CTkFrame for the dot to have precise control over dimensions
        dot = ctk.CTkFrame(
            container, width=12, height=12, corner_radius=6, fg_color=DOT_COLORS["idle"]
        )
        dot.pack(side="left", padx=(7, 0))
        dot.pack_propagate(False)  # Prevent the frame from shrinking to fit contents
        return dot

    def _set_dot(self, dot, color_key):
        dot.configure(fg_color=DOT_COLORS[color_key])

    def _short_path(self, path, max_len=30):
        if not path:
            return None
        return path if len(path) <= max_len else "..." + path[-(max_len - 3) :]

    # --------------------------------------------------------- video area
    def _build_video_area(self):
        self.SIDE_MARGIN = 50
        self.TOP_MARGIN = 40
        self.BOTTOM_GAP = 130
        self.SIZE_SHRINK = 0.82

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)

        outer = ctk.CTkFrame(self, corner_radius=0, fg_color="#0b0d10")
        outer.grid(row=0, column=1, sticky="nsew")
        self.video_frame = outer

        self.video_label = tk.Label(
            outer,
            bg="#0b0d10",
            fg="white",
            text="Select a camera to begin",
            font=("Arial", 14),
        )

        self.video_label.place(
            relx=0.5,
            rely=0.5,
            anchor="center",
            y=(self.TOP_MARGIN - self.BOTTOM_GAP) // 2,
        )

        self.video_label.bind("<Button-1>", self._on_click)

        self.notice_bar = ctk.CTkLabel(
            outer,
            text="",
            font=ctk.CTkFont(size=20),
            fg_color="transparent",
            text_color=NOTICE_COLORS["idle"],
        )

        self.notice_bar.place(
            relx=0.5, rely=1.0, anchor="center", y=-(self.BOTTOM_GAP // 2)
        )

        self.right_sidebar_container = ctk.CTkFrame(
            self, width=320, corner_radius=0, fg_color="#1a1d23"
        )

        self.right_sidebar_container.grid(row=0, column=2, sticky="nse")

        self.right_sidebar_container.grid_propagate(False)
        self.right_sidebar_container.grid_rowconfigure(0, weight=1)
        self.right_sidebar_container.grid_columnconfigure(0, weight=1)

        self.right_sidebar = ViewControls(self.right_sidebar_container, self)

        self.right_sidebar.grid(row=0, column=0, sticky="nsew")

    def _show_notice(self, text, color_key="ok"):
        self.notice_bar.configure(
            text=text, text_color=NOTICE_COLORS.get(color_key, "gray70")
        )
        self.notice_bar.lift()
        self.after(4000, lambda: self.notice_bar.configure(text=""))

    # ----------------------------------------------------- camera (locked)
    def select_camera(self):
        if self.camera_locked:
            return
        self._show_notice("Scanning for cameras...", "warn")
        self.camera_manager.scan_async(self._on_scan_complete)

    def _on_scan_complete(self, options):
        if not options:
            self._set_dot(self.camera_dot, "error")
            self._show_notice("No camera detected", "error")
            return

        if len(options) == 1:
            self._start_camera(options[0])
            return

        picker = ctk.CTkToplevel(self.master)
        picker.title("Select Camera")
        picker.geometry("300x240")
        ctk.CTkLabel(
            picker,
            text="Multiple cameras found",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(padx=12, pady=(16, 8))
        for opt in options:
            ctk.CTkButton(
                picker,
                text=opt,
                command=lambda o=opt: (self._start_camera(o), picker.destroy()),
            ).pack(fill="x", padx=16, pady=4)

    def _start_camera(self, choice):
        try:
            self.camera_source = self.camera_manager.build(choice)
        except Exception as e:
            self._set_dot(self.camera_dot, "error")
            self._show_notice(str(e), "error")
            return

        self.camera_locked = True
        self.camera_btn.configure(state="disabled", text="Camera Locked")
        self._set_dot(self.camera_dot, "ok")
        self._show_notice(f"Camera ready: {choice}", "ok")
        self.mouse_clicked = False  # reset click flag for new camera
        self.update_frame()

    # ------------------------------------------------------ model
    def select_model(self):
        path = choose_file(
            "Select Model Weight", pattern="*.pt", pattern_label="PyTorch weights"
        )
        if not path:
            return
        self._show_notice("Loading model...", "warn")
        threading.Thread(
            target=self._load_model_async, args=(path,), daemon=True
        ).start()

    def _load_model_async(self, path):
        try:
            model = YoloEngine(path)
        except Exception as e:
            self.after(0, lambda: self._on_model_load_failed(str(e)))
            return
        self.after(0, lambda: self._on_model_loaded(model, path))

    def _on_model_loaded(self, model, path):
        if self.detection_worker:
            self.detection_worker.stop()
        self.model = model
        self.detection_worker = DetectionWorker(model)
        self._set_dot(self.model_dot, "ok")
        self._show_notice(f"Model loaded: {path.split('/')[-1]}", "ok")

    def _on_model_load_failed(self, message):
        self._set_dot(self.model_dot, "error")
        self._show_notice(message, "error")

    def skip_model(self):
        if self.detection_worker:
            self.detection_worker.stop()
            self.detection_worker = None
        self.model = None
        self._set_dot(self.model_dot, "warn")
        self._show_notice("Click mode enabled (no detection)", "warn")
        self.mouse_clicked = False  # reset click flag when entering click mode

    # --------------------------------------------------------- configuration
    def open_configuration(self):
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window.lift()
            self.config_window.focus_force()
            return
        self.config_window = ConfigWindow(
            self.master, self.settings, has_model_fn=lambda: self.model is not None
        )

    # --------------------------------------------------- arduino (toggle)
    def toggle_arduino(self):
        if self.arduino is None or not self.arduino.is_connected:
            if not SERIAL_AVAILABLE:
                self._show_notice("Install pyserial: pip install pyserial", "error")
                return
            self.arduino = ArduinoLink(
                port=self.settings.get("arduino_port"),
                baud=self.settings.get("arduino_baud"),
            )
            try:
                self.arduino.connect()
                self.arduino_btn.configure(
                    text="Disconnect Arduino", fg_color="#7f1d1d", hover_color="#991b1b"
                )
                self._set_dot(self.arduino_dot, "ok")
                self._show_notice("Arduino connected", "ok")
            except Exception as e:
                self._set_dot(self.arduino_dot, "error")
                self._show_notice(str(e), "error")
        else:
            self.arduino.disconnect()
            self.arduino_btn.configure(
                text="Connect Arduino", fg_color="#065f46", hover_color="#047857"
            )
            self._set_dot(self.arduino_dot, "idle")
            self._show_notice("Arduino disconnected", "idle")

    # -------------------------------------------------------- recording
    def select_video_folder(self):
        folder = choose_directory("Select folder to save VIDEO recordings")
        if not folder:
            return
        self.settings.set("video_output_dir", folder)
        self.video_folder_label.configure(text=self._short_path(folder))
        self._show_notice("Video folder set", "ok")

    def select_screenshot_folder(self):
        folder = choose_directory("Select folder to save SCREENSHOTS")
        if not folder:
            return
        self.settings.set("screenshot_output_dir", folder)
        self.screenshot_folder_label.configure(text=self._short_path(folder))
        self._show_notice("Screenshot folder set", "ok")

    def toggle_recording(self):
        if not self.recorder.recording:
            video_dir = self.settings.get("video_output_dir")
            if not video_dir:
                self._show_notice("Select a video folder first", "error")
                return
            if self.current_frame is not None:
                self.recorder.start_recording(self.current_frame, video_dir)
                self.record_btn.configure(text="Stop Recording")
                self.recording_start_time = time.time()
                self._update_recording_notice()
        else:
            self.recorder.stop_recording()
            self.record_btn.configure(text="Start Recording")
            if self._recording_after_id is not None:
                self.after_cancel(self._recording_after_id)
                self._recording_after_id = None
            self.recording_start_time = None
            self._show_notice("Recording saved", "ok")

    def _update_recording_notice(self):
        """Update the notice bar with elapsed recording time."""
        if self.recording_start_time is None:
            return
        elapsed = int(time.time() - self.recording_start_time)
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        if hrs:
            time_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        else:
            time_str = f"{mins:02d}:{secs:02d}"
        self.notice_bar.configure(
            text=f"Recording: {time_str}", text_color=NOTICE_COLORS["warn"]
        )
        self.notice_bar.lift()
        # schedule next update
        self._recording_after_id = self.after(1000, self._update_recording_notice)

    def take_screenshot(self):
        """Full overlay - exactly what's on screen / recorded to video."""
        self._save_screenshot(self.current_frame)

    def take_screenshot_clean(self):
        """No UI elements at all - the raw camera frame."""
        if self._raw_frame is None:
            return
        frame = self._transform_display_image(self._raw_frame.copy())
        self._save_screenshot(frame)

    def take_screenshot_boxes_only(self):
        """Detection boxes + labels (or the click dot, in click-mode) -
        no crosshair, no diagonal line, no text overlay."""
        if self._raw_frame is None or self._last_plan is None:
            return
        frame = self._transform_display_image(self._raw_frame.copy())
        self._apply_render_plan(frame, self._last_plan, scale=1.0, mode="boxes_only")
        self._save_screenshot(frame)

    def _save_screenshot(self, frame):
        screenshot_dir = self.settings.get("screenshot_output_dir")
        if not screenshot_dir:
            self._show_notice("Select a screenshot folder first", "error")
            return
        if frame is not None:
            path = self.recorder.save_screenshot(frame, screenshot_dir)
            self._show_notice(f"Screenshot saved: {path.split('/')[-1]}", "ok")

    # ----------------------------------------------------------- mouse
    def _on_click(self, event):
        if self.current_frame is None:
            return
        display_x = int(event.x / self.display_scale)
        display_y = int(event.y / self.display_scale)
        raw_h, raw_w = self.current_frame.shape[:2]
        raw_x, raw_y = self._inverse_transform_point(display_x, display_y, raw_w, raw_h)
        self.mouse_x = raw_x
        self.mouse_y = raw_y

        self.mouse_clicked = True

    # ------------------------------------------------------- frame loop
    def update_frame(self):
        if self.camera_source is None:
            return

        raw_image, depth_frame, cx, cy = self.camera_source.read()
        self.raw_width = raw_image.shape[1]
        self.raw_height = raw_image.shape[0]

        now = time.perf_counter()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = inst_fps if self.fps == 0 else (self.fps * 0.9 + inst_fps * 0.1)
        self._last_frame_time = now

        if raw_image is not None:
            # Compute everything ONCE - detection inference, nozzle math,
            # and any Arduino send all happen exactly one time per frame,
            # regardless of how many times the result gets drawn below.
            plan = self._build_render_plan(raw_image, depth_frame, cx, cy)
            self._raw_frame = (
                raw_image.copy()
            )  # clean, no overlay - for screenshot variants
            self._last_plan = plan

            # Native-resolution pass: what gets recorded/screenshotted -
            # unchanged from before, same resolution as the raw camera feed.
            record_image = raw_image.copy()
            record_image = self._transform_display_image(record_image)
            self._apply_render_plan(record_image, plan, scale=1.0)
            self.current_frame = record_image
            self.recorder.write_frame(record_image)

            # Display pass: resize the CLEAN raw frame first, then draw the
            # same plan directly at that resolution (scaled coordinates,
            # thicker lines, bigger text) instead of upscaling an already-
            # rasterized overlay - this is what keeps text/lines crisp.
            display_image, scale = self._scale_raw_to_panel(raw_image)
            display_image = self._transform_display_image(display_image)
            self._apply_render_plan(display_image, plan, scale=scale)

            rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.imgtk = imgtk  # keep a reference
            self.video_label.configure(image=imgtk, text="")

        self.after(15, self.update_frame)

    def _scale_raw_to_panel(self, image):
        """Resizes a CLEAN (no overlay) frame using the margins/gap from
        _build_video_area, and returns (resized_image, scale_used). Pure
        resize - no drawing happens here."""
        panel_w = self.video_frame.winfo_width() - (self.SIDE_MARGIN * 2)
        panel_h = self.video_frame.winfo_height() - self.TOP_MARGIN - self.BOTTOM_GAP
        img_h, img_w = image.shape[:2]

        if panel_w < 10 or panel_h < 10:
            self.display_scale = 1.0
            return image.copy(), 1.0

        fit_scale = min(panel_w / img_w, panel_h / img_h)
        scale = max(fit_scale * self.SIZE_SHRINK, 0.01)
        self.display_scale = scale

        new_w, new_h = int(img_w * scale), int(img_h * scale)
        return cv2.resize(image, (new_w, new_h)), scale

    def _transform_display_image(self, image):
        """
        Transform ONLY the displayed camera image.

        Overlay elements are NOT transformed here.
        They will be redrawn later using transformed coordinates.
        """

        if self.flip_horizontal_enabled:
            image = cv2.flip(image, 1)

        if self.flip_vertical_enabled:
            image = cv2.flip(image, 0)

        if self.rotation_angle == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        elif self.rotation_angle == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)

        elif self.rotation_angle == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.view_mode == "red":
            image = image.copy()
            image[:, :, 0] = 0
            image[:, :, 1] = 0

        elif self.view_mode == "green":
            image = image.copy()
            image[:, :, 0] = 0
            image[:, :, 2] = 0

        elif self.view_mode == "blue":
            image = image.copy()
            image[:, :, 1] = 0
            image[:, :, 2] = 0

        elif self.view_mode == "gray":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif self.view_mode == "threshold":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.threshold_method == "otsu":
                flag = cv2.THRESH_BINARY_INV if self.otsu_invert else cv2.THRESH_BINARY
                _, result = cv2.threshold(gray, 0, self.otsu_maxval, flag + cv2.THRESH_OTSU)
            elif self.threshold_method == "adaptive":
                method_flag = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if self.adaptive_method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
                result = cv2.adaptiveThreshold(gray, self.adaptive_maxval, method_flag, cv2.THRESH_BINARY, self.adaptive_block_size, self.adaptive_c)
            else:
                _, result = cv2.threshold(gray, self.binary_thresh, self.binary_maxval, cv2.THRESH_BINARY)
            image = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elif self.view_mode == "hsv":

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower = (
                self.hsv_h_min,
                self.hsv_s_min,
                self.hsv_v_min
            )

            upper = (
                self.hsv_h_max,
                self.hsv_s_max,
                self.hsv_v_max
            )

            mask = cv2.inRange(hsv, lower, upper)

            image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        elif self.view_mode == "hsl":

            hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

            lower = (
                self.hsl_h_min,
                self.hsl_l_min,
                self.hsl_s_min
            )

            upper = (
                self.hsl_h_max,
                self.hsl_l_max,
                self.hsl_s_max
            )

            mask = cv2.inRange(hls, lower, upper)

            image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        elif self.view_mode == "depth":
            if self.camera_source is not None and getattr(self.camera_source, "has_depth", False):
                pass
            else:
                self.view_mode = "rgb"
                self._show_notice("No depth camera detected. Showing RGB stream.", "error")

        elif self.view_mode == "thermal":
            if self.has_thermal:
                pass
            else:
                self.view_mode = "rgb"
                self._show_notice("No thermal camera detected. Showing RGB stream.", "error")

        return image

    def _transform_point(self, x, y, width, height):
        """
        Transform one pixel coordinate according to the current
        display orientation.

        This transforms coordinates only.

        Nothing is drawn here.
        """

        if self.flip_horizontal_enabled:
            x = width - 1 - x

        if self.flip_vertical_enabled:
            y = height - 1 - y

        if self.rotation_angle == 90:
            x, y = height - 1 - y, x
            width, height = height, width

        elif self.rotation_angle == 180:
            x = width - 1 - x
            y = height - 1 - y

        elif self.rotation_angle == 270:
            x, y = y, width - 1 - x
            width, height = height, width

        return x, y

    def _inverse_transform_point(self, x, y, width, height):
        """
        Convert a DISPLAY coordinate back into the original RAW
        camera coordinate.
        """

        if self.rotation_angle == 90:
            x, y = y, height - 1 - x
            width, height = height, width

        elif self.rotation_angle == 180:
            x = width - 1 - x
            y = height - 1 - y

        elif self.rotation_angle == 270:
            x, y = width - 1 - y, x
            width, height = height, width

        if self.flip_vertical_enabled:
            y = height - 1 - y

        if self.flip_horizontal_enabled:
            x = width - 1 - x

        return x, y

    def _transform_box(self, x1, y1, x2, y2, width, height):
        corners = [
            self._transform_point(x1, y1, width, height),
            self._transform_point(x2, y1, width, height),
            self._transform_point(x1, y2, width, height),
            self._transform_point(x2, y2, width, height),
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _compute_axis_y(self, cy, h):
        """Row the horizontal crosshair line should be drawn on, in the
        RAW camera's native resolution. Only "External Actuation" mode
        lets the X-Axis slider move it; every other mode keeps it pinned
        to the true center row (cy), same as before this feature existed."""
        if self.settings.get("actuation_mode") != "external":
            return cy

        slider = self.settings.get("x_axis_slider")
        if slider is None:
            slider = 0.5

        # t: -1 (full left) .. +1 (full right). Right moves the line UP
        # toward row 0; left moves it DOWN toward the bottom edge. The
        # true center (cx, cy) itself never moves - only this row does.
        t = (slider - 0.5) * 2
        if t >= 0:
            axis_y = cy - t * cy
        else:
            axis_y = cy + (-t) * (h - 1 - cy)
        return int(round(axis_y))

    def _build_render_plan(self, image, depth_frame, cx, cy):
        """Runs detection (if a model is loaded) and the nozzle-targeting
        math/Arduino send EXACTLY ONCE, and returns a plain-data description
        of what needs to be drawn. All coordinates here are in the RAW
        camera's native resolution - _apply_render_plan scales them for
        whichever image it's drawing onto."""
        plan = {
            "center": (cx, cy),
            "axis_y": self._compute_axis_y(cy, image.shape[0]),
            "boxes": [],  # (x1, y1, x2, y2, label, conf) - every detection
            "centroids": [],  # (cx, cy) - every detection's centroid
            "primary_target": None,
            "target_style": None,  # "ok" | "error" | "unavailable" | None
            "text_lines": [],
        }

        if self.model is not None and self.detection_worker is not None:
            # Non-blocking: hand off this frame to the background worker and
            # draw whatever it most recently finished (may lag a frame or
            # two behind on a slow model, but the video feed itself never
            # freezes waiting on inference).
            self.detection_worker.submit_frame(image)
            detections = self.detection_worker.get_latest_detections()
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["box"]
                plan["boxes"].append((x1, y1, x2, y2, det["label"], det["conf"]))
                ocx, ocy = (x1 + x2) // 2, (y1 + y2) // 2
                plan["centroids"].append((ocx, ocy))
                if i == 0:
                    plan["primary_target"] = (ocx, ocy)  # first detection drives line/text/nozzle
        else:
            if self.mouse_clicked:
                mx, my = self.mouse_x, self.mouse_y
            else:
                mx, my = cx, cy
            plan["primary_target"] = (mx, my)

        if plan["primary_target"] is None:
            return plan  # e.g. model loaded but nothing detected this frame

        tx, ty = plan["primary_target"]

        if self.camera_source.has_depth and depth_frame is not None:
            if PRIVATE_MODULE_AVAILABLE:
                arduino_conn = self.arduino.connection if (self.arduino and self.arduino.is_connected) else None
                result = compute_target(self.camera_source, depth_frame, tx, ty, self.nozzle, arduino_conn)
                if result is None:
                    plan["target_style"] = "error"
                else:
                    plan["target_style"] = "ok"
                    plan["text_lines"] = [
                        (f"Diag Dist: {result['diag']:.3f} m", (0, 255, 0)),
                        (f"Target Ang: {result['angle']:.1f} deg ({result['direction']})", (0, 255, 0)),
                        (f"Steps to Move: {result['steps']}", (0, 255, 0)),
                        (f"Nozzle At: {result['nozzle_angle']:.1f} deg", (0, 255, 0)),
                    ]
            else:
                plan["target_style"] = "unavailable"
        else:
            dist = pixel_distance(cx, cy, tx, ty)
            plan["target_style"] = "ok"
            plan["text_lines"] = [(f"Pixel Dist: {dist:.1f} px", (0, 255, 0))]

        return plan

    def _draw_target_lines(self, image, plan, s_cx, s_axis_y, s_tx, s_ty, scale, color=None):
        """Draws one diagonal line (in the existing style) per detected
        box, pointing at that box's own centroid - not just at the first
        detection's. Falls back to a single line to the click/primary
        target when there are no boxes at all (click-mode)."""
        if plan["boxes"]:
            for box, centroid in zip(plan["boxes"], plan["centroids"]):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                tx1, ty1, tx2, ty2 = self._transform_box(x1, y1, x2, y2, self.raw_width, self.raw_height)
                sbox = (int(tx1 * scale), int(ty1 * scale), int(tx2 * scale), int(ty2 * scale))
                ocx, ocy = centroid
                tocx, tocy = self._transform_point(ocx, ocy, self.raw_width, self.raw_height)
                s_ocx = int(tocx * scale)
                s_ocy = int(tocy * scale)
                overlay.draw_click_marker(image, s_cx, s_axis_y, s_ocx, s_ocy, box=sbox, color=color, scale=scale)
        else:
            overlay.draw_click_marker(image, s_cx, s_axis_y, s_tx, s_ty, color=color, scale=scale)

    def _apply_render_plan(self, image, plan, scale=1.0, mode="full"):
        """Pure drawing - safe to call more than once per frame with the
        SAME plan (no side effects), at any resolution/scale.

        `mode` controls which layers get drawn:
          - "full": everything, gated by the Diagonal Distance setting -
            this is what's shown on screen and recorded to video. The FPS
            viewer (if on) always draws here regardless of that gating.
          - "boxes_only": detection boxes/labels (or the click dot, in
            click-mode) and nothing else - no crosshair, no diagonal
            line, no text, no FPS viewer - for the "Screenshot
            (Boxes/Clicks Only)" button.
        """
        cx, cy = plan["center"]
        img_w = self.raw_width
        img_h = self.raw_height
        tx, ty = self._transform_point(cx, cy, img_w, img_h)
        s_cx = int(tx * scale)
        s_cy = int(ty * scale)

        axis_y = plan.get("axis_y", cy)

        axis_tx, axis_ty = self._transform_point(cx, axis_y, img_w, img_h)

        # At 0/180 rotation the raw horizontal axis line stays horizontal on
        # screen, so the slider-driven row lives in the transformed Y
        # component (axis_ty) and the vertical crosshair line stays fixed
        # at the true center column (s_cx). At 90/270 rotation the axes
        # swap: that same raw line is now drawn as a VERTICAL line, so the
        # slider-driven column lives in the transformed X component
        # (axis_tx) instead, and the fixed line becomes horizontal at the
        # true center row (s_cy). Using the wrong component is exactly why
        # the slider previously had no visible effect at 90/270.
        if self.rotation_angle in (90, 270):
            s_axis_col = int(axis_tx * scale)
            s_axis_row = s_cy
        else:
            s_axis_col = s_cx
            s_axis_row = int(axis_ty * scale)

        if mode == "boxes_only":
            if plan["boxes"]:
                for x1, y1, x2, y2, label, conf in plan["boxes"]:
                    tx1, ty1, tx2, ty2 = self._transform_box(x1, y1, x2, y2, img_w, img_h)
                    sbox = (int(tx1 * scale), int(ty1 * scale), int(tx2 * scale), int(ty2 * scale))
                    overlay.draw_detection_box(image, sbox, label, conf, scale=scale)
                for ocx, ocy in plan["centroids"]:
                    tx, ty = self._transform_point(ocx, ocy, img_w, img_h)
                    overlay.draw_centroid_marker(image, int(tx * scale), int(ty * scale), scale=scale)
            elif plan["primary_target"] is not None:
                tx, ty = plan["primary_target"]
                ttx, tty = self._transform_point(tx, ty, img_w, img_h)
                overlay.draw_centroid_marker(image, int(ttx * scale), int(tty * scale), scale=scale)
            return

        overlay.draw_axes(image, s_axis_col, s_axis_row, thickness=max(1, round(scale)))
        if s_axis_col != s_cx or s_axis_row != s_cy:
            # X-Axis slider has shifted the crosshair - show where the
            # true, unmoved center still is, in a distinct color
            overlay.draw_fixed_center_marker(image, s_cx, s_cy, scale=scale)

        # FPS viewer - independent of every other config option/toggle above
        # (Diagonal Distance, actuation mode, target_style, etc.) and of the
        # "boxes_only" screenshot variant, which already returned above.
        if self.settings.get("fps_viewer_on"):
            overlay.draw_fps(image, self.fps, scale=scale)

        if plan["target_style"] is None:
            return

        diagonal_on = bool(self.settings.get("diagonal_distance_on"))

        tx, ty = plan["primary_target"]
        ttx, tty = self._transform_point(tx, ty, img_w, img_h)
        s_tx = int(ttx * scale)
        s_ty = int(tty * scale)

        if plan["target_style"] == "error":
            if diagonal_on:
                self._draw_target_lines(image, plan, s_axis_col, s_axis_row, s_tx, s_ty, scale, color=(0, 0, 255))
                cv2.putText(image, "No Depth Data", (int(20 * scale), int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 255), max(1, round(2 * scale)), cv2.LINE_AA)
        elif plan["target_style"] == "unavailable":
            if diagonal_on:
                self._draw_target_lines(image, plan, s_axis_col, s_axis_row, s_tx, s_ty, scale, color=(0, 0, 255))
                cv2.putText(image, "Depth targeting module not available", (int(20 * scale), int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 255), max(1, round(2 * scale)), cv2.LINE_AA)
        else:  # "ok"
            if diagonal_on:
                self._draw_target_lines(image, plan, s_axis_col, s_axis_row, s_tx, s_ty, scale)
            elif not plan["boxes"]:
                # no line, but still show the click point itself (a
                # detection's own centroid dot is drawn below regardless)
                overlay.draw_centroid_marker(image, s_tx, s_ty, scale=scale)

            for x1, y1, x2, y2, label, conf in plan["boxes"]:
                tx1, ty1, tx2, ty2 = self._transform_box(x1, y1, x2, y2, img_w, img_h)
                sbox = (int(tx1 * scale), int(ty1 * scale), int(tx2 * scale), int(ty2 * scale))
                overlay.draw_detection_box(image, sbox, label, conf, scale=scale)

            for ocx, ocy in plan["centroids"]:
                tx, ty = self._transform_point(ocx, ocy, img_w, img_h)
                overlay.draw_centroid_marker(image, int(tx * scale), int(ty * scale), scale=scale)

            if diagonal_on:
                overlay.draw_text_lines(image, plan["text_lines"], scale=scale)

    def flip_vertical(self):
        self.flip_vertical_enabled = not self.flip_vertical_enabled
        state = "ON" if self.flip_vertical_enabled else "OFF"
        self._show_notice(f"Vertical Flip : {state}", "ok")

    def flip_horizontal(self):
        self.flip_horizontal_enabled = not self.flip_horizontal_enabled
        state = "ON" if self.flip_horizontal_enabled else "OFF"
        self._show_notice(f"Horizontal Flip : {state}", "ok")

    def rotate_cw(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self._show_notice(f"Rotation : {self.rotation_angle}°", "ok")

    def rotate_ccw(self):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self._show_notice(f"Rotation : {self.rotation_angle}°", "ok")

    # ------------------------------------------------------------ close
    def on_close(self):
        if self.config_window is not None and self.config_window.winfo_exists():
            self.config_window.destroy()
        self.recorder.stop_recording()
        if self.camera_source:
            self.camera_source.stop()
        if self.arduino:
            self.arduino.disconnect()
        if self.detection_worker:
            self.detection_worker.stop()

    def show_rgb_channel(self):
        self.view_mode = "rgb"

    def show_red_channel(self):
        self.view_mode = "red"

    def show_green_channel(self):
        self.view_mode = "green"

    def show_blue_channel(self):
        self.view_mode = "blue"

    def show_grayscale(self):
        self.view_mode = "gray"

    def show_depth_channel(self):
        self.view_mode = "depth"

    def show_thermal_channel(self):
        self.view_mode = "thermal"

    # --------------------------------------------------------- window utils
    def _open_singleton_window(self, attr, builder):
        """Reuses an already-open Toplevel instead of stacking duplicates -
        repeated clicks on the same sidebar button just refocus whatever
        that button already opened, instead of piling up copies of it."""
        win = getattr(self, attr, None)
        if win is not None and win.winfo_exists():
            win.lift()
            win.focus_force()
            return win
        win = builder()
        setattr(self, attr, win)
        return win

    def _labeled_slider(self, parent, text, minimum, maximum, current, on_change, steps=None, pady=(10, 2)):
        """One slider row with its live value shown at the far right of its
        title - so the current number is always visible, not just implied
        by the handle's position."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=pady)
        row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(row, text=text, anchor="w").grid(row=0, column=0, sticky="w")
        value_label = ctk.CTkLabel(row, text=str(int(current)), width=44, anchor="e", font=ctk.CTkFont(weight="bold"), text_color="#4CC9F0")
        value_label.grid(row=0, column=1, sticky="e")

        def _apply(value):
            on_change(value)
            value_label.configure(text=str(int(round(float(value)))))

        slider = ctk.CTkSlider(parent, from_=minimum, to=maximum, number_of_steps=steps or max(1, int(maximum - minimum)), command=_apply)
        slider.set(current)
        slider.pack(fill="x", padx=20, pady=(0, 4))
        return slider

    # ---------------------------------------------------------- thresholding
    def open_threshold_settings(self):
        """"Thresholding" now offers a pick of the 3 techniques used
        worldwide in image processing/computer vision - simple binary,
        Otsu's automatic method, and local adaptive thresholding - before
        opening that specific method's own configuration window."""
        def build():
            picker = ctk.CTkToplevel(self)
            picker.title("Select Thresholding Method")
            picker.geometry("340x260")
            picker.resizable(False, False)
            ctk.CTkLabel(picker, text="Choose a Thresholding Technique", font=ctk.CTkFont(size=15, weight="bold")).pack(padx=16, pady=(20, 12))
            methods = [("Binary Thresholding", self._open_binary_threshold_window), ("Otsu's Method", self._open_otsu_window), ("Adaptive Thresholding", self._open_adaptive_window)]
            for label, opener in methods:
                ctk.CTkButton(picker, text=label, command=lambda o=opener, p=picker: (p.destroy(), o())).pack(fill="x", padx=20, pady=6)
            return picker
        self._open_singleton_window("threshold_picker_window", build)

    def _open_binary_threshold_window(self):
        self.view_mode, self.threshold_method = "threshold", "binary"

        def build():
            win = ctk.CTkToplevel(self)
            win.title("Thresholding - Binary")
            win.geometry("380x260")
            win.resizable(False, False)
            self._labeled_slider(win, "Threshold Value", 0, 255, self.binary_thresh, lambda v: setattr(self, "binary_thresh", int(v)), pady=(20, 2))
            self._labeled_slider(win, "Maximum Value", 0, 255, self.binary_maxval, lambda v: setattr(self, "binary_maxval", int(v)))
            return win
        self._open_singleton_window("binary_window", build)

    def _open_otsu_window(self):
        self.view_mode, self.threshold_method = "threshold", "otsu"

        def build():
            win = ctk.CTkToplevel(self)
            win.title("Thresholding - Otsu's Method")
            win.geometry("380x240")
            win.resizable(False, False)
            ctk.CTkLabel(win, text="Otsu's method computes the split point automatically from the image histogram. Only the output level is configurable here.",
                         font=ctk.CTkFont(size=12), text_color="gray60", justify="left", wraplength=340).pack(padx=20, pady=(18, 4))
            self._labeled_slider(win, "Maximum Value", 0, 255, self.otsu_maxval, lambda v: setattr(self, "otsu_maxval", int(v)))
            invert_var = tk.BooleanVar(value=self.otsu_invert)
            ctk.CTkSwitch(win, text="Invert Intensities", variable=invert_var, onvalue=True, offvalue=False,
                          command=lambda: setattr(self, "otsu_invert", bool(invert_var.get()))).pack(anchor="w", padx=20, pady=(8, 4))
            return win
        self._open_singleton_window("otsu_window", build)

    def _open_adaptive_window(self):
        self.view_mode, self.threshold_method = "threshold", "adaptive"

        def build():
            win = ctk.CTkToplevel(self)
            win.title("Thresholding - Adaptive")
            win.geometry("400x400")
            win.resizable(False, False)
            ctk.CTkLabel(win, text="Local Method", anchor="w").pack(fill="x", padx=20, pady=(18, 2))
            method_var = tk.StringVar(value=self.adaptive_method)
            ctk.CTkSegmentedButton(win, values=["Mean", "Gaussian"], variable=method_var,
                                    command=lambda v: setattr(self, "adaptive_method", v)).pack(fill="x", padx=20, pady=(0, 8))
            self._labeled_slider(win, "Maximum Value", 0, 255, self.adaptive_maxval, lambda v: setattr(self, "adaptive_maxval", int(v)))
            self._labeled_slider(win, "Block Size (Odd, Neighborhood Width)", 3, 51, self.adaptive_block_size, self._adaptive_block_changed, steps=24)
            self._labeled_slider(win, "C (Constant Subtracted from Mean)", -50, 50, self.adaptive_c, lambda v: setattr(self, "adaptive_c", int(v)))
            return win
        self._open_singleton_window("adaptive_window", build)

    def _adaptive_block_changed(self, value):
        block = int(round(float(value)))
        self.adaptive_block_size = block if block % 2 == 1 else block + 1

    # ------------------------------------------------------------ HSV / HSL
    def open_hsv_settings(self):
        self.view_mode = "hsv"

        def build():
            win = ctk.CTkToplevel(self)
            win.title("HSV Filter")
            win.geometry("420x560")
            win.resizable(False, False)
            sliders = [
                ("Hue Minimum", 0, 179, self.hsv_h_min, lambda v: setattr(self, "hsv_h_min", int(v))),
                ("Hue Maximum", 0, 179, self.hsv_h_max, lambda v: setattr(self, "hsv_h_max", int(v))),
                ("Saturation Minimum", 0, 255, self.hsv_s_min, lambda v: setattr(self, "hsv_s_min", int(v))),
                ("Saturation Maximum", 0, 255, self.hsv_s_max, lambda v: setattr(self, "hsv_s_max", int(v))),
                ("Value Minimum", 0, 255, self.hsv_v_min, lambda v: setattr(self, "hsv_v_min", int(v))),
                ("Value Maximum", 0, 255, self.hsv_v_max, lambda v: setattr(self, "hsv_v_max", int(v))),
            ]
            for text, minimum, maximum, current, callback in sliders:
                self._labeled_slider(win, text, minimum, maximum, current, callback)
            return win
        self._open_singleton_window("hsv_window", build)

    def open_hsl_settings(self):
        self.view_mode = "hsl"

        def build():
            win = ctk.CTkToplevel(self)
            win.title("HSL Filter")
            win.geometry("420x560")
            win.resizable(False, False)
            sliders = [
                ("Hue Minimum", 0, 179, self.hsl_h_min, lambda v: setattr(self, "hsl_h_min", int(v))),
                ("Hue Maximum", 0, 179, self.hsl_h_max, lambda v: setattr(self, "hsl_h_max", int(v))),
                ("Lightness Minimum", 0, 255, self.hsl_l_min, lambda v: setattr(self, "hsl_l_min", int(v))),
                ("Lightness Maximum", 0, 255, self.hsl_l_max, lambda v: setattr(self, "hsl_l_max", int(v))),
                ("Saturation Minimum", 0, 255, self.hsl_s_min, lambda v: setattr(self, "hsl_s_min", int(v))),
                ("Saturation Maximum", 0, 255, self.hsl_s_max, lambda v: setattr(self, "hsl_s_max", int(v))),
            ]
            for text, minimum, maximum, current, callback in sliders:
                self._labeled_slider(win, text, minimum, maximum, current, callback)
            return win
        self._open_singleton_window("hsl_window", build)