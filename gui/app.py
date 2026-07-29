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
        self._raw_frame = None      # clean frame (no overlay), refreshed every loop
        self._last_plan = None      # render plan for the same frame, for on-demand overlay variants
        self.config_window = None

        self.mouse_x, self.mouse_y = 0, 0
        self.mouse_clicked = False  # True after first click in click mode
        self.display_scale = 1.0  # updated every frame by _scale_to_panel
        self.recording_start_time = None
        self._recording_after_id = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_video_area()

    # ------------------------------------------------------------ sidebar
    def _build_sidebar(self):
        sidebar_container = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#1a1d23")
        sidebar_container.grid(row=0, column=0, sticky="nsw")
        sidebar_container.grid_propagate(False)
        sidebar_container.grid_rowconfigure(0, weight=1)
        sidebar_container.grid_columnconfigure(0, weight=1)

        sidebar = ctk.CTkScrollableFrame(sidebar_container, corner_radius=0, fg_color="#1a1d23")
        sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(sidebar, text="Setup", font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=16, pady=(16, 12)
        )

        # --- Camera ---
        self.camera_dot = self._section_label_with_dot(sidebar, "1. Camera")
        self.camera_btn = ctk.CTkButton(sidebar, text="Select Camera", command=self.select_camera, font=ctk.CTkFont(size=14))
        self.camera_btn.pack(fill="x", padx=16, pady=(4, 2))

        # --- Model ---
        self.model_dot = self._section_label_with_dot(sidebar, "2. Detection Model")
        self.model_btn = ctk.CTkButton(sidebar, text="Select Model Weight (.pt)", command=self.select_model, font=ctk.CTkFont(size=14))
        self.model_btn.pack(fill="x", padx=16, pady=(4, 2))
        self.skip_model_btn = ctk.CTkButton(
            sidebar, text="Skip Model (Click Mode)", command=self.skip_model,
            fg_color="#5e6471", hover_color="#4b5563", font=ctk.CTkFont(size=14)
        )
        self.skip_model_btn.pack(fill="x", padx=16, pady=(0, 2))

        # --- Arduino ---
        self.arduino_dot = self._section_label_with_dot(sidebar, "3. Arduino")
        self.arduino_btn = ctk.CTkButton(
            sidebar, text="Connect Arduino", command=self.toggle_arduino,
            fg_color="#065f46", hover_color="#047857", font=ctk.CTkFont(size=14)
        )
        self.arduino_btn.pack(fill="x", padx=16, pady=(4, 2))

        # --- Recording ---
        self._section_label(sidebar, "4. Recording")

        self.video_folder_btn = ctk.CTkButton(
            sidebar, text="Select Video Folder", command=self.select_video_folder,
            fg_color="transparent", border_width=1, border_color="gray40", font=ctk.CTkFont(size=14)
        )
        self.video_folder_btn.pack(fill="x", padx=16, pady=(4, 2))
        self.video_folder_label = ctk.CTkLabel(
            sidebar, text=self._short_path(self.settings.get("video_output_dir")) or "No folder selected",
            font=ctk.CTkFont(size=14), text_color="gray60", anchor="w", justify="left",
        )
        self.video_folder_label.pack(anchor="w", padx=16, pady=(0, 8), fill="x")

        self.record_btn = ctk.CTkButton(
            sidebar, text="Start Recording", command=self.toggle_recording,
            fg_color="#7f1d1d", hover_color="#991b1b", font=ctk.CTkFont(size=14)
        )
        self.record_btn.pack(fill="x", padx=16, pady=(0, 14))

        # --- Screenshot ---
        self._section_label(sidebar, "5. Screenshot")

        self.screenshot_folder_btn = ctk.CTkButton(
            sidebar, text="Select Image Folder", command=self.select_screenshot_folder,
            fg_color="transparent", border_width=1, border_color="gray40", font=ctk.CTkFont(size=14)
        )
        self.screenshot_folder_btn.pack(fill="x", padx=16, pady=(4, 2))
        self.screenshot_folder_label = ctk.CTkLabel(
            sidebar, text=self._short_path(self.settings.get("screenshot_output_dir")) or "No folder selected",
            font=ctk.CTkFont(size=14), text_color="gray60", anchor="w", justify="left",
        )
        self.screenshot_folder_label.pack(anchor="w", padx=16, pady=(0, 8), fill="x")

        self.screenshot_btn = ctk.CTkButton(
            sidebar, text="Window Screenshot", command=self.take_screenshot,
            fg_color="#065f46", hover_color="#047857", font=ctk.CTkFont(size=14)
        )
        self.screenshot_btn.pack(fill="x", padx=16, pady=(0, 2))

        self.screenshot_clean_btn = ctk.CTkButton(
            sidebar, text="Capture Cam. Frame", command=self.take_screenshot_clean,
            fg_color="#05523c", hover_color="#036247", font=ctk.CTkFont(size=14)
        )
        self.screenshot_clean_btn.pack(fill="x", padx=16, pady=(0, 2))

        self.screenshot_boxes_btn = ctk.CTkButton(
            sidebar, text="Detection UI Window", command=self.take_screenshot_boxes_only,
            fg_color="#033f2e", hover_color="#02553E", font=ctk.CTkFont(size=14)
        )
        self.screenshot_boxes_btn.pack(fill="x", padx=16, pady=(0, 14))

        # --- Configuration: pinned below the scrollable list (own footer
        # area, doesn't scroll away with everything else) ---
        footer = ctk.CTkFrame(sidebar_container, fg_color="#15181d", corner_radius=0)
        footer.grid(row=1, column=0, sticky="ew")
        self.config_btn = ctk.CTkButton(
            footer, text="Configuration", command=self.open_configuration,
            fg_color="#374151", hover_color="#4b5563", font=ctk.CTkFont(size=16)
        )
        self.config_btn.pack(fill="x", padx=16, pady=30)

    def _section_label(self, parent, text):
        ctk.CTkLabel(
            parent, text=text, font=ctk.CTkFont(size=17, weight="bold"), text_color="gray70"
        ).pack(anchor="w", padx=16, pady=(16, 2))

    def _section_label_with_dot(self, parent, text):
        """Create a section label with a status dot right after it."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(anchor="w", padx=16, pady=(16, 2))

        label = ctk.CTkLabel(
            container, text=text, font=ctk.CTkFont(size=17, weight="bold"), text_color="gray70"
        )
        label.pack(side="left")

        # Use a CTkFrame for the dot to have precise control over dimensions
        dot = ctk.CTkFrame(container, width=12, height=12, corner_radius=6,
                          fg_color=DOT_COLORS["idle"])
        dot.pack(side="left", padx=(7, 0))
        dot.pack_propagate(False)  # Prevent the frame from shrinking to fit contents
        return dot

    def _set_dot(self, dot, color_key):
        dot.configure(fg_color=DOT_COLORS[color_key])

    def _short_path(self, path, max_len=30):
        if not path:
            return None
        return path if len(path) <= max_len else "..." + path[-(max_len - 3):]

    # --------------------------------------------------------- video area
    def _build_video_area(self):
        # shared layout constants - used here AND in _scale_to_panel, so the
        # video's placement and its available scaling space never drift out
        # of sync with each other
        self.SIDE_MARGIN = 50
        self.TOP_MARGIN = 40
        self.BOTTOM_GAP = 130     # reserved band at the bottom for the notice
        self.SIZE_SHRINK = 0.82   # extra shrink beyond best-fit -> noticeably smaller by default

        outer = ctk.CTkFrame(self, corner_radius=0, fg_color="#0b0d10")
        outer.grid(row=0, column=1, sticky="nsew")
        self.video_frame = outer  # _scale_to_panel reads this size each frame

        self.video_label = tk.Label(outer, bg="#0b0d10", fg="white",
                                     text="Select a camera to begin",
                                     font=("Arial", 14))
        # Centered within the region ABOVE the reserved bottom gap (not the
        # whole window) - shifting up by half the (gap - top margin) keeps
        # it visually centered in its own area rather than the full panel.
        self.video_label.place(relx=0.5, rely=0.5, anchor="center",
                                y=(self.TOP_MARGIN - self.BOTTOM_GAP) // 2)
        self.video_label.bind("<Button-1>", self._on_click)

        # floating notice - vertically centered in the gap between the
        # video's bottom edge and the window's bottom edge (not glued to
        # either one)
        self.notice_bar = ctk.CTkLabel(
            outer, text="", font=ctk.CTkFont(size=20),
            fg_color="transparent", text_color=NOTICE_COLORS["idle"],
        )
        self.notice_bar.place(relx=0.5, rely=1.0, anchor="center",
                               y=-(self.BOTTOM_GAP // 2))

    def _show_notice(self, text, color_key="ok"):
        self.notice_bar.configure(text=text, text_color=NOTICE_COLORS.get(color_key, "gray70"))
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
        ctk.CTkLabel(picker, text="Multiple cameras found",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(padx=12, pady=(16, 8))
        for opt in options:
            ctk.CTkButton(
                picker, text=opt,
                command=lambda o=opt: (self._start_camera(o), picker.destroy())
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
        path = choose_file("Select Model Weight", pattern="*.pt", pattern_label="PyTorch weights")
        if not path:
            return
        self._show_notice("Loading model...", "warn")
        threading.Thread(target=self._load_model_async, args=(path,), daemon=True).start()

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
                self.arduino_btn.configure(text="Disconnect Arduino", fg_color="#7f1d1d", hover_color="#991b1b")
                self._set_dot(self.arduino_dot, "ok")
                self._show_notice("Arduino connected", "ok")
            except Exception as e:
                self._set_dot(self.arduino_dot, "error")
                self._show_notice(str(e), "error")
        else:
            self.arduino.disconnect()
            self.arduino_btn.configure(text="Connect Arduino", fg_color="#065f46", hover_color="#047857")
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
        self.notice_bar.configure(text=f"Recording: {time_str}", text_color=NOTICE_COLORS["warn"])
        self.notice_bar.lift()
        # schedule next update
        self._recording_after_id = self.after(1000, self._update_recording_notice)

    def take_screenshot(self):
        """Full overlay - exactly what's on screen / recorded to video."""
        self._save_screenshot(self.current_frame)

    def take_screenshot_clean(self):
        """No UI elements at all - the raw camera frame."""
        self._save_screenshot(self._raw_frame)

    def take_screenshot_boxes_only(self):
        """Detection boxes + labels (or the click dot, in click-mode) -
        no crosshair, no diagonal line, no text overlay."""
        if self._raw_frame is None or self._last_plan is None:
            return
        frame = self._raw_frame.copy()
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
        # event.x/y are in displayed (scaled) pixels - convert back to the
        # original camera resolution so depth/detection lookups stay correct
        self.mouse_x = int(event.x / self.display_scale)
        self.mouse_y = int(event.y / self.display_scale)
        self.mouse_clicked = True  # mark that user has clicked

    # ------------------------------------------------------- frame loop
    def update_frame(self):
        if self.camera_source is None:
            return

        raw_image, depth_frame, cx, cy = self.camera_source.read()
        if raw_image is not None:
            # Compute everything ONCE - detection inference, nozzle math,
            # and any Arduino send all happen exactly one time per frame,
            # regardless of how many times the result gets drawn below.
            plan = self._build_render_plan(raw_image, depth_frame, cx, cy)
            self._raw_frame = raw_image.copy()   # clean, no overlay - for screenshot variants
            self._last_plan = plan

            # Native-resolution pass: what gets recorded/screenshotted -
            # unchanged from before, same resolution as the raw camera feed.
            record_image = raw_image.copy()
            self._apply_render_plan(record_image, plan, scale=1.0)
            self.current_frame = record_image
            self.recorder.write_frame(record_image)

            # Display pass: resize the CLEAN raw frame first, then draw the
            # same plan directly at that resolution (scaled coordinates,
            # thicker lines, bigger text) instead of upscaling an already-
            # rasterized overlay - this is what keeps text/lines crisp.
            display_image, scale = self._scale_raw_to_panel(raw_image)
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
            "boxes": [],        # (x1, y1, x2, y2, label, conf) - every detection
            "centroids": [],    # (cx, cy) - every detection's centroid
            "primary_target": None,
            "target_style": None,   # "ok" | "error" | "unavailable" | None
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
                sbox = (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
                ocx, ocy = centroid
                s_ocx, s_ocy = int(ocx * scale), int(ocy * scale)
                overlay.draw_click_marker(image, s_cx, s_axis_y, s_ocx, s_ocy, box=sbox, color=color, scale=scale)
        else:
            overlay.draw_click_marker(image, s_cx, s_axis_y, s_tx, s_ty, color=color, scale=scale)

    def _apply_render_plan(self, image, plan, scale=1.0, mode="full"):
        """Pure drawing - safe to call more than once per frame with the
        SAME plan (no side effects), at any resolution/scale.

        `mode` controls which layers get drawn:
          - "full": everything, gated by the Diagonal Distance setting -
            this is what's shown on screen and recorded to video.
          - "boxes_only": detection boxes/labels (or the click dot, in
            click-mode) and nothing else - no crosshair, no diagonal
            line, no text - for the "Screenshot (Boxes/Clicks Only)" button.
        """
        cx, cy = plan["center"]
        s_cx, s_cy = int(cx * scale), int(cy * scale)
        axis_y = plan.get("axis_y", cy)
        s_axis_y = int(axis_y * scale)

        if mode == "boxes_only":
            if plan["boxes"]:
                for (x1, y1, x2, y2, label, conf) in plan["boxes"]:
                    sbox = (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
                    overlay.draw_detection_box(image, sbox, label, conf, scale=scale)
                for (ocx, ocy) in plan["centroids"]:
                    overlay.draw_centroid_marker(image, int(ocx * scale), int(ocy * scale), scale=scale)
            elif plan["primary_target"] is not None:
                tx, ty = plan["primary_target"]
                overlay.draw_centroid_marker(image, int(tx * scale), int(ty * scale), scale=scale)
            return

        overlay.draw_axes(image, s_cx, s_axis_y, thickness=max(1, round(scale)))
        if s_axis_y != s_cy:
            # X-Axis slider has shifted the crosshair - show where the
            # true, unmoved center still is, in a distinct color
            overlay.draw_fixed_center_marker(image, s_cx, s_cy, scale=scale)

        if plan["target_style"] is None:
            return

        diagonal_on = bool(self.settings.get("diagonal_distance_on"))

        tx, ty = plan["primary_target"]
        s_tx, s_ty = int(tx * scale), int(ty * scale)

        if plan["target_style"] == "error":
            if diagonal_on:
                self._draw_target_lines(image, plan, s_cx, s_axis_y, s_tx, s_ty, scale, color=(0, 0, 255))
                cv2.putText(image, "No Depth Data", (int(20 * scale), int(40 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 255), max(1, round(2 * scale)), cv2.LINE_AA)
        elif plan["target_style"] == "unavailable":
            if diagonal_on:
                self._draw_target_lines(image, plan, s_cx, s_axis_y, s_tx, s_ty, scale, color=(0, 0, 255))
                cv2.putText(image, "Depth targeting module not available", (int(20 * scale), int(40 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 255), max(1, round(2 * scale)), cv2.LINE_AA)
        else:  # "ok"
            if diagonal_on:
                self._draw_target_lines(image, plan, s_cx, s_axis_y, s_tx, s_ty, scale)
            elif not plan["boxes"]:
                # no line, but still show the click point itself (a
                # detection's own centroid dot is drawn below regardless)
                overlay.draw_centroid_marker(image, s_tx, s_ty, scale=scale)

            for (x1, y1, x2, y2, label, conf) in plan["boxes"]:
                sbox = (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
                overlay.draw_detection_box(image, sbox, label, conf, scale=scale)

            for (ocx, ocy) in plan["centroids"]:
                overlay.draw_centroid_marker(image, int(ocx * scale), int(ocy * scale), scale=scale)

            if diagonal_on:
                overlay.draw_text_lines(image, plan["text_lines"], scale=scale)

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