import customtkinter as ctk

DOT_COLORS = {
    "idle": "#6b7280",
    "ok": "#22c55e",
    "warn": "#f59e0b",
    "error": "#ef4444",
}

class LeftSidebar:
    def __init__(self, parent, app):
        self.app = app
        self.container = ctk.CTkFrame(parent, width=260, corner_radius=0, fg_color="#1a1d23")
        self.container.grid(row=0, column=0, sticky="nsw")
        self.container.grid_propagate(False)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.sidebar = ctk.CTkScrollableFrame(self.container, corner_radius=0, fg_color="#1a1d23")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="I/O Commands", font=ctk.CTkFont(size=20, weight="bold")).pack(fill="x", padx=16, pady=(16, 12))
        self.camera_dot = self._section_label_with_dot("1. Camera")
        self.camera_btn = ctk.CTkButton(self.sidebar, text="Select Camera", command=app.select_camera, font=ctk.CTkFont(size=14))
        self.camera_btn.pack(fill="x", padx=16, pady=(4,2))
        self.model_dot = self._section_label_with_dot("2. Vision Tasks")
        self.model_btn = ctk.CTkButton(self.sidebar, text="Select a Vision Task", command=app.select_vision_task, font=ctk.CTkFont(size=14))
        self.model_btn.pack(fill="x", padx=16, pady=(4,2))
        self._section_label("3. Recording")
        self.video_folder_btn = ctk.CTkButton(self.sidebar, text="Select Video Folder", command=app.select_video_folder, fg_color="transparent", border_width=1, border_color="gray40", font=ctk.CTkFont(size=14))
        self.video_folder_btn.pack(fill="x", padx=16, pady=(4,2))
        self.video_folder_label = ctk.CTkLabel(self.sidebar, text=app._short_path(app.settings.get("video_output_dir")) or "No folder selected", font=ctk.CTkFont(size=14), text_color="gray60", anchor="w", justify="left")
        self.video_folder_label.pack(anchor="w", padx=16, pady=(0,8), fill="x")
        self.record_btn = ctk.CTkButton(self.sidebar, text="Start Recording", command=app.toggle_recording, fg_color="#7f1d1d", hover_color="#991b1b", font=ctk.CTkFont(size=14))
        self.record_btn.pack(fill="x", padx=16, pady=(0,14))
        self._section_label("4. Screenshot")
        self.screenshot_folder_btn = ctk.CTkButton(self.sidebar, text="Select Image Folder", command=app.select_screenshot_folder, fg_color="transparent", border_width=1, border_color="gray40", font=ctk.CTkFont(size=14))
        self.screenshot_folder_btn.pack(fill="x", padx=16, pady=(4,2))
        self.screenshot_folder_label = ctk.CTkLabel(self.sidebar, text=app._short_path(app.settings.get("screenshot_output_dir")) or "No folder selected", font=ctk.CTkFont(size=14), text_color="gray60", anchor="w", justify="left")
        self.screenshot_folder_label.pack(anchor="w", padx=16, pady=(0,8), fill="x")
        self.screenshot_btn = ctk.CTkButton(self.sidebar, text="Window Screenshot", command=app.take_screenshot, fg_color="#065f46", hover_color="#047857", font=ctk.CTkFont(size=14))
        self.screenshot_btn.pack(fill="x", padx=16, pady=(0,2))
        self.screenshot_clean_btn = ctk.CTkButton(self.sidebar, text="Capture Cam. Frame", command=app.take_screenshot_clean, fg_color="#05523c", hover_color="#036247", font=ctk.CTkFont(size=14))
        self.screenshot_clean_btn.pack(fill="x", padx=16, pady=(0,2))
        self.screenshot_boxes_btn = ctk.CTkButton(self.sidebar, text="Detection UI Only", command=app.take_screenshot_boxes_only, fg_color="#033f2e", hover_color="#02553E", font=ctk.CTkFont(size=14))
        self.screenshot_boxes_btn.pack(fill="x", padx=16, pady=(0,14))
        footer = ctk.CTkFrame(self.container, fg_color="#15181d", corner_radius=0)
        footer.grid(row=1, column=0, sticky="ew")
        self.config_btn = ctk.CTkButton(footer, text="Configuration", command=app.open_configuration, fg_color="#374151", hover_color="#4b5563", font=ctk.CTkFont(size=16))
        self.config_btn.pack(fill="x", padx=16, pady=30)

    def _section_label(self, text):
        ctk.CTkLabel(self.sidebar, text=text, font=ctk.CTkFont(size=17, weight="bold"), text_color="gray70").pack(anchor="w", padx=16, pady=(16,2))

    def _section_label_with_dot(self, text):
        container=ctk.CTkFrame(self.sidebar, fg_color="transparent")
        container.pack(anchor="w", padx=16, pady=(16,2))
        ctk.CTkLabel(container,text=text,font=ctk.CTkFont(size=17,weight="bold"),text_color="gray70").pack(side="left")
        dot=ctk.CTkFrame(container,width=12,height=12,corner_radius=6,fg_color=DOT_COLORS["idle"])
        dot.pack(side="left",padx=(7,0))
        dot.pack_propagate(False)
        return dot

    def set_dot(self, dot, color_key):
        dot.configure(fg_color=DOT_COLORS[color_key])
