import customtkinter as ctk


class ViewControls(ctk.CTkScrollableFrame):
    def __init__(self, parent, app):
        super().__init__(parent, corner_radius=0, fg_color="#1a1d23")

        self.app = app

        self.button_font = ctk.CTkFont(size=14)
        self.section_font = ctk.CTkFont(size=17, weight="bold")
        self.title_font = ctk.CTkFont(size=20, weight="bold")

        ctk.CTkLabel(
            self,
            text="View Controls",
            font=self.title_font,
        ).pack(fill="x", padx=16, pady=(16, 12))

        self._build_orientation()
        self._build_rotation()
        self._build_rgb_section()
        self._build_hsv_hsl_section()
        self._build_grayscale_section()
        self._build_binary_section()
        self._build_depth_section()

    def _reset_orientation(self):
        self.app.rotation_angle = 0
        self.app.flip_vertical_enabled = False
        self.app.flip_horizontal_enabled = False

    def _section_label(self, text):
        ctk.CTkLabel(
            self,
            text=text,
            font=self.section_font,
            text_color="gray70"
        ).pack(anchor="w", padx=16, pady=(16, 6))

    def _two_button_row(self, left_text, left_cmd, right_text, right_cmd):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(0, 4))

        frame.grid_columnconfigure((0, 1), weight=1)

        left = ctk.CTkButton(
            frame,
            text=left_text,
            command=left_cmd,
            font=self.button_font
        )
        left.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        right = ctk.CTkButton(
            frame,
            text=right_text,
            command=right_cmd,
            font=self.button_font
        )
        right.grid(row=0, column=1, padx=(4, 0), sticky="ew")

    def _build_orientation(self):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(16, 6))

        label = ctk.CTkLabel(frame, text="Orientation", font=self.section_font, text_color="gray70")
        label.pack(side="left")

        reset_btn = ctk.CTkButton(
            frame,
            text="⟳",
            width=28,
            height=28,
            command=self._reset_orientation,
            fg_color="transparent",
            hover_color="#1a1d23",
            text_color="white",
            border_width=0,
            font=ctk.CTkFont(size=17)
        )
        reset_btn.pack(side="left", padx=(2, 0))
        reset_btn.bind("<Enter>", lambda e: reset_btn.configure(text_color="#4CC9F0"))
        reset_btn.bind("<Leave>", lambda e: reset_btn.configure(text_color="white"))

        self._two_button_row(
            "Flip Vertical",
            self.app.flip_vertical,
            "Flip Horizontal",
            self.app.flip_horizontal
        )

    def _build_rotation(self):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(16, 6))

        label = ctk.CTkLabel(frame, text="Rotation", font=self.section_font, text_color="gray70")
        label.pack(side="left")

        reset_btn = ctk.CTkButton(
            frame,
            text="⟳",
            width=28,
            height=28,
            command=self._reset_orientation,
            fg_color="transparent",
            hover_color="#1a1d23",
            text_color="white",
            border_width=0,
            font=ctk.CTkFont(size=17)
        )
        reset_btn.pack(side="left", padx=(2, 0))
        reset_btn.bind("<Enter>", lambda e: reset_btn.configure(text_color="#4CC9F0"))
        reset_btn.bind("<Leave>", lambda e: reset_btn.configure(text_color="white"))

        self._two_button_row(
            "Rotate CCW",
            self.app.rotate_ccw,
            "Rotate CW",
            self.app.rotate_cw
        )

    def _build_rgb_section(self):
        self._section_label("RGB Channel")

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=(0, 4))

        frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(
            frame,
            text="RGB",
            command=self.app.show_rgb_channel,
            font=self.button_font,
            fg_color="#5e6471",
            hover_color="#4b5563"
        ).grid(row=0, column=0, padx=(0, 3), sticky="ew")

        ctk.CTkButton(
            frame,
            text="Red",
            command=self.app.show_red_channel,
            font=self.button_font,
            fg_color="#7f1d1d",
            hover_color="#991b1b"
        ).grid(row=0, column=1, padx=3, sticky="ew")

        ctk.CTkButton(
            frame,
            text="Green",
            command=self.app.show_green_channel,
            font=self.button_font,
            fg_color="#065f46",
            hover_color="#047857"
        ).grid(row=0, column=2, padx=3, sticky="ew")

        ctk.CTkButton(
            frame,
            text="Blue",
            command=self.app.show_blue_channel,
            font=self.button_font,
            fg_color="#2a6093",
            hover_color="#3d77ad"
        ).grid(row=0, column=3, padx=(3, 0), sticky="ew")

    def _build_depth_section(self):
        self._section_label("Depth Channel")

        ctk.CTkButton(
            self,
            text="Depth View",
            command=self.app.show_depth_channel,
            font=self.button_font,
            fg_color="#374151",
            hover_color="#4b5563",
        ).pack(fill="x", padx=16, pady=(0, 4))

    def _build_grayscale_section(self):
        self._section_label("Grayscale")

        ctk.CTkButton(
            self,
            text="Show Grayscale",
            command=self.app.show_grayscale,
            font=self.button_font,
            fg_color="#374151",
            hover_color="#4b5563",
        ).pack(fill="x", padx=16, pady=(0, 4))

    def _build_binary_section(self):
        self._section_label("Thresholding")

        ctk.CTkButton(
            self,
            text="Threshold Settings",
            command=self.app.open_threshold_settings,
            font=self.button_font,
            fg_color="#374151",
            hover_color="#4b5563",
        ).pack(fill="x", padx=16, pady=(0, 4))

    def _build_hsv_hsl_section(self):
        self._section_label("HSV / HSL Channel")

        self._two_button_row(
            "HSV Settings",
            self.app.open_hsv_settings,
            "HSL Settings",
            self.app.open_hsl_settings
        )