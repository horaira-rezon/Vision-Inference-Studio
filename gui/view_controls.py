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
        self._section_label("Orientation")

        self._two_button_row(
            "Flip Vertical",
            self.app.flip_vertical,
            "Flip Horizontal",
            self.app.flip_horizontal
        )

    def _build_rotation(self):
        self._section_label("Rotation")

        self._two_button_row(
            "Rotate CCW",
            self.app.rotate_ccw,
            "Rotate CW",
            self.app.rotate_cw
        )