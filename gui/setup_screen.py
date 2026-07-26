"""
First screen shown on launch: Local vs Remote. Same logic as before
(Remote is a dead end for now) - only the look changed.
"""

import customtkinter as ctk
from tkinter import messagebox


class SetupScreen(ctk.CTkFrame):
    def __init__(self, master, on_local_selected):
        super().__init__(master, fg_color="transparent")
        self.on_local_selected = on_local_selected

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            wrapper, text="Spraying Dashboard",
            font=ctk.CTkFont(size=26, weight="bold")
        ).pack(pady=(0, 4))

        ctk.CTkLabel(
            wrapper, text="Select an input source to begin",
            font=ctk.CTkFont(size=14), text_color="gray60"
        ).pack(pady=(0, 24))

        self._make_card(
            wrapper,
            title="Local Setup",
            subtitle="Use this computer's webcam or a connected RealSense camera",
            command=self._choose_local,
            accent="#2563eb",
        )

        self._make_card(
            wrapper,
            title="Remote Setup",
            subtitle="Stream from a Raspberry Pi in the field  ·  coming soon",
            command=self._choose_remote,
            accent="#6b7280",
        )

    def _make_card(self, parent, title, subtitle, command, accent):
        card = ctk.CTkButton(
            parent, text="", command=command,
            width=380, height=76, corner_radius=12,
            fg_color="#1f2937", hover_color="#27303f",
        )
        card.pack(pady=8)

        # overlay text on top of the button using place, so we get two-line
        # rich text on a single clickable surface
        label = ctk.CTkLabel(
            card, text=f"{title}\n{subtitle}",
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="left", text_color="white",
        )
        label.place(relx=0.06, rely=0.5, anchor="w")
        # forward clicks on the label to the button underneath
        label.bind("<Button-1>", lambda e: command())

        strip = ctk.CTkFrame(card, width=6, height=76, fg_color=accent, corner_radius=0)
        strip.place(relx=0.0, rely=0.5, anchor="w")

    def _choose_local(self):
        self.on_local_selected()

    def _choose_remote(self):
        messagebox.showinfo(
            "Remote Setup",
            "Underdeveloped: remote (Raspberry Pi) mode is not available yet."
        )