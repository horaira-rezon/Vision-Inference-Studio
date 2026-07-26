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
            wrapper, text="Camera Dashboard",
            font=ctk.CTkFont(size=26, weight="bold")
        ).pack(pady=(0, 4))

        ctk.CTkLabel(
            wrapper, text="Select an Input",
            font=ctk.CTkFont(size=20, weight="normal"), text_color="gray60"
        ).pack(pady=(0, 24))

        self._make_card(
            wrapper,
            title="Local Setup",
            subtitle="Use this computer's Webcam or an external Camera",
            command=self._choose_local,
            accent="#2563eb",
        )

        self._make_card(
            wrapper,
            title="Remote Setup",
            subtitle="Stream from a Raspberry Pi remotely ~ coming soon",
            command=self._choose_remote,
            accent="#cb2d2d",
        )

    def _make_card(self, parent, title, subtitle, command, accent):
        card = ctk.CTkButton(
            parent, text=f"{title}\n{subtitle}", command=command,
            width=380, height=76, corner_radius=12,
            fg_color="#1f2937", hover_color="#27303f",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white",
        )
        card.pack(pady=8)

        strip = ctk.CTkFrame(card, width=6, height=76, fg_color=accent, corner_radius=0)
        strip.place(relx=0.0, rely=0.5, anchor="w")

    def _choose_local(self):
        self.on_local_selected()

    def _choose_remote(self):
        messagebox.showinfo(
            "Remote Setup",
            "Underdeveloped: remote (Raspberry Pi) mode is not available yet."
        )