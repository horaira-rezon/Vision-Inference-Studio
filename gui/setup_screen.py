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

        wrapper = ctk.CTkFrame(self, fg_color="#191919", border_width=2, border_color="#2B2B2B", corner_radius=8)
        wrapper.place(relx=0.5, rely=0.5, anchor="center")

        # Inner frame for padding
        inner = ctk.CTkFrame(wrapper, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=37, pady=37)

        ctk.CTkLabel(
            inner, text="Camera Dashboard",
            font=ctk.CTkFont(size=50, weight="bold")
        ).pack(pady=(0, 22))

        ctk.CTkLabel(
            inner, text="Select an Input from below ↓",
            font=ctk.CTkFont(size=23, weight="normal"), text_color="gray60"
        ).pack(pady=(0, 26))

        self._make_card(
            inner,
            title="Local Setup",
            subtitle="Use this computer's Webcam or an external Camera",
            command=self._choose_local,
            accent="#2563eb",
        )

        self._make_card(
            inner,
            title="Remote Setup",
            subtitle="Stream from a Raspberry Pi remotely ~ coming soon",
            command=self._choose_remote,
            accent="#cb2d2d",
        )

    def _make_card(self, parent, title, subtitle, command, accent):
        card = ctk.CTkFrame(parent, width=450, height=80, corner_radius=12, fg_color="#151c26")
        card.pack(pady=10)
        # hover effect
        def on_enter(e):
            card.configure(fg_color="#1f2937")
        def on_leave(e):
            card.configure(fg_color="#151c26")
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        card.bind("<Button-1>", lambda e: command())
        # accent strip
        strip = ctk.CTkFrame(card, width=6, height=100, fg_color=accent, corner_radius=0)
        strip.place(relx=0.0, rely=0.5, anchor="w")
        # title label
        title_label = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=20, weight="bold"), text_color="white", anchor="w")
        title_label.place(relx=0.06, rely=0.3, anchor="w")
        title_label.bind("<Button-1>", lambda e: command())
        # subtitle label
        subtitle_label = ctk.CTkLabel(card, text=subtitle, font=ctk.CTkFont(size=17), text_color="white", anchor="w")
        subtitle_label.place(relx=0.06, rely=0.7, anchor="w")
        subtitle_label.bind("<Button-1>", lambda e: command())

    def _choose_local(self):
        self.on_local_selected()

    def _choose_remote(self):
        messagebox.showinfo(
            "Remote Setup",
            "Underdeveloped: remote mode is not available yet."
        )