"""
Entry point. Run this file: python3 main.py
"""

import customtkinter as ctk

from assets.config.settings import Settings
from gui.setup_screen import SetupScreen
from gui.app import MainApp

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RootController:
    """Owns the window and swaps between the setup screen and the main app."""

    def __init__(self, root):
        self.root = root
        self.root.title("Vision Inference Studio")
        self.root.geometry("1600x900")
        self.root.minsize(900, 550)

        self.settings = Settings()
        self.main_app = None
 
        self.setup_screen = SetupScreen(root, on_local_selected=self.launch_local_app)
        self.setup_screen.pack(fill="both", expand=True)

    def launch_local_app(self):
        self.setup_screen.destroy()
        self.main_app = MainApp(self.root, self.settings)
        self.main_app.pack(fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        if self.main_app:
            self.main_app.on_close()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    RootController(root)
    root.mainloop()