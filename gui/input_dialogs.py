import customtkinter as ctk

class ChoiceWindow(ctk.CTkToplevel):
    def __init__(self, master, title, heading, options, command, width=360, height=300):
        super().__init__(master)
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.resizable(False, False)
        ctk.CTkLabel(self, text=heading, font=ctk.CTkFont(size=17, weight="bold")).pack(padx=20, pady=(24, 16))
        for key, label in options:
            ctk.CTkButton(self, text=label, command=lambda k=key: command(k, self)).pack(fill="x", padx=24, pady=5)
        self.after(50, self._focus)

    def _focus(self):
        self.lift()
        self.focus_force()

class CameraInputWindow(ChoiceWindow):
    pass
