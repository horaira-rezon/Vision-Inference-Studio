import customtkinter as ctk
from models.task_registry import VISION_TASKS, architecture_choices

class VisionTaskWindow(ctk.CTkToplevel):
    def __init__(self, master, on_task_selected):
        super().__init__(master)
        self.title("Select a Vision Task")
        self.geometry("420x380")
        self.resizable(False, False)
        self.on_task_selected = on_task_selected
        ctk.CTkLabel(self, text="Select a Vision Task", font=ctk.CTkFont(size=18, weight="bold")).pack(padx=20, pady=(24,16))
        for key, label in VISION_TASKS:
            ctk.CTkButton(self, text=label, command=lambda k=key: self._choose(k)).pack(fill="x", padx=28, pady=6)

    def _choose(self, task):
        self.destroy()
        ArchitectureWindow(self.master, task, self.on_task_selected)

class ArchitectureWindow(ctk.CTkToplevel):
    def __init__(self, master, task, on_selected):
        super().__init__(master)
        self.title("Select Model Architecture")
        self.geometry("420x420")
        self.resizable(False, False)
        self.task = task
        self.on_selected = on_selected
        ctk.CTkLabel(self, text="Select Model Architecture", font=ctk.CTkFont(size=18, weight="bold")).pack(padx=20, pady=(24,16))
        for key, label in architecture_choices(task):
            ctk.CTkButton(self, text=label, command=lambda k=key: self._choose(k)).pack(fill="x", padx=28, pady=6)

    def _choose(self, architecture):
        self.destroy()
        self.on_selected(self.task, architecture)
