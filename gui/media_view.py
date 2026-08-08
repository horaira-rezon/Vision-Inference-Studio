import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

class MediaView:
    def __init__(self, parent, app):
        self.app = app
        self.outer = ctk.CTkFrame(parent, corner_radius=0, fg_color="#0b0d10")
        self.outer.grid(row=0, column=1, sticky="nsew")
        self.label = tk.Label(self.outer, bg="#0b0d10", fg="white", text="Select a camera to begin", font=("Arial", 14))
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        self.label.bind("<Button-1>", app._on_click)
        self.controls = ctk.CTkFrame(self.outer, fg_color="transparent")
        self.controls.place(relx=0.5, rely=1.0, anchor="s", relwidth=0.82, y=-12)
        self.play_button = ctk.CTkButton(self.controls, text="Pause", width=80, command=app.toggle_media_pause)
        self.play_button.pack(side="left", padx=(0,8))
        self.slider = ctk.CTkSlider(self.controls, from_=0, to=1, command=app.seek_media)
        self.slider.pack(side="left", fill="x", expand=True)
        self.controls.place_forget()
        self.side_margin = 50
        self.top_margin = 40
        self.bottom_gap = 130
        self.size_shrink = 0.82

    def show_file_controls(self, visible=True):
        if visible:
            self.controls.place(relx=0.5, rely=1.0, anchor="s", relwidth=0.82, y=-12)
        else:
            self.controls.place_forget()

    def set_slider(self, value, maximum):
        self.slider.configure(from_=0, to=max(1, maximum-1))
        self.slider.set(value)

    def set_paused(self, paused):
        self.play_button.configure(text="Resume" if paused else "Pause")

    def set_image(self, image):
        rgb = __import__("cv2").cvtColor(image, __import__("cv2").COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk, text="")
