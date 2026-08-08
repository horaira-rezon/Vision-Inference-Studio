import customtkinter as ctk
import tkinter as tk

BG = "#191919"
BORDER = "#2B2B2B"
CARD_BG = "#151c26"
CARD_BG_HOVER = "#1f2937"
CARD_BG_SELECTED = "#1e3a5f"
ACCENT = "#2563eb"
DESC_COLOR = "gray60"

TRACKER_OPTIONS = [
    ("none", "No Tracking"),
    ("bytetrack", "ByteTrack"),
    ("botsort", "BotSORT"),
    ("ocsort", "OC-SORT"),
    ("deepocsort", "DeepOC-SORT"),
]

DEFAULTS = {
    "tracker_mode": "none",
    "confidence_threshold": 50,
    "fps_viewer_on": True,
}

class ConfigWindow(ctk.CTkToplevel):
    def __init__(self, master, settings, get_task_fn=None):
        super().__init__(master)
        self.settings = settings
        self.get_task_fn = get_task_fn or (lambda: None)
        self.title("Configuration")
        self.geometry("620x560")
        self.minsize(560, 500)
        self.configure(fg_color=BG)
        self._tracker_buttons = {}
        self._wrap_labels = []
        self._poll_after_id = None
        self._build_ui()
        self._refresh_from_settings()
        self.bind("<Escape>", lambda e: self._close())
        self.bind("<Configure>", self._update_wraplengths)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._poll_gating()
        self.after(50, self._focus_and_grab)
        self.after(60, self._update_wraplengths)

    def _focus_and_grab(self):
        self.lift()
        self.focus_force()
        try:
            self.grab_set()
        except Exception:
            pass

    def _close(self):
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except Exception:
                pass
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20,4))
        ctk.CTkLabel(header, text="Configuration", font=ctk.CTkFont(size=26, weight="bold")).pack(side="left")
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=(8,16))
        self.scroll = scroll

        self._section_title(scroll, "1. Object Tracking")
        tracker_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        tracker_card.pack(fill="x", pady=(2,4))
        tracker_inner = ctk.CTkFrame(tracker_card, fg_color="transparent")
        tracker_inner.pack(fill="x", padx=16, pady=12)
        tracker_row = ctk.CTkFrame(tracker_inner, fg_color="transparent")
        tracker_row.pack(fill="x")
        for i,(key,label) in enumerate(TRACKER_OPTIONS):
            row,col=divmod(i,3)
            btn=ctk.CTkButton(tracker_row,text=label,command=lambda k=key:self._select_tracker(k))
            btn.grid(row=row,column=col,padx=4,pady=4,sticky="ew")
            self._tracker_buttons[key]=btn
        for col in range(3):
            tracker_row.columnconfigure(col,weight=1)
        desc=ctk.CTkLabel(tracker_inner,text="Select the tracking algorithm used for supported vision tasks. Tracking can be changed independently of the confidence threshold and FPS viewer.",font=ctk.CTkFont(size=13),text_color=DESC_COLOR,justify="left",anchor="w")
        desc.pack(anchor="w",pady=(8,0),fill="x")
        self._register_wrap_label(desc)

        self._section_title(scroll, "2. Confidence Threshold")
        conf_card=ctk.CTkFrame(scroll,fg_color=CARD_BG,corner_radius=10,border_width=1,border_color=BORDER)
        conf_card.pack(fill="x",pady=(2,4))
        conf_inner=ctk.CTkFrame(conf_card,fg_color="transparent")
        conf_inner.pack(fill="x",padx=16,pady=12)
        conf_header=ctk.CTkFrame(conf_inner,fg_color="transparent")
        conf_header.pack(fill="x")
        ctk.CTkLabel(conf_header,text="Minimum confidence",font=ctk.CTkFont(size=13),text_color="gray70").pack(side="left")
        self.confidence_value_label=ctk.CTkLabel(conf_header,text="50%",font=ctk.CTkFont(size=13,weight="bold"))
        self.confidence_value_label.pack(side="right")
        self.confidence_slider_var=tk.IntVar(value=50)
        self.confidence_slider=ctk.CTkSlider(conf_inner,from_=0,to=100,number_of_steps=100,variable=self.confidence_slider_var,command=self._on_confidence_slide,progress_color=ACCENT)
        self.confidence_slider.pack(fill="x",pady=(6,0))
        self._prevent_slider_wheel_hijack(self.confidence_slider)
        self._flush_on_release(self.confidence_slider)
        desc=ctk.CTkLabel(conf_inner,text="Only results at or above this confidence score are shown in the streaming window. The selected tracking algorithm only processes results that pass this threshold.",font=ctk.CTkFont(size=13),text_color=DESC_COLOR,justify="left",anchor="w")
        desc.pack(anchor="w",pady=(8,0),fill="x")
        self._register_wrap_label(desc)

        self._section_title(scroll, "FPS Viewer")
        fps_card=ctk.CTkFrame(scroll,fg_color=CARD_BG,corner_radius=10,border_width=1,border_color=BORDER)
        fps_card.pack(fill="x",pady=(2,4))
        fps_inner=ctk.CTkFrame(fps_card,fg_color="transparent")
        fps_inner.pack(fill="x",padx=16,pady=12)
        self.fps_switch_var=tk.BooleanVar(value=bool(self.settings.get("fps_viewer_on")))
        self.fps_switch=ctk.CTkSwitch(fps_inner,text="Show FPS viewer",variable=self.fps_switch_var,onvalue=True,offvalue=False,command=self._on_fps_toggle,progress_color=ACCENT)
        self.fps_switch.pack(anchor="w")
        desc=ctk.CTkLabel(fps_inner,text="ON by default, independent of every option above. Shown in the live view and the full-overlay screenshot, but not the clean or detection-only screenshot variants.",font=ctk.CTkFont(size=13),text_color=DESC_COLOR,justify="left",anchor="w")
        desc.pack(anchor="w",pady=(6,0),fill="x")
        self._register_wrap_label(desc)

        footer=ctk.CTkFrame(scroll,fg_color="transparent")
        footer.pack(fill="x",pady=(20,4))
        self.save_btn=ctk.CTkButton(footer,text="Save",command=self._save_configuration,fg_color=ACCENT,hover_color="#1d4ed8")
        self.save_btn.pack(side="left",expand=True,fill="x",padx=(0,6))
        self.reset_btn=ctk.CTkButton(footer,text="Reset",command=self._reset_configuration,fg_color="transparent",border_width=1,border_color="gray40")
        self.reset_btn.pack(side="left",expand=True,fill="x",padx=(6,0))
        self.save_status_label=ctk.CTkLabel(scroll,text="",font=ctk.CTkFont(size=12),text_color="#4ade80")
        self.save_status_label.pack(anchor="w",pady=(6,0))

    def _section_title(self,parent,text):
        ctk.CTkLabel(parent,text=text,font=ctk.CTkFont(size=16,weight="bold"),text_color="gray85").pack(anchor="w",pady=(14,6))

    def _prevent_slider_wheel_hijack(self,slider):
        def handler(event):
            if getattr(event, "delta", 0):
                delta = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            else:
                delta = 0
            try:
                self.scroll._parent_canvas.yview_scroll(delta, "units")
            except Exception:
                pass
            return "break"
        for widget in [slider,getattr(slider,"_canvas",None)]:
            if widget is not None:
                widget.bind("<MouseWheel>",handler)
                widget.bind("<Button-4>",handler)
                widget.bind("<Button-5>",handler)

    def _flush_on_release(self,slider):
        def flush(_event=None):
            self.settings.save()
        for widget in [slider,getattr(slider,"_canvas",None)]:
            if widget is not None:
                widget.bind("<ButtonRelease-1>",flush,add="+")

    def _register_wrap_label(self,label,side_padding=100):
        self._wrap_labels.append((label,side_padding))
        self._update_wraplengths()

    def _update_wraplengths(self,event=None):
        width=max(1,self.winfo_width())
        for label,padding in self._wrap_labels:
            label.configure(wraplength=max(200,width-padding))

    def _select_tracker(self,key):
        self.settings.set("tracker_mode",key)
        self._refresh_tracker_buttons()

    def _refresh_tracker_buttons(self):
        current=self.settings.get("tracker_mode") or "none"
        disabled=self.get_task_fn()=="classification"
        for key,btn in self._tracker_buttons.items():
            if disabled:
                btn.configure(state="disabled",fg_color="#374151",hover_color="#374151")
            elif key==current:
                btn.configure(state="normal",fg_color=ACCENT,hover_color="#1d4ed8",border_width=0)
            else:
                btn.configure(state="normal",fg_color="transparent",hover_color=CARD_BG_HOVER,border_width=1,border_color="gray40")

    def _on_confidence_slide(self,value):
        pct=int(round(float(value)))
        self.settings.set("confidence_threshold",pct,persist=False)
        self.confidence_value_label.configure(text=f"{pct}%")

    def _on_fps_toggle(self):
        self.settings.set("fps_viewer_on",bool(self.fps_switch_var.get()))

    def _save_configuration(self):
        self.settings.save()
        self._show_save_feedback("Configuration saved")

    def _reset_configuration(self):
        for key,value in DEFAULTS.items():
            self.settings.set(key,value)
        self._refresh_from_settings()
        self._show_save_feedback("Reset to defaults")

    def _show_save_feedback(self,text):
        self.save_status_label.configure(text=text)
        self.after(1800,lambda:self.save_status_label.configure(text=""))

    def _poll_gating(self):
        self._refresh_tracker_buttons()
        self._poll_after_id=self.after(500,self._poll_gating)

    def _refresh_from_settings(self):
        confidence=self.settings.get("confidence_threshold")
        if confidence is None:
            confidence=50
        self.confidence_slider_var.set(confidence)
        self.confidence_value_label.configure(text=f"{confidence}%")
        self.fps_switch_var.set(bool(self.settings.get("fps_viewer_on")))
        self._refresh_tracker_buttons()
