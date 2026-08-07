"""
Configuration window: External Actuation mode, the Diagonal Distance
toggle, the X-Axis line mover, Object Tracking, and the Confidence
Threshold slider. Themed to match setup_screen.py's card look. Reads/
writes straight through the shared Settings instance - MainApp's render
loop reads those same settings fresh every frame, so nothing here needs
to push updates back to it directly.

Save/Reset: every toggle/slider/checkbox still applies immediately (so you
get live preview in the video feed while adjusting things) and is written
to disk right away via Settings.set(). "Save" is an explicit confirmation
of that (useful since there's no other feedback that it stuck); "Reset"
restores all five numbered sections to their default values.

Sections 4 (Object Tracking) and 5 (Confidence Threshold) are deliberately
NOT gated by sections 1-3 (actuation mode / diagonal distance / axis) -
they're independent controls that apply regardless of what those are set
to, unlike the old Multi-Box Decision section they replaced.
"""

import customtkinter as ctk
import tkinter as tk

BG = "#191919"
BORDER = "#2B2B2B"
CARD_BG = "#151c26"
CARD_BG_HOVER = "#1f2937"
CARD_BG_SELECTED = "#1e3a5f"
ACCENT = "#2563eb"
DESC_COLOR = "gray60"
NOTE_COLOR = "#f59e0b"

MODE_OPTIONS = [
    (
        "none",
        "No External Actuation",
        "Only mouse clicks and detection boxes are shown, with their centroids. Nothing below is needed, so the diagonal line, its distance readout, and the other options all stay off.",
    ),
    (
        "external",
        "External Actuation",
        "Unlocks every option below - the Diagonal Distance toggle and the X-Axis line mover both become available.",
    ),
    (
        "diagonal_only",
        "Diagonal Distance Only",
        "No external hardware, but you still need the diagonal line and distance readout. Diagonal Distance switches on automatically; the X-Axis line mover stays locked.",
    ),
]

TRACKER_OPTIONS = [
    ("none", "No Tracking"),
    ("bytetrack", "ByteTrack"),
    ("botsort", "BotSORT"),
]

DEFAULTS = {
    "actuation_mode": "none",
    "diagonal_distance_on": False,
    "x_axis_slider": 0.5,
    "tracker_mode": "none",
    "confidence_threshold": 50,
}


class ConfigWindow(ctk.CTkToplevel):
    def __init__(self, master, settings, has_model_fn=None):
        super().__init__(master)
        self.settings = settings
        self.has_model_fn = has_model_fn or (lambda: False)  # kept for API compatibility; unused now

        self.title("Configuration")
        self.geometry("620x760")
        self.minsize(560, 560)
        self.configure(fg_color=BG)

        self._mode_cards = {}
        self._tracker_buttons = {}
        self._wrap_labels = []  # (label, side_padding) - kept in sync with window width
        self._poll_after_id = None
        self._last_gating_state = None

        self._build_ui()
        self._refresh_from_settings()

        self.bind("<Escape>", lambda e: self._close())
        self.bind("<Configure>", self._update_wraplengths)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._poll_gating()

        self.after(50, self._focus_and_grab)
        self.after(60, self._update_wraplengths)  # sane wrap widths before first paint settles

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

    # ------------------------------------------------------------- layout
    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 4))

        ctk.CTkLabel(
            header, text="Configuration", font=ctk.CTkFont(size=26, weight="bold")
        ).pack(side="left")
        # No custom close ("X") button here on purpose - the OS-provided
        # title bar close button already closes this window.

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=(8, 16))
        self.scroll = scroll

        # --- 1. External Actuation ---
        self._section_title(scroll, "1. External Actuation")
        for mode_key, title, desc in MODE_OPTIONS:
            self._mode_cards[mode_key] = self._make_mode_card(scroll, mode_key, title, desc)

        # --- 2. Diagonal Distance ---
        self._section_title(scroll, "2. Diagonal Distance")
        diag_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        diag_card.pack(fill="x", pady=(2, 4))
        diag_inner = ctk.CTkFrame(diag_card, fg_color="transparent")
        diag_inner.pack(fill="x", padx=16, pady=12)

        self.diag_switch_var = tk.BooleanVar(value=False)
        self.diag_switch = ctk.CTkSwitch(
            diag_inner, text="Show diagonal line + distance readout",
            variable=self.diag_switch_var, onvalue=True, offvalue=False,
            command=self._on_diag_toggle, progress_color=ACCENT,
        )
        self.diag_switch.pack(anchor="w")
        diag_desc = ctk.CTkLabel(
            diag_inner,
            text="Draws the line from the axis lines intersection to the mouse click or bounding box centroid. OFF means no line and no overlay text at all.",
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w",
        )
        diag_desc.pack(anchor="w", pady=(6, 0), fill="x")
        self._register_wrap_label(diag_desc)

        # --- 3. X-Axis line mover ---
        self._section_title(scroll, "3. X-Axis Line Position")
        axis_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        axis_card.pack(fill="x", pady=(2, 4))
        axis_inner = ctk.CTkFrame(axis_card, fg_color="transparent")
        axis_inner.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(axis_inner, text="Left (Down) - Right (Up)", font=ctk.CTkFont(size=13), text_color="gray70").pack(anchor="w")
        self.axis_slider_var = tk.DoubleVar(value=0.5)
        self.axis_slider = ctk.CTkSlider(
            axis_inner, from_=0.0, to=1.0, variable=self.axis_slider_var,
            command=self._on_axis_slide, progress_color=ACCENT,
        )
        self.axis_slider.pack(fill="x", pady=(6, 0))
        self._prevent_slider_wheel_hijack(self.axis_slider)
        self._flush_on_release(self.axis_slider)

        axis_desc = ctk.CTkLabel(
            axis_inner,
            text="Moves the horizontal crosshair line up or down without moving the true camera center. The diagonal line always follows the moving intersection point, not the fixed center.",
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w",
        )
        axis_desc.pack(anchor="w", pady=(6, 0), fill="x")
        self._register_wrap_label(axis_desc)

        # --- 4. Object Tracking (independent of sections 1-3 above) ---
        self._section_title(scroll, "4. Object Tracking")
        tracker_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        tracker_card.pack(fill="x", pady=(2, 4))
        tracker_inner = ctk.CTkFrame(tracker_card, fg_color="transparent")
        tracker_inner.pack(fill="x", padx=16, pady=12)

        tracker_row = ctk.CTkFrame(tracker_inner, fg_color="transparent")
        tracker_row.pack(fill="x")

        for i, (tracker_key, label) in enumerate(TRACKER_OPTIONS):
            padx = (0, 4) if i == 0 else ((4, 4) if i < len(TRACKER_OPTIONS) - 1 else (4, 0))
            btn = ctk.CTkButton(tracker_row, text=label, command=lambda k=tracker_key: self._select_tracker(k))
            btn.pack(side="left", expand=True, fill="x", padx=padx)
            self._tracker_buttons[tracker_key] = btn

        # --- 5. Confidence Threshold (also independent of sections 1-3) ---
        self._section_title(scroll, "5. Confidence Threshold")
        conf_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        conf_card.pack(fill="x", pady=(2, 4))
        conf_inner = ctk.CTkFrame(conf_card, fg_color="transparent")
        conf_inner.pack(fill="x", padx=16, pady=12)

        conf_header = ctk.CTkFrame(conf_inner, fg_color="transparent")
        conf_header.pack(fill="x")
        ctk.CTkLabel(conf_header, text="Minimum confidence", font=ctk.CTkFont(size=13), text_color="gray70").pack(side="left")
        self.confidence_value_label = ctk.CTkLabel(conf_header, text="50%", font=ctk.CTkFont(size=13, weight="bold"))
        self.confidence_value_label.pack(side="right")

        self.confidence_slider_var = tk.IntVar(value=50)
        self.confidence_slider = ctk.CTkSlider(
            conf_inner, from_=0, to=100, number_of_steps=100,
            variable=self.confidence_slider_var, command=self._on_confidence_slide, progress_color=ACCENT,
        )
        self.confidence_slider.pack(fill="x", pady=(6, 0))
        self._prevent_slider_wheel_hijack(self.confidence_slider)
        self._flush_on_release(self.confidence_slider)

        conf_desc = ctk.CTkLabel(
            conf_inner,
            text="Only boxes at or above this confidence score are shown in the streaming window and sent for actuation. Any tracking algorithm selected above only ever sees boxes that already pass this threshold.",
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w",
        )
        conf_desc.pack(anchor="w", pady=(8, 0), fill="x")
        self._register_wrap_label(conf_desc)

        # --- FPS Viewer (independent - not gated by anything above, and
        # not touched by Reset, which only restores the 5 numbered sections) ---
        self._section_title(scroll, "FPS Viewer")
        fps_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        fps_card.pack(fill="x", pady=(2, 4))
        fps_inner = ctk.CTkFrame(fps_card, fg_color="transparent")
        fps_inner.pack(fill="x", padx=16, pady=12)

        self.fps_switch_var = tk.BooleanVar(value=bool(self.settings.get("fps_viewer_on")))
        self.fps_switch = ctk.CTkSwitch(
            fps_inner, text="Show FPS viewer", variable=self.fps_switch_var,
            onvalue=True, offvalue=False, command=self._on_fps_toggle, progress_color=ACCENT,
        )
        self.fps_switch.pack(anchor="w")
        fps_desc = ctk.CTkLabel(
            fps_inner,
            text="ON by default, independent of every option above. Shown in the live view and the full-overlay screenshot, but not the clean or boxes/clicks-only screenshot variants.",
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w",
        )
        fps_desc.pack(anchor="w", pady=(6, 0), fill="x")
        self._register_wrap_label(fps_desc)

        # --- Save / Reset ---
        footer = ctk.CTkFrame(scroll, fg_color="transparent")
        footer.pack(fill="x", pady=(20, 4))

        self.save_btn = ctk.CTkButton(
            footer, text="Save", command=self._save_configuration,
            fg_color=ACCENT, hover_color="#1d4ed8",
        )
        self.save_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.reset_btn = ctk.CTkButton(
            footer, text="Reset", command=self._reset_configuration,
            fg_color="transparent", border_width=1, border_color="gray40",
        )
        self.reset_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        self.save_status_label = ctk.CTkLabel(
            scroll, text="", font=ctk.CTkFont(size=12), text_color="#4ade80"
        )
        self.save_status_label.pack(anchor="w", pady=(6, 0))

    def _make_mode_card(self, parent, mode_key, title, desc):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=CARD_BG, border_width=1, border_color=BORDER)
        card.pack(fill="x", pady=4)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=12)

        title_label = ctk.CTkLabel(inner, text=title, font=ctk.CTkFont(size=16, weight="bold"), anchor="w")
        title_label.pack(anchor="w")
        desc_label = ctk.CTkLabel(
            inner, text=desc, font=ctk.CTkFont(size=13), text_color=DESC_COLOR,
            justify="left", anchor="w",
        )
        desc_label.pack(anchor="w", pady=(4, 0), fill="x")
        self._register_wrap_label(desc_label)

        def choose(_e=None, k=mode_key):
            self._select_mode(k)

        for w in (card, inner, title_label, desc_label):
            w.bind("<Button-1>", choose)

        def on_enter(_e, c=card, k=mode_key):
            if self.settings.get("actuation_mode") != k:
                c.configure(fg_color=CARD_BG_HOVER)

        def on_leave(_e, c=card, k=mode_key):
            if self.settings.get("actuation_mode") != k:
                c.configure(fg_color=CARD_BG)

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

        return card

    def _section_title(self, parent, text):
        ctk.CTkLabel(
            parent, text=text, font=ctk.CTkFont(size=16, weight="bold"), text_color="gray85"
        ).pack(anchor="w", pady=(14, 6))

    # ------------------------------------------------------- wheel safety
    def _prevent_slider_wheel_hijack(self, slider):
        """Scrolling the mouse wheel while hovering a slider should scroll
        the PAGE, not change the slider's value. Binds on both the slider's
        outer widget and its internal canvas (whichever actually receives
        the wheel event), forwards the scroll to the page's own scrollable
        canvas, and blocks further handling so the slider itself never
        sees the wheel event."""

        def handler(event):
            delta = 0
            if getattr(event, "delta", 0):
                delta = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            try:
                self.scroll._parent_canvas.yview_scroll(delta, "units")
            except Exception:
                pass
            return "break"

        targets = [slider]
        inner_canvas = getattr(slider, "_canvas", None)
        if inner_canvas is not None:
            targets.append(inner_canvas)

        for widget in targets:
            widget.bind("<MouseWheel>", handler)  # Windows / macOS
            widget.bind("<Button-4>", handler)     # Linux scroll up
            widget.bind("<Button-5>", handler)     # Linux scroll down

    def _flush_on_release(self, slider):
        """The slider's `command` updates settings in-memory only (see
        _on_axis_slide/_on_confidence_slide) so dragging doesn't hit the
        disk on every tick. This writes the final value once the mouse
        button is released, so the drag's end result still gets persisted.
        Uses add="+" so CTkSlider's own release handling (which stops the
        drag) keeps working - this only adds a second callback, it doesn't
        replace it."""

        def flush(_event=None):
            self.settings.save()

        targets = [slider]
        inner_canvas = getattr(slider, "_canvas", None)
        if inner_canvas is not None:
            targets.append(inner_canvas)

        for widget in targets:
            widget.bind("<ButtonRelease-1>", flush, add="+")

    # -------------------------------------------------- dynamic wrapping
    def _register_wrap_label(self, label, side_padding=100):
        self._wrap_labels.append((label, side_padding))
        self._update_wraplengths()

    def _update_wraplengths(self, event=None):
        width = self.winfo_width()
        if width <= 1:
            width = 620  # window not realized yet - fall back to initial geometry
        for label, side_padding in self._wrap_labels:
            label.configure(wraplength=max(200, width - side_padding))

    # ------------------------------------------------------------ actions
    def _select_mode(self, mode_key):
        self.settings.set("actuation_mode", mode_key)

        if mode_key == "none":
            self.settings.set("diagonal_distance_on", False)
        elif mode_key == "diagonal_only":
            self.settings.set("diagonal_distance_on", True)
        # "external" leaves whatever diagonal_distance_on already was

        self._refresh_from_settings()

    def _on_diag_toggle(self):
        self.settings.set("diagonal_distance_on", bool(self.diag_switch_var.get()))
        self._apply_gating()

    def _on_fps_toggle(self):
        # Fully independent of mode/gating - just writes straight through
        self.settings.set("fps_viewer_on", bool(self.fps_switch_var.get()))

    def _on_axis_slide(self, value):
        self.settings.set("x_axis_slider", float(value), persist=False)

    def _select_tracker(self, tracker_key):
        # Independent of sections 1-3 - always available, never gated
        self.settings.set("tracker_mode", tracker_key)
        self._refresh_tracker_buttons()

    def _refresh_tracker_buttons(self):
        current = self.settings.get("tracker_mode") or "none"
        for key, btn in self._tracker_buttons.items():
            if key == current:
                btn.configure(fg_color=ACCENT, hover_color="#1d4ed8", border_width=0)
            else:
                btn.configure(fg_color="transparent", hover_color=CARD_BG_HOVER, border_width=1, border_color="gray40")

    def _on_confidence_slide(self, value):
        pct = int(round(float(value)))
        self.settings.set("confidence_threshold", pct, persist=False)
        self.confidence_value_label.configure(text=f"{pct}%")

    def _save_configuration(self):
        self.settings.save()  # everything already writes through live; this confirms it
        self._show_save_feedback("Configuration saved")

    def _reset_configuration(self):
        for key, value in DEFAULTS.items():
            self.settings.set(key, value)
        self._refresh_from_settings()
        self._show_save_feedback("Reset to defaults")

    def _show_save_feedback(self, text):
        self.save_status_label.configure(text=text)
        self.after(1800, lambda: self.save_status_label.configure(text=""))

    # ------------------------------------------------------------ gating
    def _poll_gating(self):
        self._apply_gating()
        self._poll_after_id = self.after(500, self._poll_gating)

    def _apply_gating(self):
        mode = self.settings.get("actuation_mode")
        diagonal_on = bool(self.settings.get("diagonal_distance_on"))

        state = (mode, diagonal_on)
        if state == self._last_gating_state:
            return
        self._last_gating_state = state

        # highlight the selected mode card
        for key, card in self._mode_cards.items():
            card.configure(fg_color=CARD_BG_SELECTED if key == mode else CARD_BG)

        # section 2: diagonal toggle
        if mode == "none":
            self.diag_switch_var.set(False)
            self.diag_switch.configure(state="disabled")
        elif mode == "diagonal_only":
            self.diag_switch_var.set(True)
            self.diag_switch.configure(state="disabled")
        elif mode == "external":
            self.diag_switch_var.set(diagonal_on)
            self.diag_switch.configure(state="normal")
        else:
            self.diag_switch_var.set(False)
            self.diag_switch.configure(state="disabled")

        # section 3: X-axis slider - only usable under "external"
        axis_enabled = mode == "external"
        self.axis_slider.configure(state="normal" if axis_enabled else "disabled")

        # sections 4 & 5 (Object Tracking, Confidence Threshold) are
        # intentionally never gated here - always enabled regardless of
        # mode/diagonal/model state.

    # --------------------------------------------------------------- sync
    def _refresh_from_settings(self):
        mode = self.settings.get("actuation_mode")
        diagonal_on = bool(self.settings.get("diagonal_distance_on"))
        axis_val = self.settings.get("x_axis_slider")
        confidence = self.settings.get("confidence_threshold")
        if confidence is None:
            confidence = 50

        self.diag_switch_var.set(diagonal_on)
        self.axis_slider_var.set(axis_val if axis_val is not None else 0.5)
        self.confidence_slider_var.set(confidence)
        self.confidence_value_label.configure(text=f"{confidence}%")

        for key, card in self._mode_cards.items():
            card.configure(fg_color=CARD_BG_SELECTED if key == mode else CARD_BG)

        self._refresh_tracker_buttons()
        self._last_gating_state = None  # force _apply_gating to re-sync widget states below
        self._apply_gating()