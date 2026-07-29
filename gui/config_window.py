"""
Configuration window: External Actuation mode, the Diagonal Distance
toggle, the X-Axis line mover, and the (scaffold-only, for now)
Multiple Box Distance Merge section. Themed to match setup_screen.py's
card look. Reads/writes straight through the shared Settings instance -
MainApp's render loop reads those same settings fresh every frame, so
nothing here needs to push updates back to it directly.

Save/Reset: every toggle/slider/checkbox still applies immediately (so you
get live preview in the video feed while adjusting things) and is written
to disk right away via Settings.set(). "Save" is an explicit confirmation
of that (useful since there's no other feedback that it stuck); "Reset"
restores all four sections to their default values.
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
        "Only mouse clicks and detection boxes are shown, with their "
        "centroids. Nothing below is needed, so the diagonal line, its "
        "distance readout, and the other options all stay off.",
    ),
    (
        "external",
        "External Actuation",
        "Unlocks every option below - the Diagonal Distance toggle, the "
        "X-Axis line mover, and the Multiple Box Distance Merge section "
        "all become available.",
    ),
    (
        "diagonal_only",
        "Diagonal Distance Only",
        "No external hardware, but you still need the diagonal line and "
        "distance readout. Diagonal Distance switches on automatically; "
        "the rest of the options below stay locked.",
    ),
]

DEFAULTS = {
    "actuation_mode": "none",
    "diagonal_distance_on": False,
    "x_axis_slider": 0.5,
    "multi_box_count": 1,
    "axis_quadrants": {},
}


class ConfigWindow(ctk.CTkToplevel):
    def __init__(self, master, settings, has_model_fn=None):
        super().__init__(master)
        self.settings = settings
        self.has_model_fn = has_model_fn or (lambda: False)

        self.title("Configuration")
        self.geometry("620x760")
        self.minsize(560, 560)
        self.configure(fg_color=BG)

        self._mode_cards = {}
        self._axis_vars = {}
        self._axis_checkboxes = []
        self._wrap_labels = []  # (label, side_padding) - kept in sync with window width
        self._poll_after_id = None

        self._build_ui()
        self._refresh_from_settings()

        self.bind("<Escape>", lambda e: self._close())
        self.bind("<Configure>", self._update_wraplengths)
        self.protocol("WM_DELETE_WINDOW", self._close)

        # keep the multi-box section's enabled/disabled state in sync even
        # if a model finishes loading (in a background thread) while this
        # window is left open
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
        # title bar close button already closes this window; having both
        # was a duplicate control.

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
            text="Draws the line (and its distance / depth / nozzle text) from the "
                 "crosshair intersection to the mouse click or box centroid. Off means "
                 "no line and no overlay text at all.",
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

        ctk.CTkLabel(axis_inner, text="Left (down) \u2190\u2192 Right (up)", font=ctk.CTkFont(size=13), text_color="gray70").pack(anchor="w")
        self.axis_slider_var = tk.DoubleVar(value=0.5)
        self.axis_slider = ctk.CTkSlider(
            axis_inner, from_=0.0, to=1.0, variable=self.axis_slider_var,
            command=self._on_axis_slide, progress_color=ACCENT,
        )
        self.axis_slider.pack(fill="x", pady=(6, 0))
        self._prevent_slider_wheel_hijack(self.axis_slider)

        axis_desc = ctk.CTkLabel(
            axis_inner,
            text="Moves the horizontal crosshair line up or down without moving the "
                 "true camera center. The diagonal line always follows the moving "
                 "intersection point, not the fixed center.",
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w",
        )
        axis_desc.pack(anchor="w", pady=(6, 0), fill="x")
        self._register_wrap_label(axis_desc)

        # --- 4. Multiple Box Distance Merge (scaffold) ---
        self._section_title(scroll, "4. Multiple Box Distance Merge")
        self.box_card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        self.box_card.pack(fill="x", pady=(2, 4))
        box_inner = ctk.CTkFrame(self.box_card, fg_color="transparent")
        box_inner.pack(fill="x", padx=16, pady=12)
        self.box_inner = box_inner

        self.box_gate_label = ctk.CTkLabel(
            box_inner, text="", font=ctk.CTkFont(size=13, weight="bold"), text_color=NOTE_COLOR,
        )
        self.box_gate_label.pack(anchor="w", pady=(0, 8))

        count_row = ctk.CTkFrame(box_inner, fg_color="transparent")
        count_row.pack(fill="x")

        self.box_minus_btn = ctk.CTkButton(count_row, text="-", width=32, command=lambda: self._nudge_count(-1))
        self.box_minus_btn.pack(side="left")

        self.box_count_slider_var = tk.IntVar(value=1)
        self.box_count_slider = ctk.CTkSlider(
            count_row, from_=1, to=10, number_of_steps=9,
            variable=self.box_count_slider_var, command=self._on_count_slide, progress_color=ACCENT,
        )
        self.box_count_slider.pack(side="left", fill="x", expand=True, padx=8)
        self._prevent_slider_wheel_hijack(self.box_count_slider)

        self.box_plus_btn = ctk.CTkButton(count_row, text="+", width=32, command=lambda: self._nudge_count(1))
        self.box_plus_btn.pack(side="left")

        self.box_count_label = ctk.CTkLabel(box_inner, text="1 box", font=ctk.CTkFont(size=14, weight="bold"))
        self.box_count_label.pack(anchor="w", pady=(6, 10))

        box_desc = ctk.CTkLabel(
            box_inner,
            text="Sets how many detected boxes are considered together. The axis "
                 "restriction below applies globally to all of them, not per box. "
                 "Leave both sub-checkboxes off to cover the whole half. This section "
                 "is an early scaffold - the actual merge behavior will be built out "
                 "in a later pass.",
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w",
        )
        box_desc.pack(anchor="w", pady=(0, 10), fill="x")
        self._register_wrap_label(box_desc)

        # Single, global axis configuration - applies to however many boxes
        # are set above, rather than one set of checkboxes per box.
        ctk.CTkLabel(
            box_inner, text="Axis Configuration", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", pady=(4, 4))

        self._make_axis_group(box_inner, "+Y axis", "pos_y")
        self._make_axis_group(box_inner, "-Y axis", "neg_y")

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

    def _make_axis_group(self, parent, group_label, group_key):
        """Single global set of Y/X restriction checkboxes for group_key
        ('pos_y' or 'neg_y') - shared across however many boxes are set,
        not duplicated per box. Packed flush-left in `parent`, same
        indentation as the section's description text above it."""
        group = ctk.CTkFrame(parent, fg_color="transparent")
        group.pack(anchor="w", fill="x", pady=(4, 0))

        saved = self.settings.get("axis_quadrants") or {}
        group_var = tk.BooleanVar(value=bool(saved.get(group_key, False)))
        posx_var = tk.BooleanVar(value=bool(saved.get(f"{group_key}_posx", False)))
        negx_var = tk.BooleanVar(value=bool(saved.get(f"{group_key}_negx", False)))
        self._axis_vars[group_key] = (group_var, posx_var, negx_var)

        def on_change(*_a, group_key=group_key, gv=group_var, pv=posx_var, nv=negx_var):
            self._save_axis_quadrant(group_key, gv.get(), pv.get(), nv.get())

        cb = ctk.CTkCheckBox(group, text=group_label, variable=group_var, command=on_change)
        cb.pack(side="left")
        cb_posx = ctk.CTkCheckBox(group, text="+X", variable=posx_var, command=on_change, width=20)
        cb_posx.pack(side="left", padx=(16, 0))
        cb_negx = ctk.CTkCheckBox(group, text="-X", variable=negx_var, command=on_change, width=20)
        cb_negx.pack(side="left", padx=(8, 0))

        self._axis_checkboxes.extend([cb, cb_posx, cb_negx])

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

    def _on_axis_slide(self, value):
        self.settings.set("x_axis_slider", float(value))

    def _on_count_slide(self, value):
        count = int(round(float(value)))
        self.settings.set("multi_box_count", count)
        self.box_count_label.configure(text=f"{count} box" if count == 1 else f"{count} boxes")

    def _nudge_count(self, delta):
        count = int(self.settings.get("multi_box_count") or 1)
        count = max(1, min(10, count + delta))
        self.box_count_slider_var.set(count)
        self._on_count_slide(count)

    def _save_axis_quadrant(self, group_key, group_on, posx_on, negx_on):
        stored = dict(self.settings.get("axis_quadrants") or {})
        stored[group_key] = bool(group_on)
        stored[f"{group_key}_posx"] = bool(posx_on)
        stored[f"{group_key}_negx"] = bool(negx_on)
        self.settings.set("axis_quadrants", stored)

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
        has_model = bool(self.has_model_fn())

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

        # section 4: multi-box merge - needs "external" + diagonal ON + a model
        box_enabled = mode == "external" and diagonal_on and has_model
        if mode != "external":
            gate_text = "Requires External Actuation mode"
        elif not diagonal_on:
            gate_text = "Turn Diagonal Distance ON to use this"
        elif not has_model:
            gate_text = "Input Model Weight"
        else:
            gate_text = ""

        self.box_gate_label.configure(text=gate_text)
        widget_state = "normal" if box_enabled else "disabled"
        self.box_count_slider.configure(state=widget_state)
        self.box_minus_btn.configure(state=widget_state)
        self.box_plus_btn.configure(state=widget_state)
        self.box_card.configure(fg_color=CARD_BG if box_enabled else "#12161c")
        for cb in self._axis_checkboxes:
            cb.configure(state=widget_state)

    # --------------------------------------------------------------- sync
    def _refresh_from_settings(self):
        mode = self.settings.get("actuation_mode")
        diagonal_on = bool(self.settings.get("diagonal_distance_on"))
        axis_val = self.settings.get("x_axis_slider")
        count = int(self.settings.get("multi_box_count") or 1)

        self.diag_switch_var.set(diagonal_on)
        self.axis_slider_var.set(axis_val if axis_val is not None else 0.5)
        self.box_count_slider_var.set(count)
        self.box_count_label.configure(text=f"{count} box" if count == 1 else f"{count} boxes")

        for key, card in self._mode_cards.items():
            card.configure(fg_color=CARD_BG_SELECTED if key == mode else CARD_BG)

        saved = self.settings.get("axis_quadrants") or {}
        for group_key, (gv, pv, nv) in self._axis_vars.items():
            gv.set(bool(saved.get(group_key, False)))
            pv.set(bool(saved.get(f"{group_key}_posx", False)))
            nv.set(bool(saved.get(f"{group_key}_negx", False)))

        self._apply_gating()