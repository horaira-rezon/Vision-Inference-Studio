import customtkinter as ctk
import tkinter as tk
from assets.detection.coco_classes import COCO_CLASSES

BG = "#191919"
BORDER = "#2B2B2B"
CARD_BG = "#151c26"
ACCENT = "#2563eb"
DESC_COLOR = "gray60"

DESC_LIVE = "Check whichever classes you want detected. Nothing checked means all 80. You can always re-open this window."
DESC_PENDING = "Select All / Clear All / Check whichever classes you want detected. Nothing checked means all 80 classes."


class CocoClassesWindow(ctk.CTkToplevel):
    def __init__(self, master, settings, on_change, pending=False, on_start=None):
        super().__init__(master)
        self.settings = settings
        self.on_change = on_change
        # pending=True: no model is loaded yet - this window is shown
        # BEFORE detection starts, so you can pick/select-all/clear-all
        # classes first. Clicking "Start Detection" calls on_start() and
        # closes this window immediately - your selection is already
        # saved via on_change as you toggled things, so there's nothing
        # left to show once detection kicks off; it just appears on the
        # video feed.
        self.pending = pending
        self.on_start = on_start
        self.title("COCO Classes")
        self.geometry("420x640")
        self.minsize(360, 420)
        self.configure(fg_color=BG)

        self._vars = {}     # class_id -> BooleanVar
        self._checks = {}   # class_id -> CTkCheckBox (for search filtering)

        self._build_ui()
        self._restore_from_settings()

        self.bind("<Escape>", lambda e: self._close())
        self.bind("<Configure>", self._update_wraplength)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._focus_and_grab)
        self.after(60, self._update_wraplength)  # sane wrap width before first paint settles

    def _focus_and_grab(self):
        self.lift()
        self.focus_force()
        try:
            self.grab_set()
        except Exception:
            pass

    def _close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(20, 4))
        ctk.CTkLabel(header, text="COCO Classes", font=ctk.CTkFont(size=26, weight="bold")).pack(side="left")

        self.desc = ctk.CTkLabel(
            self,
            text=DESC_PENDING if self.pending else DESC_LIVE,
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w",
        )
        self.desc.pack(fill="x", padx=24, pady=(0, 10))

        self.search_var = tk.StringVar(value="")
        search_entry = ctk.CTkEntry(self, placeholder_text="Search classes...", textvariable=self.search_var)
        search_entry.pack(fill="x", padx=24, pady=(0, 8))
        self.search_var.trace_add("write", lambda *_a: self._apply_search())

        button_row = ctk.CTkFrame(self, fg_color="transparent")
        button_row.pack(fill="x", padx=24, pady=(0, 10))
        button_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(button_row, text="Select All", command=self._select_all).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(button_row, text="Clear All", fg_color="#374151", hover_color="#4b5563", command=self._clear_all).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=(0, 8))
        self.scroll = scroll

        card = ctk.CTkFrame(scroll, fg_color=CARD_BG, corner_radius=10, border_width=1, border_color=BORDER)
        card.pack(fill="both", expand=True, pady=2)
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=12, pady=10)

        # No per-row wrapper Frame - a wrapper per checkbox doubled the
        # widget count (160 widgets instead of 80) for no benefit, since
        # CTkCheckBox itself can be pack_forget()/pack()'d directly for
        # search filtering. Fewer widgets = a noticeably less janky
        # build, since every widget added mid-pack forces a geometry
        # recompute of everything already placed.
        for class_id, name in enumerate(COCO_CLASSES):
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(inner, text=name, variable=var, command=self._on_toggle, fg_color=ACCENT, hover_color="#1d4ed8")
            cb.pack(anchor="w", pady=2, fill="x")
            self._vars[class_id] = var
            self._checks[class_id] = cb

        self.status_label = ctk.CTkLabel(self, text="0 of 80 selected", font=ctk.CTkFont(size=13), text_color=DESC_COLOR)
        self.status_label.pack(padx=24, pady=(0, 8), anchor="w")

        self.start_btn = ctk.CTkButton(
            self, text="Start Detection", command=self._start_detection,
            fg_color=ACCENT, hover_color="#1d4ed8", font=ctk.CTkFont(size=14, weight="bold"),
        )
        if self.pending:
            self.start_btn.pack(fill="x", padx=24, pady=(0, 20))

    def _update_wraplength(self, event=None):
        width = self.winfo_width()
        if width <= 1:
            width = 420  # window not realized yet - fall back to initial geometry
        self.desc.configure(wraplength=max(200, width - 48))

    def _apply_search(self):
        query = self.search_var.get().strip().lower()
        for class_id, name in enumerate(COCO_CLASSES):
            cb = self._checks[class_id]
            if query in name.lower():
                if not cb.winfo_ismapped():
                    cb.pack(anchor="w", pady=2, fill="x")
            else:
                if cb.winfo_ismapped():
                    cb.pack_forget()

    def _selected_ids(self):
        return [class_id for class_id, var in self._vars.items() if var.get()]

    def _on_toggle(self):
        selected = self._selected_ids()
        self.status_label.configure(text=f"{len(selected)} of 80 selected")
        self.on_change(selected)

    def _select_all(self):
        for var in self._vars.values():
            var.set(True)
        self._on_toggle()

    def _clear_all(self):
        for var in self._vars.values():
            var.set(False)
        self._on_toggle()

    def _start_detection(self):
        if self.on_start:
            self.on_start()
        self._close()

    def set_live_mode(self):
        """Switches this SAME window from pending -> live in place - no
        destroy/rebuild. Called the moment Start Detection is clicked
        (the filter is already fully usable pre-load; on_change safely
        no-ops until a detection_worker exists) and is a safe no-op if
        called again later (e.g. app.py's post-load callback)."""
        self.pending = False
        self.on_start = None
        self.desc.configure(text=DESC_LIVE)
        self.start_btn.pack_forget()

    def _restore_from_settings(self):
        saved = self.settings.get("coco_class_filter") or []
        saved_set = set(saved)
        for class_id, var in self._vars.items():
            var.set(class_id in saved_set)
        self.status_label.configure(text=f"{len(saved_set)} of 80 selected")
