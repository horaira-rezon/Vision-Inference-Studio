import customtkinter as ctk
import tkinter as tk
from assets.detection.coco_classes import COCO_CLASSES

BG = "#191919"
BORDER = "#2B2B2B"
CARD_BG = "#151c26"
ACCENT = "#2563eb"
DESC_COLOR = "gray60"


class CocoClassesWindow(ctk.CTkToplevel):
    def __init__(self, master, settings, on_change):
        super().__init__(master)
        self.settings = settings
        self.on_change = on_change
        self.title("COCO Classes")
        self.geometry("420x640")
        self.minsize(360, 420)
        self.configure(fg_color=BG)

        self._vars = {}       # class_id -> BooleanVar
        self._rows = {}       # class_id -> row frame (for search filtering)

        self._build_ui()
        self._restore_from_settings()

        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._focus_and_grab)

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

        desc = ctk.CTkLabel(
            self,
            text="Check whichever classes you want detected. Leave nothing checked to show all 80.",
            font=ctk.CTkFont(size=13), text_color=DESC_COLOR, justify="left", anchor="w", wraplength=370,
        )
        desc.pack(fill="x", padx=24, pady=(0, 10))

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

        for class_id, name in enumerate(COCO_CLASSES):
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", anchor="w")
            var = tk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(row, text=name, variable=var, command=self._on_toggle, progress_color=ACCENT)
            cb.pack(anchor="w", pady=2)
            self._vars[class_id] = var
            self._rows[class_id] = row

        self.status_label = ctk.CTkLabel(self, text="0 of 80 selected", font=ctk.CTkFont(size=13), text_color=DESC_COLOR)
        self.status_label.pack(padx=24, pady=(0, 16), anchor="w")

    def _apply_search(self):
        query = self.search_var.get().strip().lower()
        for class_id, name in enumerate(COCO_CLASSES):
            row = self._rows[class_id]
            if query in name.lower():
                if not row.winfo_ismapped():
                    row.pack(fill="x", anchor="w")
            else:
                if row.winfo_ismapped():
                    row.pack_forget()

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

    def _restore_from_settings(self):
        saved = self.settings.get("coco_class_filter") or []
        saved_set = set(saved)
        for class_id, var in self._vars.items():
            var.set(class_id in saved_set)
        self.status_label.configure(text=f"{len(saved_set)} of 80 selected")
