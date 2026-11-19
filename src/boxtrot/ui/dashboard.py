"""Project dashboard: image table, preview, Stats and Model tabs."""
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import ImageTk

from ..core import assist
from ..core import project as pj
from .common import RIGHT, SHORTCUTS, menu_button, open_folder
from .model_panel import ModelPanel
from .stats import StatsPanel


class ProjectView(ttk.Frame):
    COLS = (("site", 140, "Area"), ("date", 90, "Date"), ("item_id", 220, "Item"), ("dets", 75, "Detections"),
            ("decided", 70, "Decided"), ("valid", 55, "Valid"), ("invalid", 60, "Invalid"),
            ("added", 70, "Added"), ("scene", 100, "Scene check"), ("objects", 65, "Objects"),
            ("complete", 75, "Complete"), ("set", 95, "Set"))

    def __init__(self, master, app, tab=None, welcome=None):
        super().__init__(master, padding=12)
        self.app = app
        self.project = app.project
        self.thumb = None
        head = ttk.Frame(self)
        head.pack(fill="x")
        ttk.Label(head, text=self.project.name, style="Title.TLabel").pack(side="left")
        ttk.Button(head, text="Close", command=app.close_project).pack(side="right")
        ttk.Button(head, text="Settings", command=app.edit_settings).pack(side="right", padx=6)
        if self.project.description:
            ttk.Label(self, text=self.project.description, style="Muted.TLabel", wraplength=1100).pack(anchor="w")

        if welcome:
            self._welcome(welcome)
        started = any(v.get("status") for v in self.project.state.values()) or self.project.additions["features"]
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(10, 6))
        ttk.Button(bar, text="▶  " + ("Resume review" if started else "Start review"), style="Accent.TButton",
                   command=app.start_review).pack(side="left")
        ttk.Button(bar, text="＋ Add images", command=app.add_imagery).pack(side="left", padx=6)
        ttk.Button(bar, text="Export labels", command=app.export_labels).pack(side="left")
        menu_button(bar, "More…", [
            ("Review the selected image", self._review_selected),
            ("Review boxes I added", lambda: app.start_review(mode="additions")),
            None,
            ("Remove selected images from project", self._remove),
            ("Import review data from another folder…", app.import_review_data),
            None,
            ("Open project folder", lambda: open_folder(self.project.root)),
            ("Keyboard shortcuts", lambda: messagebox.showinfo("Keyboard shortcuts", SHORTCUTS)),
        ]).pack(side="left", padx=6)

        self.summary = ttk.Label(self, text="", style="H2.TLabel")
        self.summary.pack(anchor="w", pady=(4, 4))
        self.bar = ttk.Progressbar(self, maximum=1, length=400)
        self.bar.pack(anchor="w", pady=(0, 8))

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        main = ttk.Frame(self.tabs, padding=(0, 6, 0, 0))
        self.tabs.add(main, text="Images")
        self.stats = StatsPanel(self.tabs, self.project)
        self.tabs.add(self.stats, text="Stats")
        self.model_holder = ttk.Frame(self.tabs)
        self.tabs.add(self.model_holder, text="Model (assisted labelling)")
        self.model_panel = None
        self.tabs.bind("<<NotebookTabChanged>>", self._on_tab)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(main, columns=[c[0] for c in self.COLS], show="headings", selectmode="extended")
        for c, w, t in self.COLS:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="w" if c in ("site", "date", "item_id", "scene") else "center")
        # fewer columns by default; item id / valid / invalid are in the preview panel's details
        self.tree.configure(displaycolumns=[c[0] for c in self.COLS if c[0] not in ("item_id", "valid", "invalid")])
        self.tree.tag_configure("complete", foreground="#1a7c4c")
        self.tree.tag_configure("unreadable", foreground="#999")
        self.tree.tag_configure("started", foreground="#9a6a00")
        self.tree.tag_configure("holdout", background="#eef3fb")
        sb = ttk.Scrollbar(main, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<Double-1>", lambda e: self._review_selected())
        self.tree.bind("<Return>", lambda e: self._review_selected())
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind(f"<Button-{RIGHT}>", self._context_menu)

        side = ttk.Frame(main, padding=(12, 0, 0, 0))
        side.grid(row=0, column=2, sticky="ns")
        self.preview = tk.Canvas(side, width=280, height=280, bg="#111", highlightthickness=0)
        self.preview.pack()
        self.detail = ttk.Label(side, text="Select an image to preview it.", wraplength=280, justify="left",
                                style="Muted.TLabel")
        self.detail.pack(anchor="w", pady=6)

        if not self.project.imagery:
            empty = ttk.Frame(self, padding=20)
            empty.place(relx=0.35, rely=0.5, anchor="center")
            ttk.Label(empty, text="No imagery yet", style="H2.TLabel").pack()
            ttk.Button(empty, text="＋ Add Imagery…", style="Accent.TButton", command=app.add_imagery).pack(pady=8)
        self._fill()
        if tab == "Model" or (app.assist_job is not None and app.assist_job["root"] == str(self.project.root)):
            self.tabs.select(2)   # asked for, or a job is running: show its progress
            self._on_tab()        # build now; the tab-changed event only fires once Tk idles

    def _on_tab(self, _e=None):
        i = self.tabs.index("current")
        if i == 1:
            self.stats.refresh()
        elif i == 2 and self.model_panel is None:
            self.model_panel = ModelPanel(self.model_holder, self.app, self)
            self.model_panel.pack(fill="both", expand=True)

    def _welcome(self, mode):
        msg = {"review": "Project created. Press Start review: decide each detection (v valid, x invalid), "
                         "fix boxes by dragging their corners, then check each whole image for anything missed.",
               "assist": "Project created. The app has picked test images (hold-out) and a first set of images to "
                         "label -- follow the Next step button on the Model tab.",
               "plain": "Project created. Press Start review and mark every object on each image (a, then drag a "
                        "box); press v when an image is done."}.get(mode, "")
        box = tk.Frame(self, bg="#eef3fb", padx=12, pady=8)
        box.pack(fill="x", pady=(8, 0))
        tk.Label(box, text=msg, bg="#eef3fb", fg="#1a1a19", justify="left", wraplength=1050,
                 font=("TkDefaultFont", 10)).pack(side="left")
        ttk.Button(box, text="Dismiss", command=box.destroy).pack(side="right")

    def _context_menu(self, event):
        row = self.tree.identify_row(event.y)
        if not row:
            return
        if row not in self.tree.selection():
            self.tree.selection_set(row)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Review this image", command=self._review_selected)
        m.add_command(label="Remove from project", command=self._remove)
        m.tk_popup(event.x_root, event.y_root)

    def reload_rows(self):
        self.tree.delete(*self.tree.get_children())
        self._fill()

    def _fill(self):
        prog = self.project.image_progress()
        self.prog = prog
        self.entries = {pj.image_key(e): e for e in self.project.imagery}
        ast = assist.load_assist(self.project)
        hold, sugg, prel = set(ast["holdout"]), set(ast["suggested"]), set(ast.get("prelabels", {}))
        for k, e in self.entries.items():
            p = prog[k]
            pre = k in prel or any(assist.is_model_file(d) for d in (e.get("detections") or []))
            set_ = "hold-out" if k in hold else "suggested" if k in sugg else "pre-labelled" if pre else ""
            decided = p["valid"] + p["invalid"] + p["skipped"]
            scene = {"valid": "done", "invalid": "unreadable", None: "pending"}.get(p["scene"], p["scene"]) \
                if p["has_scene"] else "-"
            tag = "complete" if p["complete"] else "unreadable" if p["unreadable"] else \
                "started" if decided or p["added"] or p["scene"] else ""
            tags = tuple(t for t in (tag, "holdout" if k in hold else "") if t)
            self.tree.insert("", "end", iid=k, tags=tags, values=(
                e["site"], e.get("date") or "?", e["item_id"], p["n_det"],
                f"{decided}/{p['n_det']}" + (f" +{p['edited_only']}e" if p["edited_only"] else ""),
                p["valid"], p["invalid"], p["added"], scene, p["objects"], "yes" if p["complete"] else "", set_))
        n = len(prog)
        n_c = sum(p["complete"] for p in prog.values())
        n_det = sum(p["n_det"] for p in prog.values())
        n_dec = sum(p["valid"] + p["invalid"] + p["skipped"] for p in prog.values())
        n_add = sum(p["added"] for p in prog.values())
        n_v = sum(p["objects"] for p in prog.values())
        self.summary.config(text=f"{n} images · {n_c} complete · {n_dec}/{n_det} detections decided · "
                                 f"{n_add} missed objects added · {n_v} objects labelled")
        self.bar.config(maximum=max(n, 1), value=n_c)

    def _selected_keys(self):
        return list(self.tree.selection())

    def _review_selected(self):
        keys = self._selected_keys()
        if not keys:
            messagebox.showinfo("Select an image", "Select an image in the list first (or use Resume Review).")
            return
        self.app.start_review(image_key=keys[0])

    def _remove(self):
        keys = self._selected_keys()
        if not keys:
            return
        if not messagebox.askyesno("Remove imagery", f"Remove {len(keys)} image(s) from the project?\n\n"
                                                     "Review decisions are kept on disk and come back if "
                                                     "the same image is added again."):
            return
        self.project.remove_imagery(keys)
        self.app.show_project()

    def _on_select(self, _e=None):
        keys = self._selected_keys()
        if not keys:
            return
        e, p = self.entries[keys[0]], self.prog[keys[0]]
        self.preview.delete("all")
        img = pj.render_thumbnail(e["path"], 280, self.project.settings.get("rgb_bands", "auto"))
        if img is None:
            self.preview.create_text(140, 140, text="image file not found", fill="white")
        else:
            self.thumb = ImageTk.PhotoImage(img)
            self.preview.create_image(140, 140, image=self.thumb)
        self.detail.config(text="\n".join([
            f"{e['site']}  ·  {e.get('date')}", e["item_id"], e["path"],
            f"{p['n_det']} detections: {p['valid']} valid, {p['invalid']} invalid, {p['skipped']} skipped, "
            f"{p['undecided']} undecided" + (f", {p['edited_only']} edited only" if p["edited_only"] else ""),
            f"{p['added']} missed objects added", "Double-click to review."]))
