"""Dialogs: Add images (advanced picker), Settings, Export."""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from ..core import project as pj
from .common import Collapsible, open_folder, presets, run_in_thread


class Dialog(tk.Toplevel):
    def __init__(self, master, title):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.body = ttk.Frame(self, padding=16)
        self.body.pack(fill="both", expand=True)
        self.bind("<Escape>", lambda e: self.destroy())

    def modal(self):
        self.update_idletasks()
        self.grab_set()
        self.focus_set()


class AddImageryDialog(Dialog):
    COLS = (("site", 150, "Area"), ("date", 90, "Date"), ("item_id", 230, "Item"), ("size", 90, "Size"),
            ("bands", 110, "Bands"), ("dets", 70, "Detections"), ("note", 150, "Note"))

    def __init__(self, master, project):
        super().__init__(master, f"Add Imagery — {project.name}")
        self.geometry("1260x760")
        self.project = project
        self.added = 0
        self.candidates = []
        self.have = {pj.image_key(e) for e in project.imagery}
        self.thumb = None
        b = self.body
        b.columnconfigure(0, weight=1)
        b.rowconfigure(3, weight=1)

        src = ttk.LabelFrame(b, text="1. Where is the imagery?", padding=8)
        src.grid(row=0, column=0, columnspan=2, sticky="we")
        src.columnconfigure(1, weight=1)
        self.img_dir = tk.StringVar()
        self.det_dir = tk.StringVar()
        ttk.Label(src, text="Imagery folder (GeoTIFFs, searched recursively)").grid(row=0, column=0, sticky="w")
        ttk.Entry(src, textvariable=self.img_dir).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(src, text="Browse…", command=lambda: self._browse(self.img_dir)).grid(row=0, column=2)
        ttk.Label(src, text="Detections folder (optional, .geojson)").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(src, textvariable=self.det_dir).grid(row=1, column=1, sticky="we", padx=6, pady=(4, 0))
        ttk.Button(src, text="Browse…", command=lambda: self._browse(self.det_dir)).grid(row=1, column=2, pady=(4, 0))
        pre = ttk.Frame(src)
        pre.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(pre, text="Presets:").pack(side="left")
        self.preset_var = tk.StringVar()
        presets_ = presets()
        self.presets = {p[0]: p for p in presets_}
        cb = ttk.Combobox(pre, textvariable=self.preset_var, values=list(self.presets), width=60, state="readonly")
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", self._apply_preset)
        ttk.Button(pre, text="Scan", style="Accent.TButton", command=self._scan).pack(side="left", padx=(12, 0))
        self.scan_label = ttk.Label(pre, text="", style="Muted.TLabel")
        self.scan_label.pack(side="left", padx=8)

        flt = ttk.LabelFrame(b, text="2. Filter and select images", padding=8)
        flt.grid(row=1, column=0, columnspan=2, sticky="we", pady=(8, 0))
        self.f_site = tk.StringVar(value="All areas")
        self.f_from = tk.StringVar()
        self.f_to = tk.StringVar()
        self.f_text = tk.StringVar()
        self.f_dets = tk.BooleanVar(value=False)
        self.f_hide_have = tk.BooleanVar(value=True)
        ttk.Label(flt, text="Area").pack(side="left")
        self.site_cb = ttk.Combobox(flt, textvariable=self.f_site, values=["All areas"], width=22, state="readonly")
        self.site_cb.pack(side="left", padx=(4, 12))
        ttk.Label(flt, text="Date from").pack(side="left")
        ttk.Entry(flt, textvariable=self.f_from, width=11).pack(side="left", padx=4)
        ttk.Label(flt, text="to").pack(side="left")
        ttk.Entry(flt, textvariable=self.f_to, width=11).pack(side="left", padx=(4, 12))
        ttk.Label(flt, text="Search").pack(side="left")
        ttk.Entry(flt, textvariable=self.f_text, width=18).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(flt, text="only with detections", variable=self.f_dets).pack(side="left")
        ttk.Checkbutton(flt, text="hide already in project", variable=self.f_hide_have).pack(side="left", padx=8)
        for v in (self.f_site, self.f_from, self.f_to, self.f_text, self.f_dets, self.f_hide_have):
            v.trace_add("write", lambda *a: self._refill())

        tools = ttk.Frame(b)
        tools.grid(row=2, column=0, sticky="we", pady=(8, 2))
        ttk.Button(tools, text="Select all shown", command=self._select_all).pack(side="left")
        ttk.Button(tools, text="Select none", command=lambda: self.tree.selection_set(())).pack(side="left", padx=6)
        ttk.Label(tools, text="Set area of selected:").pack(side="left", padx=(16, 4))
        self.site_set = tk.StringVar()
        self.site_set_cb = ttk.Combobox(tools, textvariable=self.site_set, width=22,
                                        values=[])
        self.site_set_cb.pack(side="left")
        ttk.Button(tools, text="Apply", command=self._set_site).pack(side="left", padx=4)
        self.count_label = ttk.Label(tools, text="", style="Muted.TLabel")
        self.count_label.pack(side="right")

        tf = ttk.Frame(b)
        tf.grid(row=3, column=0, sticky="nsew")
        tf.rowconfigure(0, weight=1)
        tf.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(tf, columns=[c[0] for c in self.COLS], show="headings", selectmode="extended")
        for c, w, t in self.COLS:
            self.tree.heading(c, text=t, command=lambda c=c: self._sort(c))
            self.tree.column(c, width=w, anchor="w")
        self.tree.tag_configure("bad", foreground="#999")
        self.tree.tag_configure("have", foreground="#1a9c5c")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.sort_key = ("site", False)

        side = ttk.Frame(b, padding=(12, 0, 0, 0))
        side.grid(row=3, column=1, sticky="ns")
        ttk.Label(side, text="Preview", style="H2.TLabel").pack(anchor="w")
        self.preview = tk.Canvas(side, width=280, height=280, bg="#111", highlightthickness=0)
        self.preview.pack(pady=4)
        self.detail = ttk.Label(side, text="", wraplength=280, justify="left", style="Muted.TLabel")
        self.detail.pack(anchor="w")

        foot = ttk.Frame(b)
        foot.grid(row=4, column=0, columnspan=2, sticky="we", pady=(10, 0))
        ttk.Button(foot, text="Close", command=self.destroy).pack(side="right")
        self.add_btn = ttk.Button(foot, text="Add selected to project", style="Accent.TButton", command=self._add)
        self.add_btn.pack(side="right", padx=6)
        if presets_:
            cb.current(0)
            self._apply_preset()
        self.modal()

    def _browse(self, var):
        d = filedialog.askdirectory(parent=self, initialdir=var.get() or str(Path.home()))
        if d:
            var.set(d)

    def _apply_preset(self, _e=None):
        p = self.presets.get(self.preset_var.get())
        if p:
            self.img_dir.set(str(p[1]))
            self.det_dir.set(str(p[2]) if p[2] and p[2].exists() else "")

    def _scan(self):
        img_dir, det_dir = self.img_dir.get().strip(), self.det_dir.get().strip() or None
        if not img_dir or not Path(img_dir).is_dir():
            messagebox.showerror("Imagery folder", "Choose an existing imagery folder.", parent=self)
            return
        if det_dir and not Path(det_dir).is_dir():
            messagebox.showerror("Detections folder", "That detections folder doesn't exist.", parent=self)
            return
        self.scan_label.config(text="scanning…")
        prog = {"i": 0, "n": 0}

        def tick():
            if self.scan_label.cget("text").startswith("scanning"):
                self.scan_label.config(text=f"scanning… {prog['i']}/{prog['n']}")
                self.after(200, tick)
        self.after(200, tick)

        def done(cands):
            self.candidates = cands
            n_det = sum(1 for c in cands if c["n_detections"])
            self.scan_label.config(text=f"{len(cands)} images found, {n_det} with detections")
            sites = sorted({c["site"] for c in cands})
            self.site_cb.config(values=["All areas"] + sites)
            self.site_set_cb.config(values=sites)   # existing areas; typing a new name also works
            self._refill()

        run_in_thread(self, lambda: pj.scan_imagery(img_dir, det_dir, progress=lambda i, n: prog.update(i=i, n=n)),
                      done, on_error=lambda e: (self.scan_label.config(text=""),
                                                messagebox.showerror("Scan failed", str(e), parent=self)))

    def _visible(self):
        site, t = self.f_site.get(), self.f_text.get().strip().lower()
        d0, d1 = self.f_from.get().strip(), self.f_to.get().strip()
        out = []
        for i, c in enumerate(self.candidates):
            if site not in ("", "All areas") and c["site"] != site:
                continue
            if d0 and (c["date"] or "") < d0:
                continue
            if d1 and (c["date"] or "9999") > d1:
                continue
            if t and t not in c["path"].lower():
                continue
            if self.f_dets.get() and not c["n_detections"]:
                continue
            if self.f_hide_have.get() and pj.image_key(c) in self.have:
                continue
            out.append(i)
        k, rev = self.sort_key
        keyf = {"dets": lambda i: self.candidates[i]["n_detections"],
                "size": lambda i: self.candidates[i].get("width", 0) * self.candidates[i].get("height", 0)}.get(
            k, lambda i: str(self.candidates[i].get(k) or ""))
        return sorted(out, key=keyf, reverse=rev)

    def _refill(self):
        sel = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        for i in self._visible():
            c = self.candidates[i]
            note, tag = "", ()
            if c.get("error"):
                note, tag = c["error"], ("bad",)
            elif pj.image_key(c) in self.have:
                note, tag = "already in project", ("have",)
            elif not c["detections"]:
                note = "scene check only"
            size = f"{c['width']}×{c['height']}" if "width" in c else ""
            bands = f"{c['count']} × {c['dtype']}" if "count" in c else ""
            self.tree.insert("", "end", iid=str(i), tags=tag,
                             values=(c["site"], c["date"] or "?", c["item_id"], size, bands, c["n_detections"], note))
        keep = [s for s in sel if self.tree.exists(s)]
        if keep:
            self.tree.selection_set(keep)
        self._count()

    def _sort(self, col):
        k, rev = self.sort_key
        self.sort_key = (col, not rev if k == col else False)
        self._refill()

    def _count(self):
        sel = [self.candidates[int(s)] for s in self.tree.selection()]
        ok = [c for c in sel if not c.get("error") and pj.image_key(c) not in self.have]
        self.count_label.config(text=f"{len(self.tree.get_children())} shown · {len(ok)} selected to add")
        self.add_btn.config(text=f"Add {len(ok)} selected to project")

    def _select_all(self):
        self.tree.selection_set(self.tree.get_children())

    def _set_site(self):
        site = self.site_set.get().strip()
        if not site:
            return
        for s in self.tree.selection():
            self.candidates[int(s)]["site"] = site
        self._refill()

    def _on_select(self, _e=None):
        self._count()
        focus = self.tree.focus() or (self.tree.selection()[0] if self.tree.selection() else None)
        if not focus:
            return
        c = self.candidates[int(focus)]
        self.preview.delete("all")
        if c.get("error"):
            self.preview.create_text(140, 140, text=c["error"], fill="white")
        else:
            img = pj.render_thumbnail(c["path"], 280, self.project.settings.get("rgb_bands", "auto"))
            if img is not None:
                self.thumb = ImageTk.PhotoImage(img)
                self.preview.create_image(140, 140, image=self.thumb)
        lines = [Path(c["path"]).name, c["path"]]
        if "res_m" in c:
            lines.append(f"{c['res_m']:.1f} m/px · {c['crs']} · {c['extent_m'][0] / 1000:.1f}×"
                         f"{c['extent_m'][1] / 1000:.1f} km")
        lines.append("detections: " + (", ".join(Path(d).name for d in c["detections"]) or "none"))
        self.detail.config(text="\n".join(lines))

    def _add(self):
        sel = [self.candidates[int(s)] for s in self.tree.selection()]
        n = self.project.add_imagery(sel)
        if not n:
            messagebox.showinfo("Nothing added", "Select one or more images that aren't already in the project.",
                                parent=self)
            return
        self.added += n
        self.have = {pj.image_key(e) for e in self.project.imagery}
        self.scan_label.config(text=f"added {n} image{'s' if n != 1 else ''} ({len(self.project.imagery)} in project)")
        self._refill()


class SettingsDialog(Dialog):
    """Basic (name, notes, classes) up front; display and scoring details under Advanced."""

    def __init__(self, master, project):
        super().__init__(master, f"Settings — {project.name}")
        self.project = project
        self.changed = False
        s = project.settings
        b = self.body
        self.name = tk.StringVar(value=project.name)
        self.use_classes = tk.BooleanVar(value=bool(project.classes))
        self.classes = tk.StringVar(value=", ".join(project.classes))
        self.radius = tk.IntVar(value=s["crop_radius_m"])
        self.canvas = tk.IntVar(value=s["canvas_size"])
        self.scene = tk.BooleanVar(value=s["scene_checks"])
        self.rgb = tk.StringVar(value=s["rgb_bands"])
        self.lenient = tk.BooleanVar(value=s["lenient_edited"])
        self.stretch = tk.StringVar(value=s.get("stretch", "image"))
        self.edit_valid = tk.BooleanVar(value=s.get("edit_marks_valid", True))
        self.blind = tk.BooleanVar(value=s.get("holdout_blind", True))

        ttk.Label(b, text="Name").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(b, textvariable=self.name, width=46).grid(row=0, column=1, sticky="we", pady=4)
        ttk.Label(b, text="Notes").grid(row=1, column=0, sticky="nw", pady=4)
        self.desc = tk.Text(b, width=46, height=3, wrap="word")
        self.desc.insert("1.0", project.description)
        self.desc.grid(row=1, column=1, sticky="we", pady=4)
        ttk.Checkbutton(b, text="Sort objects into types (number keys 1-9)", variable=self.use_classes,
                        command=self._sync).grid(row=2, column=1, sticky="w", pady=(8, 2))
        self.cls_entry = ttk.Entry(b, textvariable=self.classes, width=46)
        self.cls_entry.grid(row=3, column=1, sticky="we")
        ttk.Label(b, text="comma-separated, up to 9; renaming doesn't relabel boxes already typed",
                  style="Muted.TLabel").grid(row=4, column=1, sticky="w")

        adv = Collapsible(b, "Advanced")
        adv.grid(row=5, column=0, columnspan=2, sticky="we", pady=(12, 0))
        a = adv.body
        rows = [("Detection view radius (m)", ttk.Spinbox(a, from_=40, to=1500, increment=10, textvariable=self.radius, width=8)),
                ("Image canvas (px)", ttk.Spinbox(a, from_=400, to=1400, increment=20, textvariable=self.canvas, width=8)),
                ("RGB bands", ttk.Combobox(a, textvariable=self.rgb, values=["auto", "3,2,1", "1,2,3", "1"], width=10))]
        for i, (lbl, w) in enumerate(rows):
            ttk.Label(a, text=lbl).grid(row=i, column=0, sticky="w", pady=3, padx=(12, 8))
            w.grid(row=i, column=1, sticky="w", pady=3)
        ttk.Label(a, text="auto = band names, else 1,2,3 for 8-bit, else 3,2,1; \"1\" = grayscale (SAR in dB)",
                  style="Muted.TLabel").grid(row=3, column=1, sticky="w")
        ttk.Label(a, text="Contrast").grid(row=4, column=0, sticky="nw", pady=3, padx=(12, 8))
        st = ttk.Frame(a)
        st.grid(row=4, column=1, sticky="w", pady=3)
        ttk.Radiobutton(st, text="whole image (stable colours)", value="image", variable=self.stretch).pack(anchor="w")
        ttk.Radiobutton(st, text="per view (max detail, noisy on open water)", value="view", variable=self.stretch).pack(anchor="w")
        for i, (txt, var) in enumerate((("Whole-image scene check for every image", self.scene),
                                        ("Fixing a detection's box also marks it Valid", self.edit_valid),
                                        ("Count an edited-but-undecided detection as an object on export", self.lenient),
                                        ("Show hold-out images blind (no detector/model boxes)", self.blind))):
            ttk.Checkbutton(a, text=txt, variable=var).grid(row=5 + i, column=1, sticky="w", pady=2)

        btns = ttk.Frame(b)
        btns.grid(row=6, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Save", style="Accent.TButton", command=self._save).pack(side="right", padx=6)
        self._sync()
        self.modal()

    def _sync(self):
        self.cls_entry.state(["!disabled"] if self.use_classes.get() else ["disabled"])

    def _save(self):
        classes = [c.strip() for c in self.classes.get().split(",") if c.strip()][:9] if self.use_classes.get() else []
        try:
            self.project.settings.update({
                "crop_radius_m": int(self.radius.get()), "canvas_size": int(self.canvas.get()),
                "scene_checks": self.scene.get(), "rgb_bands": self.rgb.get().strip() or "auto",
                "lenient_edited": self.lenient.get(), "stretch": self.stretch.get(),
                "edit_marks_valid": self.edit_valid.get(), "holdout_blind": self.blind.get(),
                "classes": classes})
        except (ValueError, tk.TclError):
            messagebox.showerror("Settings", "Radius and canvas size must be whole numbers.", parent=self)
            return
        self.project.name = self.name.get().strip() or self.project.name
        self.project.description = self.desc.get("1.0", "end").strip()
        self.project.save_meta()
        self.changed = True
        self.destroy()


class ExportDialog(Dialog):
    """One button for the common case; dataset details under Advanced."""

    def __init__(self, master, project):
        super().__init__(master, f"Export labels — {project.name}")
        self.project = project
        b = self.body
        prog = project.image_progress()
        n_complete = sum(p["complete"] for p in prog.values())
        ttk.Label(b, text=f"{n_complete} of {len(prog)} images are complete (every detection decided and the whole "
                          f"image checked).", wraplength=500).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(b, text="Writes labels.geojson, negatives.geojson, image_summary.csv and a YOLO training set "
                          "of the complete images.", style="Muted.TLabel", wraplength=500).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.only_complete = tk.BooleanVar(value=True)
        self.yolo = tk.BooleanVar(value=True)
        self.lenient = tk.BooleanVar(value=project.settings.get("lenient_edited", True))
        self.yolo_classes = tk.BooleanVar(value=False)
        self.out = tk.StringVar(value="")

        adv = Collapsible(b, "Advanced")
        adv.grid(row=2, column=0, columnspan=3, sticky="we", pady=(10, 0))
        a = adv.body
        ttk.Checkbutton(a, text="Write the YOLO training set", variable=self.yolo).grid(row=0, column=0, columnspan=3, sticky="w", padx=12)
        ttk.Checkbutton(a, text="YOLO: complete images only (recommended -- unchecked images hide unlabelled objects)",
                        variable=self.only_complete).grid(row=1, column=0, columnspan=3, sticky="w", padx=12)
        if project.classes:
            ttk.Checkbutton(a, text=f"YOLO: one class per type ({', '.join(project.classes)}, unclassified)",
                            variable=self.yolo_classes).grid(row=2, column=0, columnspan=3, sticky="w", padx=12)
        ttk.Checkbutton(a, text="Count edited-but-undecided detections as objects", variable=self.lenient
                        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=12)
        ttk.Label(a, text="Output folder").grid(row=4, column=0, sticky="w", padx=12, pady=(8, 0))
        ttk.Entry(a, textvariable=self.out, width=40).grid(row=4, column=1, sticky="we", pady=(8, 0))
        ttk.Button(a, text="Browse…", command=self._browse).grid(row=4, column=2, padx=(6, 0), pady=(8, 0))
        ttk.Label(a, text="(blank = <project>/exports/<date-time>)", style="Muted.TLabel").grid(row=5, column=1, sticky="w")

        self.status = ttk.Label(b, text="")
        self.status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        btns = ttk.Frame(b)
        btns.grid(row=4, column=0, columnspan=3, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")
        self.go = ttk.Button(btns, text="Export", style="Accent.TButton", command=self._export)
        self.go.pack(side="right", padx=6)
        self.modal()

    def _browse(self):
        d = filedialog.askdirectory(parent=self, initialdir=str(self.project.root))
        if d:
            self.out.set(d)

    def _export(self):
        self.project.settings["lenient_edited"] = self.lenient.get()
        self.project.save_meta()
        self.go.state(["disabled"])
        prog = {"i": 0, "n": 0}
        self.status.config(text="exporting…")

        def tick():
            if str(self.go["state"]) == "disabled":
                self.status.config(text=f"exporting… {prog['i']}/{prog['n']} images")
                self.after(250, tick)
        self.after(250, tick)

        def done(res):
            out, s = res
            self.go.state(["!disabled"])
            self.status.config(text=f"✓ {s['object_labels']} object labels, {s['negatives']} negatives, "
                                    f"{s['yolo_images']} YOLO images / {s['yolo_boxes']} boxes")
            if messagebox.askyesno("Exported", f"Exported to\n{out}\n\nOpen the folder?", parent=self):
                open_folder(out)

        def fail(e):
            self.go.state(["!disabled"])
            self.status.config(text="")
            messagebox.showerror("Export failed", f"{e.__class__.__name__}: {e}", parent=self)

        run_in_thread(self, lambda: pj.export_labels(self.project, self.out.get().strip() or None,
                                                     only_complete=self.only_complete.get(), yolo=self.yolo.get(),
                                                     yolo_classes=self.yolo_classes.get(),
                                                     progress=lambda i, n: prog.update(i=i, n=n)), done, fail)
