"""Guided new-project setup."""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from .. import config
from ..core import assist
from ..core import project as pj
from .common import presets, run_in_thread


class SetupWizard(tk.Toplevel):
    """Guided new-project setup: name -> imagery -> detector output -> what you label -> how to start."""
    TITLES = ["Name your project", "Choose the imagery", "Detector output", "What are you labelling?",
              "How do you want to start?"]

    def __init__(self, master):
        super().__init__(master)
        self.title("New project")
        self.transient(master)
        self.geometry("860x640")
        self.result = None
        self.start_mode = None
        self.step = 0
        self.name = tk.StringVar()
        self.parent = tk.StringVar(value=str(config.default_projects_dir()))
        self.desc = ""
        self.img_dir = tk.StringVar()
        self.det_dir = tk.StringVar()
        self.has_dets = tk.StringVar(value="no")
        self.cands = []
        self.scanned = None            # (img_dir, det_dir) the candidates came from
        self.site_on = {}              # site -> BooleanVar (kept across rescans)
        self.d_from = tk.StringVar()
        self.d_to = tk.StringVar()
        self.object_name = tk.StringVar(value="")
        self.use_classes = tk.BooleanVar(value=False)
        self.classes = tk.StringVar(value="")
        self.start = tk.StringVar(value="")
        self.presets = {p[0]: p for p in presets()}

        head = ttk.Frame(self, padding=(24, 18, 24, 0))
        head.pack(fill="x")
        self.step_lbl = ttk.Label(head, text="", style="Muted.TLabel")
        self.step_lbl.pack(anchor="w")
        self.title_lbl = ttk.Label(head, text="", style="Title.TLabel")
        self.title_lbl.pack(anchor="w")
        self.pbar = ttk.Progressbar(head, maximum=len(self.TITLES))
        self.pbar.pack(fill="x", pady=(8, 0))
        self.body = ttk.Frame(self, padding=24)
        self.body.pack(fill="both", expand=True)
        foot = ttk.Frame(self, padding=(24, 0, 24, 18))
        foot.pack(fill="x")
        ttk.Button(foot, text="Cancel", command=self.destroy).pack(side="left")
        self.msg = ttk.Label(foot, text="", foreground="#c7403a", wraplength=440)
        self.msg.pack(side="left", padx=12)
        self.next_btn = ttk.Button(foot, text="Next ▶", style="Accent.TButton", command=self.next)
        self.next_btn.pack(side="right")
        self.back_btn = ttk.Button(foot, text="◀ Back", command=self.back)
        self.back_btn.pack(side="right", padx=6)
        self.bind("<Escape>", lambda e: self.destroy())
        self.show()
        self.update_idletasks()
        self.grab_set()

    # -- frame --
    def show(self):
        for w in self.body.winfo_children():
            w.destroy()
        self.msg.config(text="")
        self.step_lbl.config(text=f"Step {self.step + 1} of {len(self.TITLES)}")
        self.title_lbl.config(text=self.TITLES[self.step])
        self.pbar.config(value=self.step + 1)
        self.back_btn.state(["disabled"] if self.step == 0 else ["!disabled"])
        self.next_btn.config(text="Create project" if self.step == len(self.TITLES) - 1 else "Next ▶")
        [self._s_name, self._s_imagery, self._s_dets, self._s_labels, self._s_start][self.step]()

    def back(self):
        self._leave()
        if self.step > 0:
            self.step -= 1
            self.show()

    def next(self):
        self._leave()
        ok = [self._v_name, self._v_imagery, self._v_dets, self._v_labels, self._v_start][self.step]()
        if ok is True:
            if self.step == len(self.TITLES) - 1:
                self._finish()
            else:
                self.step += 1
                self.show()

    def _leave(self):
        if self.step == 0 and hasattr(self, "_desc_txt") and self._desc_txt.winfo_exists():
            self.desc = self._desc_txt.get("1.0", "end").strip()

    def _row(self, r, label, var, browse=None, width=56):
        ttk.Label(self.body, text=label).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 10))
        e = ttk.Entry(self.body, textvariable=var, width=width)
        e.grid(row=r, column=1, sticky="we", pady=4)
        if browse:
            ttk.Button(self.body, text="Browse…", command=browse).grid(row=r, column=2, padx=(6, 0))
        return e

    def _browse_dir(self, var, then=None):
        d = filedialog.askdirectory(parent=self, initialdir=var.get() or str(Path.home()))
        if d:
            var.set(d)
            if then:
                then()

    # -- step 1: name --
    def _s_name(self):
        e = self._row(0, "Project name", self.name)
        e.focus_set()
        self._row(1, "Save in", self.parent, lambda: self._browse_dir(self.parent))
        self._where = ttk.Label(self.body, text="", style="Muted.TLabel")
        self._where.grid(row=2, column=1, sticky="w")
        ttk.Label(self.body, text="Notes (optional)").grid(row=3, column=0, sticky="nw", pady=4)
        self._desc_txt = tk.Text(self.body, width=56, height=4, wrap="word")
        self._desc_txt.insert("1.0", self.desc)
        self._desc_txt.grid(row=3, column=1, sticky="we", pady=4)
        ttk.Label(self.body, text="Everything you do is saved in this project folder as you go -- you can close "
                                  "the app at any time and pick up where you left off.", style="Muted.TLabel",
                  wraplength=600).grid(row=4, column=1, sticky="w", pady=(12, 0))

        def upd(*_):
            n = self.name.get().strip()
            self._where.config(text=f"→ {Path(self.parent.get()) / pj.slugify(n)}" if n else "")
        self._traces = [self.name.trace_add("write", upd), self.parent.trace_add("write", upd)]
        upd()

    def _v_name(self):
        for t, v in zip(self._traces, (self.name, self.parent)):
            try:
                v.trace_remove("write", t)
            except tk.TclError:
                pass
        n = self.name.get().strip()
        if not n:
            self.msg.config(text="Give the project a name.")
            return False
        if (Path(self.parent.get()).expanduser() / pj.slugify(n) / "project.json").exists():
            self.msg.config(text="A project with that name already exists there -- pick another name.")
            return False
        return True

    # -- step 2: imagery --
    def _s_imagery(self):
        b = self.body
        b.columnconfigure(1, weight=1)
        if self.presets:
            ttk.Label(b, text="Recent folders").grid(row=0, column=0, sticky="w", pady=4)
            pv = tk.StringVar()
            cb = ttk.Combobox(b, textvariable=pv, values=list(self.presets), state="readonly", width=60)
            cb.grid(row=0, column=1, columnspan=2, sticky="we", pady=4)

            def use_preset(_e=None):
                p = self.presets[pv.get()]
                self.img_dir.set(str(p[1]))
                if p[2] and p[2].exists():
                    self.det_dir.set(str(p[2]))
                    self.has_dets.set("yes")
                else:
                    self.det_dir.set("")
                    self.has_dets.set("no")
                self._scan()
            cb.bind("<<ComboboxSelected>>", use_preset)
        self._row(1, "Imagery folder", self.img_dir, lambda: self._browse_dir(self.img_dir, self._scan))
        ttk.Label(b, text="Rasters (GeoTIFF, JP2, ...) anywhere inside this folder are found automatically.",
                  style="Muted.TLabel").grid(row=2, column=1, sticky="w")
        self._scan_lbl = ttk.Label(b, text="")
        self._scan_lbl.grid(row=3, column=1, sticky="w", pady=(10, 4))
        self._sites = ttk.Frame(b)
        self._sites.grid(row=4, column=0, columnspan=3, sticky="we")
        dr = ttk.Frame(b)
        dr.grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(dr, text="Only dates from").pack(side="left")
        ttk.Entry(dr, textvariable=self.d_from, width=11).pack(side="left", padx=4)
        ttk.Label(dr, text="to").pack(side="left")
        ttk.Entry(dr, textvariable=self.d_to, width=11).pack(side="left", padx=4)
        ttk.Label(dr, text="(YYYY-MM-DD, optional)", style="Muted.TLabel").pack(side="left", padx=6)
        self._count = ttk.Label(b, text="", style="H2.TLabel")
        self._count.grid(row=6, column=0, columnspan=3, sticky="w", pady=(14, 0))
        ttk.Label(b, text="Need to pick individual images? Do it later with Add images on the project screen.",
                  style="Muted.TLabel").grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
        for v in (self.d_from, self.d_to):
            v.trace_add("write", lambda *a: self._update_count())
        if self.cands and self.scanned and self.scanned[0] == self.img_dir.get().strip():
            self._fill_sites()
        elif self.img_dir.get().strip():
            self._scan()

    def _scan(self, then=None):
        img = self.img_dir.get().strip()
        det = self.det_dir.get().strip() if self.has_dets.get() == "yes" else ""
        if not img or not Path(img).is_dir():
            self.msg.config(text="Choose an existing imagery folder.")
            return
        self.msg.config(text="")
        lbl = getattr(self, "_scan_lbl" if self.step == 1 else "_match_lbl", None)
        prog = {"i": 0, "n": 0}
        busy = {"on": True}

        def tick():
            if busy["on"] and lbl is not None and lbl.winfo_exists():
                lbl.config(text=f"Looking at the images… {prog['i']}/{prog['n']}")
                self.after(200, tick)
        tick()
        self.next_btn.state(["disabled"])

        def done(cands):
            busy["on"] = False
            self.next_btn.state(["!disabled"])
            self.cands = cands
            self.scanned = (img, det)
            for s in {c["site"] for c in cands}:
                self.site_on.setdefault(s, tk.BooleanVar(value=True))
            if self.step == 1 and self._sites.winfo_exists():
                self._fill_sites()
            if then:
                then()

        def fail(e):
            busy["on"] = False
            self.next_btn.state(["!disabled"])
            self.msg.config(text=f"Couldn't read that folder: {e}")
        run_in_thread(self, lambda: pj.scan_imagery(img, det or None, progress=lambda i, n: prog.update(i=i, n=n)),
                      done, fail)

    def _fill_sites(self):
        for w in self._sites.winfo_children():
            w.destroy()
        good = [c for c in self.cands if not c.get("error")]
        bad = len(self.cands) - len(good)
        self._scan_lbl.config(text=f"Found {len(good)} usable images" + (f" ({bad} files can't be used: no "
                                                                           f"georeferencing)" if bad else "")
                              + (". Untick any area you don't want:" if good else "."))
        by = {}
        for c in good:
            by.setdefault(c["site"], []).append(c)
        for i, (site, cs) in enumerate(sorted(by.items())):
            dates = sorted(c["date"] for c in cs if c.get("date"))
            span = f"{dates[0]} → {dates[-1]}" if dates else "no dates"
            nd = sum(1 for c in cs if c["n_detections"])
            ttk.Checkbutton(self._sites, text=f"{site}   —   {len(cs)} images, {span}"
                                              + (f", {nd} with detections" if nd else ""),
                            variable=self.site_on[site], command=self._update_count).grid(row=i, column=0, sticky="w")
        self._update_count()

    def selected(self):
        d0, d1 = self.d_from.get().strip(), self.d_to.get().strip()
        return [c for c in self.cands if not c.get("error") and self.site_on.get(c["site"], tk.BooleanVar(value=True)).get()
                and not (d0 and (c.get("date") or "") < d0) and not (d1 and (c.get("date") or "9999") > d1)]

    def _update_count(self):
        if getattr(self, "_count", None) is not None and self._count.winfo_exists():
            n = len(self.selected())
            self._count.config(text=f"{n} image{'s' if n != 1 else ''} will be added" if self.cands else "")

    def _v_imagery(self):
        if not self.cands or not self.scanned or self.scanned[0] != self.img_dir.get().strip():
            self.msg.config(text="Choose an imagery folder (it's scanned automatically).")
            return False
        if not self.selected():
            self.msg.config(text="No images selected -- tick at least one area or widen the dates.")
            return False
        return True

    # -- step 3: detections --
    def _s_dets(self):
        b = self.body
        ttk.Label(b, text="Do you already have detections for these images -- boxes or points from a detector "
                          "(GeoJSON files)?", wraplength=700).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(b, text="No -- I'll mark the objects myself", value="no", variable=self.has_dets,
                        command=self._sync_dets).grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Radiobutton(b, text="Yes -- review and correct a detector's output", value="yes", variable=self.has_dets,
                        command=self._sync_dets).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)
        self._det_lbl = ttk.Label(b, text="Detections folder")
        self._det_lbl.grid(row=3, column=0, sticky="w", pady=(10, 4), padx=(24, 10))
        self._det_entry = ttk.Entry(b, textvariable=self.det_dir, width=52)
        self._det_entry.grid(row=3, column=1, sticky="we", pady=(10, 4))
        self._det_btn = ttk.Button(b, text="Browse…", command=lambda: self._browse_dir(self.det_dir, self._match))
        self._det_btn.grid(row=3, column=2, padx=(6, 0), pady=(10, 4))
        self._match_lbl = ttk.Label(b, text="", wraplength=640)
        self._match_lbl.grid(row=4, column=1, columnspan=2, sticky="w")
        ttk.Label(b, text="Detection files are matched to images by name and area folder, e.g. "
                          "north/scene_042.tif ↔ north/scene_042.geojson or scene_042_detections.geojson.", style="Muted.TLabel",
                  wraplength=640).grid(row=5, column=1, columnspan=2, sticky="w", pady=(6, 0))
        self._sync_dets()
        if self.has_dets.get() == "yes" and self.scanned == (self.img_dir.get().strip(), self.det_dir.get().strip()):
            self._show_match()

    def _sync_dets(self):
        st = ["!disabled"] if self.has_dets.get() == "yes" else ["disabled"]
        for w in (self._det_entry, self._det_btn):
            w.state(st)

    def _match(self, then=None):
        self._scan(then=lambda: (self._show_match(), then() if then else None))

    def _show_match(self):
        if not self._match_lbl.winfo_exists():
            return
        sel = self.selected()
        n_img = sum(1 for c in sel if c["detections"])
        n_det = sum(c["n_detections"] for c in sel)
        self._match_lbl.config(text=f"{n_img} of {len(sel)} selected images have detections ({n_det} in total)."
                               if n_img else "No detection files matched these images -- check the folder, or "
                                             "choose No.", foreground="" if n_img else "#c7403a")

    def _v_dets(self):
        if self.has_dets.get() == "no":
            if self.scanned and self.scanned[1]:   # drop detections from an earlier scan
                for c in self.cands:
                    c["detections"], c["n_detections"] = [], 0
                self.scanned = (self.scanned[0], "")
            return True
        d = self.det_dir.get().strip()
        if not d or not Path(d).is_dir():
            self.msg.config(text="Choose the folder with the detection files.")
            return False
        if self.scanned != (self.img_dir.get().strip(), d):
            self._match(then=lambda: self._advance_if_matched())
            return "pending"
        return self._advance_if_matched(advance=False)

    def _advance_if_matched(self, advance=True):
        if not any(c["detections"] for c in self.selected()):
            self._show_match()
            self.msg.config(text="No detections matched -- check the folder or choose No.")
            return False
        if advance:
            self.step += 1
            self.show()
        return True

    # -- step 4: labels --
    def _s_labels(self):
        b = self.body
        self._row(0, "Objects to label", self.object_name, width=30)
        ttk.Label(b, text="e.g. object, vehicle, building -- used to name the trained model's class.",
                  style="Muted.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(b, text="Also sort them into types (classes)", variable=self.use_classes,
                        command=self._sync_classes).grid(row=2, column=0, columnspan=2, sticky="w", pady=(16, 4))
        self._cls_entry = ttk.Entry(b, textvariable=self.classes, width=56)
        self._cls_entry.grid(row=3, column=1, sticky="we")
        ttk.Label(b, text="Comma-separated, up to 9. While reviewing, number keys 1-9 pick a type. You can change "
                          "this later in Settings.", style="Muted.TLabel", wraplength=560).grid(row=4, column=1, sticky="w")
        self._sync_classes()

    def _sync_classes(self):
        self._cls_entry.state(["!disabled"] if self.use_classes.get() else ["disabled"])

    def _class_list(self):
        return [c.strip() for c in self.classes.get().split(",") if c.strip()][:9] if self.use_classes.get() else []

    def _v_labels(self):
        if not self.object_name.get().strip():
            self.msg.config(text="Say what you're labelling (e.g. object).")
            return False
        if self.use_classes.get() and len(self._class_list()) < 2:
            self.msg.config(text="List at least two types, or untick the classes option.")
            return False
        return True

    # -- step 5: start --
    def _s_start(self):
        b = self.body
        sel = self.selected()
        sites = sorted({c["site"] for c in sel})
        n_det = sum(c["n_detections"] for c in sel) if self.has_dets.get() == "yes" else 0
        cls = self._class_list()
        ttk.Label(b, text=(f"{len(sel)} images in {len(sites)} area{'s' if len(sites) != 1 else ''}  ·  "
                           + (f"{n_det} detections to review" if n_det else "no detections") + "  ·  "
                           + (f"types: {', '.join(cls)}" if cls else f"label: {self.object_name.get().strip()}")),
                  style="H2.TLabel", wraplength=780).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        opts = []
        if n_det:
            opts.append(("review", "Review the detector's output",
                         "Go through each detection: is it real? Fix boxes, add anything it missed, and check "
                         "each whole image once."))
        opts.append(("assist", "Label from scratch with a model helping",
                     f"You label a small, varied set first. The app picks {assist.holdout_target(len(sel))} test "
                     f"images and about 20 images to start with. After roughly 150 objects you train a model, "
                     f"and it pre-labels the rest for you to correct."))
        opts.append(("plain", "Label every image myself", "Go image by image and mark every object by hand."))
        if not self.start.get() or self.start.get() not in [o[0] for o in opts]:
            self.start.set("review" if n_det else ("assist" if len(sel) >= 20 else "plain"))
        r = 1
        for key, title, desc in opts:
            rec = "  (recommended)" if (key == "review" and n_det) or (key == "assist" and not n_det and len(sel) >= 20) else ""
            ttk.Radiobutton(b, text=title + rec, value=key, variable=self.start).grid(row=r, column=0, sticky="w", pady=(8, 0))
            ttk.Label(b, text=desc, style="Muted.TLabel", wraplength=720).grid(row=r + 1, column=0, sticky="w", padx=(24, 0))
            r += 2
        ttk.Label(b, text=f"The project will be created in {Path(self.parent.get()) / pj.slugify(self.name.get())}",
                  style="Muted.TLabel", wraplength=780).grid(row=r, column=0, sticky="w", pady=(18, 0))

    def _v_start(self):
        return True

    def _finish(self):
        settings = {"classes": self._class_list(), "object_name": self.object_name.get().strip() or "object"}
        try:
            proj = pj.Project.create(self.parent.get(), self.name.get().strip(), self.desc, settings)
        except (FileExistsError, OSError) as exc:
            self.msg.config(text=str(exc))
            return
        sel = self.selected()
        if self.has_dets.get() != "yes":
            for c in sel:
                c["detections"], c["n_detections"] = [], 0
        proj.add_imagery(sel)
        if self.start.get() == "assist":
            assist.pick_holdout(proj)
            assist.suggest_images(proj, 20)
        config.remember_projects_dir(self.parent.get())
        config.remember_source(self.img_dir.get().strip(),
                               self.det_dir.get().strip() if self.has_dets.get() == "yes" else None)
        self.result = proj
        self.start_mode = self.start.get()
        self.destroy()
