"""Model tab: model-assisted labelling (hold-out, suggestions, train, predict)."""
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..core import assist, backends
from ..core import project as pj
from .common import Collapsible

STEP_STYLE = {"done": ("DONE", "#1a7c4c"), "now": ("NEXT", "#b05a00"), "todo": ("LATER", "#777"),
              "warn": ("NOTE", "#9a6a00")}


class ModelPanel(ttk.Frame):
    """Model-assisted labelling (assist.py): hold-out, suggestions, train, predict, learning curve.
    Training/prediction run as subprocesses (GPU memory and crashes stay out of the GUI); the job
    handle lives on the App so it survives switching screens."""

    def __init__(self, master, app, project_view):
        super().__init__(master, padding=(0, 8, 0, 0))
        self.app = app
        self.pv = project_view
        self.project = app.project
        self.chart_widget = None
        self._poll_job = None
        cfg = assist.settings(self.project)

        self.has_backend = backends.is_available()
        if not self.has_backend:
            note = tk.Frame(self, bg="#fbf3e4", padx=10, pady=6)
            note.pack(fill="x", pady=(0, 6))
            tk.Label(note, text="Training add-on not installed: you can pick a hold-out, get image suggestions and "
                                "recommendations, but Train / Predict need the optional add-on.", bg="#fbf3e4",
                     fg="#1a1a19", justify="left", wraplength=900).pack(side="left")
            ttk.Button(note, text="How to install", command=self._install_help).pack(side="right")
        self.headline = ttk.Label(self, text="", style="H2.TLabel", wraplength=1180, justify="left")
        self.headline.pack(anchor="w")
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=(6, 0))
        left = ttk.Frame(body, width=560)
        left.pack(side="left", fill="y")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.steps_frame = ttk.LabelFrame(left, text="Recommended steps", padding=6)
        self.steps_frame.pack(fill="x")
        self.data_label = ttk.Label(left, text="", style="Muted.TLabel", wraplength=540, justify="left")
        self.data_label.pack(anchor="w", pady=(4, 4))

        self.next_btn = ttk.Button(left, text="", style="Accent.TButton", command=lambda: None)
        self.next_btn.pack(anchor="w", pady=(8, 4), ipady=4)

        act = ttk.Frame(left)
        act.pack(fill="x", pady=(2, 0))
        self.n_hold = tk.IntVar(value=assist.holdout_target(len(self.project.imagery)))
        self.n_sugg = tk.IntVar(value=20)
        ttk.Button(act, text="Pick hold-out", command=self.pick_holdout).pack(side="left")
        ttk.Button(act, text="Suggest images", command=self.suggest).pack(side="left", padx=4)
        self.train_btn = ttk.Button(act, text="Train model", command=self.train)
        self.train_btn.pack(side="left")
        self.predict_btn = ttk.Button(act, text="Predict", command=self.predict)
        self.predict_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(act, text="Stop job", command=self.stop)
        self.stop_btn.pack(side="left")

        job = ttk.Frame(left)
        job.pack(fill="x", pady=(8, 0))
        self.job_label = ttk.Label(job, text="")
        self.job_label.pack(anchor="w")
        self.job_bar = ttk.Progressbar(job, maximum=1, length=520)

        adv = Collapsible(left, "Advanced  (counts, base model, epochs, tiles, job log)")
        adv.pack(fill="both", expand=True, pady=(8, 0))
        cnt = ttk.Frame(adv.body)
        cnt.pack(fill="x", padx=12)
        ttk.Label(cnt, text="Hold-out size").pack(side="left")
        ttk.Spinbox(cnt, from_=3, to=100, textvariable=self.n_hold, width=5).pack(side="left", padx=(4, 14))
        ttk.Label(cnt, text="Images per suggestion").pack(side="left")
        ttk.Spinbox(cnt, from_=1, to=200, textvariable=self.n_sugg, width=5).pack(side="left", padx=4)
        ttk.Button(cnt, text="Review hold-out", command=lambda: self.review_set("holdout")).pack(side="left", padx=(14, 4))
        ttk.Button(cnt, text="Review suggested", command=lambda: self.review_set("suggested")).pack(side="left")
        opt = ttk.Frame(adv.body)
        opt.pack(fill="x", padx=12, pady=(6, 0))
        self.weights = tk.StringVar(value=cfg["base_weights"])
        self.epochs = tk.IntVar(value=cfg["epochs"])
        self.tile = tk.IntVar(value=cfg["tile_px"])
        self.upscale = tk.StringVar(value=str(cfg["upscale"]))
        ttk.Label(opt, text="Base weights").grid(row=0, column=0, sticky="w")
        # blank = the backend's own default (e.g. yolov8n.pt for the Ultralytics add-on)
        ttk.Entry(opt, textvariable=self.weights, width=40).grid(row=0, column=1, columnspan=4, sticky="we", padx=4)
        ttk.Button(opt, text="Browse…", command=self._browse_weights).grid(row=0, column=5)
        ttk.Label(opt, text="Epochs").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Spinbox(opt, from_=5, to=300, increment=5, textvariable=self.epochs, width=6).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Label(opt, text="Tile px").grid(row=1, column=2, sticky="e", pady=(4, 0))
        ttk.Spinbox(opt, from_=128, to=1024, increment=32, textvariable=self.tile, width=6).grid(row=1, column=3, sticky="w", pady=(4, 0))
        ttk.Label(opt, text="Upscale").grid(row=1, column=4, sticky="e", pady=(4, 0))
        ttk.Combobox(opt, textvariable=self.upscale, values=["auto", "1", "2", "3", "4"], width=5,
                     state="readonly").grid(row=1, column=5, sticky="w", pady=(4, 0))
        ttk.Label(opt, text="Leave blank for the add-on's default base model; a domain-pretrained model needs far fewer labels. "
                            "Upscale auto enlarges tiles so small objects reach ~16 px.", style="Muted.TLabel",
                  wraplength=520).grid(row=2, column=0, columnspan=6, sticky="w", pady=(4, 0))
        self.log = tk.Text(adv.body, height=7, width=70, font=("TkFixedFont", 8), bg="#f7f6f2", relief="flat",
                           state="disabled")
        self.log.pack(fill="both", expand=True, padx=12, pady=(6, 0))

        ttk.Label(right, text="Model versions (scored on the blind hold-out)", style="H2.TLabel").pack(anchor="w")
        cols = (("v", 36, "v"), ("when", 110, "Trained"), ("objs", 70, "Objects"), ("imgs", 60, "Images"),
                ("p", 50, "P"), ("r", 50, "R"), ("f1", 50, "F1"), ("thr", 55, "Score>="), ("cr", 60, "Count r"),
                ("status", 90, "Status"))
        self.vtree = ttk.Treeview(right, columns=[c for c, _, _ in cols], show="headings", height=5)
        for c, w, t in cols:
            self.vtree.heading(c, text=t)
            self.vtree.column(c, width=w, anchor="center")
        self.vtree.pack(fill="x")
        self.chart_frame = ttk.Frame(right)
        self.chart_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.refresh()
        self._poll()
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_destroy(self, event):
        """Stop the progress timer when this tab goes away (e.g. switching to the review screen);
        the job itself keeps running and the next Model tab picks it up from app.assist_job."""
        if event.widget is self and self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None

    # -- helpers --
    def _save_options(self):
        try:
            self.project.settings["assist"] = {
                **self.project.settings.get("assist", {}), "base_weights": self.weights.get().strip(),
                "epochs": int(self.epochs.get()), "tile_px": int(self.tile.get()), "upscale": self.upscale.get()}
        except (ValueError, tk.TclError):
            messagebox.showerror("Training options", "Epochs and tile size must be whole numbers.")
            return False
        self.project.save_meta()
        return True

    def _browse_weights(self):
        f = filedialog.askopenfilename(title="Base weights (.pt)", filetypes=[("PyTorch weights", "*.pt")],
                                       initialdir=str(Path.home()))
        if f:
            self.weights.set(f)

    def _busy(self):
        job = getattr(self.app, "assist_job", None)
        return job is not None and job["proc"].poll() is None

    # -- refresh --
    def refresh(self):
        inv = assist.inventory(self.project)
        adv = assist.recommend(self.project, inv)
        self.headline.config(text=adv["headline"])
        for w in self.steps_frame.winfo_children():
            w.destroy()
        for i, stp in enumerate(adv["steps"]):
            word, col = STEP_STYLE.get(stp["status"], ("", "#666"))
            tk.Label(self.steps_frame, text=word, fg=col, bg=self.steps_frame.winfo_toplevel().cget("bg"),
                     font=("TkDefaultFont", 9, "bold"), width=6, anchor="w").grid(row=i, column=0, sticky="nw")
            ttk.Label(self.steps_frame, text=stp["name"], font=("TkDefaultFont", 9, "bold")).grid(
                row=i, column=1, sticky="nw", padx=(0, 6))
            ttk.Label(self.steps_frame, text=stp["detail"], wraplength=400, justify="left").grid(
                row=i, column=2, sticky="w", pady=1)
        cls = ", ".join(f"{c} {n}" for c, n in sorted(inv["classes"].items(), key=lambda kv: -kv[1])) or "none yet"
        self.data_label.config(text=(
            f"{inv['n_images']} images  ·  training: {inv['train_boxes']} objects in {len(inv['train_images'])} complete "
            f"images ({inv['train_empty']} empty)  ·  hold-out: {len(inv['holdout_complete'])}/{len(inv['holdout'])} "
            f"complete, {inv['holdout_boxes']} objects  ·  untouched: {len(inv['untouched'])}  ·  median object "
            + (f"{inv['median_box_px']:.1f} px" if inv["median_box_px"] else "-") + f"  ·  classes: {cls}"))
        nv = assist.next_version(self.project)
        self.train_btn.config(text=f"Train model v{nv}")
        text, cmd = self._next_action(inv, adv, nv)
        self.next_btn.config(text=text + "  ▶", command=cmd)
        self.vtree.delete(*self.vtree.get_children())
        for rec in inv["versions"]:
            m = (rec.get("metrics") or {}).get("best") or {}
            cr = (rec.get("metrics") or {}).get("count_r")
            f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
            pred = f", {rec['predicted']['images']} pre-labelled" if rec.get("predicted") else ""
            self.vtree.insert("", "end", values=(rec["version"], (rec.get("finished") or rec["created"])[:16].replace("T", " "),
                                                 rec.get("n_train_boxes", "-"), rec.get("n_train_images", "-"),
                                                 f(m.get("p")), f(m.get("r")), f(m.get("f1")), f(m.get("thr")), f(cr),
                                                 rec.get("status", "") + pred))
        self._chart([r for r in inv["versions"] if r.get("status") == "done" and r.get("metrics")])
        self._update_buttons()

    def _next_action(self, inv, adv, nv):
        """The one thing to do next, as (button text, command), from the recommendation."""
        st = assist.load_assist(self.project)
        prog = self.project.image_progress()
        if self._busy():
            return "A job is running -- see progress below", lambda: None
        if not inv["holdout"]:
            return "Pick hold-out (test) images", self.pick_holdout
        if len(inv["holdout_complete"]) < len(inv["holdout"]):
            return f"Label the hold-out images blind ({len(inv['holdout_complete'])}/{len(inv['holdout'])} done)", \
                lambda: self.review_set("holdout")
        versions = [r for r in inv["versions"] if r.get("status") == "done"]
        train_now = any(s_["status"] == "now" and s_["name"] in ("First model", "Next model") for s_ in adv["steps"])
        if train_now and not self.has_backend:
            return "Install the training add-on to train a model", self._install_help
        if train_now:
            return f"Train model v{nv}", self.train
        if versions:
            last = versions[-1]
            m = (last.get("metrics") or {}).get("best") or {}
            useful = m.get("r", 0) >= assist.USEFUL_RECALL and m.get("p", 0) >= assist.USEFUL_PRECISION
            if useful and inv["untouched"] and not last.get("predicted"):
                return f"Pre-label {len(inv['untouched'])} images with model v{last['version']}", self.predict
        todo = [k for k in st["suggested"] if k in prog and not prog[k]["complete"]]
        if todo:
            return f"Label the suggested images ({len(st['suggested']) - len(todo)}/{len(st['suggested'])} done)", \
                lambda: self.review_set("suggested")
        if inv["untouched"]:
            return "Suggest the next images to label", self.suggest
        return "Review remaining images", self.app.start_review

    def _install_help(self):
        messagebox.showinfo("Training add-on", backends.INSTALL_HINT)

    def _update_buttons(self):
        busy = self._busy()
        done = [r for r in assist.load_assist(self.project)["versions"] if r.get("status") == "done"]
        self.train_btn.state(["disabled"] if busy or not self.has_backend else ["!disabled"])
        self.next_btn.state(["disabled"] if busy else ["!disabled"])
        self.predict_btn.state(["disabled"] if busy or not done or not self.has_backend else ["!disabled"])
        self.stop_btn.state(["!disabled"] if busy else ["disabled"])

    # -- actions --
    def pick_holdout(self):
        if self._busy():
            return
        st = assist.load_assist(self.project)
        n = int(self.n_hold.get())
        if st["holdout"] and not messagebox.askyesno(
                "Hold-out", f"Top up the hold-out to {n} images? Existing hold-out images are kept (changing the "
                            f"hold-out makes older model scores incomparable)."):
            return
        keys = assist.pick_holdout(self.project, n)
        self.pv.reload_rows()
        self.refresh()
        messagebox.showinfo("Hold-out", f"{len(keys)} hold-out images. Review them with 'Review hold-out (blind)': "
                                        f"no detections are shown on them -- draw every object and finish each "
                                        f"scene check.")

    def suggest(self):
        keys = assist.suggest_images(self.project, int(self.n_sugg.get()))
        self.pv.reload_rows()
        self.refresh()
        if not keys:
            messagebox.showinfo("Suggest images", "No untouched images left to suggest.")

    def review_set(self, which):
        st = assist.load_assist(self.project)
        keys = st["holdout"] if which == "holdout" else st["suggested"]
        prog = self.project.image_progress()
        todo = [k for k in keys if k in prog and not prog[k]["complete"]]
        if not keys:
            messagebox.showinfo("Nothing to review", "Pick a hold-out / suggest images first.")
            return
        if not todo:
            messagebox.showinfo("All done", f"Every {'hold-out' if which == 'holdout' else 'suggested'} image is complete.")
            return
        self.app.start_review(image_key=todo[0], order="holdout" if which == "holdout" else "suggested")

    def _start(self, cmd, v):
        vdir = assist.version_dir(self.project, v)
        vdir.mkdir(parents=True, exist_ok=True)
        out = open(vdir / f"{cmd}_stdout.txt", "w")
        proc = subprocess.Popen([sys.executable, "-m", "boxtrot.core.assist", cmd, str(self.project.root), str(v)],
                                stdout=out, stderr=subprocess.STDOUT, cwd=str(vdir))
        self.app.assist_job = {"proc": proc, "cmd": cmd, "v": v, "root": str(self.project.root)}
        self._update_buttons()
        self._poll()

    def train(self):
        if not self.has_backend:
            self._install_help()
            return
        if self._busy() or not self._save_options():
            return
        inv = assist.inventory(self.project)
        adv = assist.recommend(self.project, inv)
        if not adv["ready_to_train"]:
            messagebox.showinfo("Not enough labels", "Label at least a couple of images completely (with at least "
                                                     "10 objects) before training.")
            return
        warn = []
        if inv["train_boxes"] < adv["need_boxes"]:
            warn.append(f"only {inv['train_boxes']} of the recommended {adv['need_boxes']} objects are labelled")
        if not inv["holdout_complete"]:
            warn.append("there's no complete hold-out, so the model can't be scored")
        if warn and not messagebox.askyesno("Train anyway?", "Training now is possible, but " + " and ".join(warn) +
                                            ".\n\nTrain anyway?"):
            return
        self._start("train", assist.next_version(self.project))

    def predict(self):
        if not self.has_backend:
            self._install_help()
            return
        if self._busy():
            return
        done = [r for r in assist.load_assist(self.project)["versions"] if r.get("status") == "done"]
        if not done:
            return
        rec = done[-1]
        inv = assist.inventory(self.project)
        m = (rec.get("metrics") or {}).get("best") or {}
        if not inv["untouched"]:
            messagebox.showinfo("Predict", "No untouched images to pre-label (predictions never overwrite images "
                                           "someone has reviewed).")
            return
        q = (f"Pre-label {len(inv['untouched'])} untouched images with model v{rec['version']}"
             + (f" (hold-out R {m.get('r', 0):.2f}, P {m.get('p', 0):.2f})" if m else " (not scored on a hold-out)")
             + "?\n\nReviewed images and the hold-out are never touched.")
        if messagebox.askyesno("Predict", q):
            self._start("predict", rec["version"])

    def stop(self):
        job = getattr(self.app, "assist_job", None)
        if job and job["proc"].poll() is None and messagebox.askyesno("Stop", f"Stop the running {job['cmd']} job?"):
            job["proc"].terminate()
            assist.write_status(self.project, job["v"], stage="failed", message="stopped by user")

    def _poll(self):
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
        job = getattr(self.app, "assist_job", None)
        if job is None or job["root"] != str(self.project.root):
            return
        st = assist.read_status(self.project, job["v"])
        msg = st.get("message", "")
        if not self.job_bar.winfo_ismapped():
            self.job_bar.pack(fill="x", pady=4)
        if st.get("epochs"):
            self.job_bar.config(maximum=st["epochs"], value=st.get("epoch", 0))
        elif st.get("total"):
            self.job_bar.config(maximum=st["total"], value=st.get("done", 0))
        stalled = ""
        try:
            idle = (datetime.now().astimezone() - datetime.fromisoformat(st["at"])).total_seconds()
            if idle > 600 and job["proc"].poll() is None:
                stalled = f"  --  NO PROGRESS for {idle / 60:.0f} min: it may be stuck (see output below; Stop if so)"
        except (KeyError, ValueError):
            pass
        self.job_label.config(text=f"{job['cmd']} v{job['v']}: {st.get('stage', 'starting')} -- {msg}{stalled}",
                              foreground="#c7403a" if stalled else "")
        vdir = assist.version_dir(self.project, job["v"])
        try:
            lines = (vdir / "log.txt").read_text().splitlines()[-8:]
            if stalled:   # show the raw output: errors from worker processes only appear there
                lines = (vdir / f"{job['cmd']}_stdout.txt").read_text(errors="replace").splitlines()[-8:]
        except OSError:
            lines = []
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", "\n".join(lines))
        self.log.config(state="disabled")
        if job["proc"].poll() is None:
            self._poll_job = self.after(1000, self._poll)
            return
        # finished
        self.app.assist_job = None
        if job["cmd"] == "predict":   # the subprocess attached detection files: reload the project from disk
            self.app.project = pj.Project(job["root"])
            self.app.show_project(tab="Model")
            return
        self.refresh()

    # -- learning curve --
    def _chart(self, versions):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        if self.chart_widget is not None:
            self.chart_widget.get_tk_widget().destroy()
        fig = Figure(figsize=(6.2, 3.8), dpi=100, facecolor="#fcfcfb")
        ax = fig.add_subplot(111, facecolor="#fcfcfb")
        ink, muted = "#1a1a19", "#6b6a63"
        ax.set_title("Learning curve: hold-out score vs objects labelled", loc="left", fontsize=11, color=ink, pad=20)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("objects in training set", fontsize=9, color=muted)
        ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c0")
        ax.tick_params(colors=muted, labelsize=8)
        ax.axhline(assist.USEFUL_RECALL, color="#9a998f", linewidth=1, dash_capstyle="round", linestyle=(0, (4, 3)))
        ax.annotate(f"pre-labels useful (R >= {assist.USEFUL_RECALL})", (0, assist.USEFUL_RECALL), xycoords=("axes fraction", "data"),
                    xytext=(2, 3), textcoords="offset points", fontsize=8, color=muted)
        pts = []
        if versions:
            xs = [v.get("n_train_boxes", 0) for v in versions]
            series = (("Recall", "r", "#2a78d6"), ("Precision", "p", "#eb6834"))
            for name, key, col in series:
                ys = [((v.get("metrics") or {}).get("best") or {}).get(key, 0) for v in versions]
                ax.plot(xs, ys, color=col, linewidth=2, label=name, zorder=2)
                ax.scatter(xs, ys, s=42, color=col, edgecolors="#fcfcfb", linewidths=2, zorder=3)
                ax.annotate(name, (xs[-1], ys[-1]), xytext=(8, 6 if key == "r" else -10), textcoords="offset points",
                            fontsize=8, color=ink)
                pts += [(x, y, name, v["version"]) for x, y, v in zip(xs, ys, versions)]
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False, labelcolor=ink)
            ax.set_xlim(0, max(xs) * 1.25 + 1)
        else:
            ax.text(0.5, 0.5, "No scored models yet -- train one to start the curve", transform=ax.transAxes,
                    ha="center", color=muted, fontsize=9)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        tip = ax.annotate("", (0, 0), xytext=(10, 10), textcoords="offset points", fontsize=8, color=ink,
                          bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#c9c8c0"), zorder=5)
        tip.set_visible(False)

        def on_move(event):
            best = None
            if event.inaxes is ax:
                for x, y, name, v in pts:
                    px, py = ax.transData.transform((x, y))
                    d = (px - event.x) ** 2 + (py - event.y) ** 2
                    if d < 144 and (best is None or d < best[0]):
                        best = (d, x, y, name, v)
            if best:
                _, x, y, name, v = best
                tip.xy = (x, y)
                tip.set_text(f"v{v}: {name} {y:.2f}\n{x} objects labelled")
            if bool(best) != tip.get_visible() or best:
                tip.set_visible(bool(best))
                canvas.draw_idle()
        canvas.mpl_connect("motion_notify_event", on_move)
        canvas.mpl_connect("resize_event", lambda e: (fig.tight_layout(), canvas.draw_idle()))
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_widget = canvas
