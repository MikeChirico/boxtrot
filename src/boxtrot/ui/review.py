"""Review screen and the side-by-side date comparison window."""
import json
import math
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageEnhance, ImageTk
from pyproj import Transformer

from ..core import assist
from ..core import project as pj
from .common import INSTRUCTIONS, RIGHT, SHORTCUTS, STATUS_COLOR, UNDO_KEYS, Collapsible, menu_button


class CompareWindow(tk.Toplevel):
    """Side-by-side view (w): the same footprint on another date of the same port, kept in sync
    with the main canvas. A pier/breakwater is there on every date; a object usually isn't."""

    def __init__(self, review):
        super().__init__(review.app.root)
        self.review = review
        self.title("Compare dates")
        self.size = 520
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Label(bar, text="Other date:").pack(side="left")
        self.choice = tk.StringVar()
        self.cb = ttk.Combobox(bar, textvariable=self.choice, width=34, state="readonly")
        self.cb.pack(side="left", padx=6)
        self.cb.bind("<<ComboboxSelected>>", lambda e: review._set_compare_key(self.cb_keys[self.cb.current()]))
        self.canvas = tk.Canvas(self, width=self.size, height=self.size, bg="black", highlightthickness=0)
        self.canvas.pack(padx=6, pady=(0, 6))
        self.caption = ttk.Label(self, text="", style="Muted.TLabel")
        self.caption.pack(pady=(0, 6))
        self.cb_keys = []
        self.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        self.review.compare_win = None
        self.destroy()

    def render(self, others, key, path, lon, lat, radius, pan, boxes):
        self.cb_keys = [k for k, _ in others]
        self.cb.config(values=[label for _, label in others])
        if key in self.cb_keys:
            self.cb.current(self.cb_keys.index(key))
        self.canvas.delete("all")
        v = pj.render_view(path, lon, lat, radius, offset_m=pan, rgb_bands=self.review.rgb,
                           max_px=self.size * 2, stretch_mode=self.review.stretch_mode,
                           view_mode=self.review.view_mode) if path else None
        if v is None:
            self.canvas.create_text(self.size // 2, self.size // 2, text="no other date of this area in the project",
                                    fill="white")
            self.caption.config(text="")
            return
        scale = self.size / max(v["img"].size)
        img = self.review._enhance(v["img"])
        self.tk_img = ImageTk.PhotoImage(img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                                                    Image.LANCZOS))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        to_v = Transformer.from_crs("EPSG:4326", v["crs"], always_xy=True)
        for geom, color in boxes:   # the current image's boxes, at the same place on the other date
            pts = [~v["transform"] * to_v.transform(a, b) for a, b in geom["coordinates"][0][:4]]
            xs = [p[0] * scale for p in pts]
            ys = [p[1] * scale for p in pts]
            self.canvas.create_rectangle(min(xs), min(ys), max(xs), max(ys), outline=color, width=2, dash=(4, 2))
        self.caption.config(text="dashed = boxes from the image under review, drawn at the same place")


class ReviewView(ttk.Frame):
    """The review screen: decide each detection, fix boxes, flag missed objects, scene checks; plus click-to-jump,
    resize handles, full undo, confidence ordering/filtering, bulk reject, object classes,
    band views, date comparison, brightness/contrast and a minimap."""
    PAN_LIMIT_M = 6000
    HANDLE = 5   # half-size of a resize handle, canvas px

    def __init__(self, master, app, queue, image_key=None, mode="detections", order=None):
        super().__init__(master, padding=6)
        self.app = app
        self.project = app.project
        self.state = self.project.state
        self.additions = self.project.additions
        self.canvas_size = int(self.project.settings.get("canvas_size", 680))
        self.default_radius = float(self.project.settings.get("crop_radius_m", pj.CROP_RADIUS_M))
        self.rgb = self.project.settings.get("rgb_bands", "auto")
        self.stretch_mode = self.project.settings.get("stretch", "image")
        self.classes = self.project.classes[:9]
        self.bindings = []

        # scene checks holding an old Edit Box (not linked to any detection) go first for a recheck
        self.recheck_ids = [it["id"] for it in queue
                            if it["kind"] == "scene" and self.state.get(it["id"], {}).get("corrected_box")]
        if self.recheck_ids:
            front = [it for it in queue if it["id"] in self.recheck_ids]
            queue = front + [it for it in queue if it["id"] not in self.recheck_ids]
        self.base_queue = queue          # every item, image order
        self.queue = list(queue)         # what navigation walks: base_queue ordered/filtered
        self.image_order = []
        for it in self.base_queue:
            if it["image_key"] not in self.image_order:
                self.image_order.append(it["image_key"])
        self.entries = {pj.image_key(e): e for e in self.project.imagery}
        self.pass_to_image = {(e["site"], e["item_id"]): k for k, e in self.entries.items()}
        for it in self.base_queue:
            self.pass_to_image.setdefault((it["site"], it["item_id"]), it["image_key"])
        self.sources = sorted({it.get("source") or "detector" for it in self.base_queue if it["kind"] != "scene"})
        ast = assist.load_assist(self.project)
        self.holdout = set(ast["holdout"])
        self.suggested = [k for k in ast["suggested"] if k in self.entries]

        self.index = 0
        self.view = None
        self.to_view = None
        self.scale = 1.0
        self.crop_radius_m = self.default_radius
        self.pan_m = [0.0, 0.0]
        self.view_key = None
        self.pan_start = None
        self.add_mode = False
        self.edit_mode = False
        self.rect_id = None
        self.rect_start = None
        self.mode = mode
        self.addition_index = 0
        self.undo_stack = []         # every mutation this session, newest last (Ctrl+Z)
        self.hit_boxes = []          # (x0, y0, x1, y1, kind, ref) for click-to-jump
        self.cur_box = None          # canvas coords of the current detection's box (resize handles)
        self.handle_drag = None
        self.active_class = None     # class given to boxes you add (number keys)
        self.view_mode = "true"
        self._views_cache = {}
        self.compare_on = False      # t: flick the main canvas to another date
        self.compare_key = None      # chosen other image (None = nearest date)
        self.compare_win = None
        self._thumb_cache = {}
        self._redraw_job = None

        self._build_ui()
        self._bind_keys()
        if order in ("suggested", "holdout"):
            self.order_var.set("Suggested first" if order == "suggested" else "Hold-out first")
            self._apply_order(show=False)
        if mode == "additions":
            self.review_additions_btn.config(text="Back to Detections (r)")
            self.show_current()
        elif image_key:
            self.jump_to_image(image_key)
        else:
            self.jump_to_first_unreviewed()

    # -- ui --------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=(4, 2))
        top.pack(fill="x")
        ttk.Button(top, text="◀ Project (Esc)", command=self.back).pack(side="left")
        self.info_label = ttk.Label(top, text="", font=("TkDefaultFont", 11))
        self.info_label.pack(side="left", padx=12)
        self.progress_label = ttk.Label(top, text="", font=("TkDefaultFont", 11))
        self.progress_label.pack(side="right")

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True)
        left = ttk.Frame(mid)
        left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg="black",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind(f"<ButtonPress-{RIGHT}>", self._pan_press)
        self.canvas.bind(f"<B{RIGHT}-Motion>", self._pan_drag)
        self.canvas.bind(f"<ButtonRelease-{RIGHT}>", self._pan_release)
        self.canvas.bind("<Button-4>", lambda e: self.zoom_in())
        self.canvas.bind("<Button-5>", lambda e: self.zoom_out())
        self.canvas.bind("<MouseWheel>", lambda e: self.zoom_in() if e.delta > 0 else self.zoom_out())
        self.canvas.bind("<Motion>", self._on_motion)
        self.scene_label = ttk.Label(left, text="", font=("TkDefaultFont", 10, "bold"),
                                     wraplength=self.canvas_size, justify="left")
        self.scene_label.pack(anchor="w", padx=4)

        side = ttk.Frame(mid, padding=(8, 0, 0, 0))
        side.pack(side="left", fill="both", expand=True)

        def grid_btn(parent, text, cmd, r, c, span=1):
            b = ttk.Button(parent, text=text, command=cmd)
            b.grid(row=r, column=c, columnspan=span, sticky="we", padx=2, pady=2)
            return b

        acts = ttk.LabelFrame(side, text="Decide", padding=4)
        acts.pack(fill="x")
        grid_btn(acts, "Valid (v)", self.mark_valid, 0, 0)
        grid_btn(acts, "Invalid (x)", self.mark_invalid, 0, 1)
        grid_btn(acts, "Skip (s)", self.mark_skip, 0, 2)
        self.add_btn = grid_btn(acts, "Flag missed object (a)", self.toggle_add_mode, 1, 0)
        grid_btn(acts, "Undo (Ctrl+Z)", self.undo, 1, 1)
        menu_button(acts, "Tools…", [
            ("Redraw box  (e)", self.toggle_edit_mode), ("Reset box to the detector's", self.reset_box),
            None,
            ("Reject the rest of this image  (X)", self.reject_rest_of_image),
            ("Delete the shown added box  (d)", self.delete_current_addition),
            None,
            ("Review boxes I added / back  (r)", self.toggle_review_mode),
            ("Next unfinished scene check  (n)", lambda: self.jump_to_next_unfinished("scene")),
            None,
            ("Zoom in  (+)", self.zoom_in), ("Zoom out  (-)", self.zoom_out), ("Re-centre  (c)", self.recenter),
            None,
            ("Keyboard shortcuts", lambda: messagebox.showinfo("Keyboard shortcuts", SHORTCUTS)),
        ]).grid(row=1, column=2, sticky="we", padx=2, pady=2)
        for i in range(3):
            acts.columnconfigure(i, weight=1, uniform="a")
        # state-only stand-ins (their actions live in the Tools menu / keys); kept so mode code can set them
        self.edit_btn = ttk.Button(acts)
        self.review_additions_btn = ttk.Button(acts)

        if self.classes:
            cls = ttk.LabelFrame(side, text="Type (keys 1-9, 0 = none)", padding=4)
            cls.pack(fill="x", pady=(4, 0))
            names = [f"{i + 1} {c}" for i, c in enumerate(self.classes)] + ["0 none"]
            ncol = 3
            for i, n in enumerate(names):
                grid_btn(cls, n, (lambda k=(i + 1 if i < len(self.classes) else 0): self.class_key(k)),
                         i // ncol, i % ncol)
            for i in range(ncol):
                cls.columnconfigure(i, weight=1, uniform="c")

        nav = ttk.Frame(side, padding=(0, 4, 0, 0))
        nav.pack(fill="x")
        grid_btn(nav, "◀ Previous", self.go_previous, 0, 0)
        grid_btn(nav, "Next ▶", self.go_next, 0, 1)
        grid_btn(nav, "Next undecided (u)", lambda: self.jump_to_next_unfinished("detection"), 0, 2)
        self.zoom_label = ttk.Label(nav, text="", style="Muted.TLabel")
        self.zoom_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=2)
        for i in range(3):
            nav.columnconfigure(i, weight=1, uniform="n")

        ui = self.project.settings.get("ui", {})
        self.view_sec = Collapsible(side, "View options  (bands, other dates, contrast, order, overview map)",
                                    open_=ui.get("view_open", False), on_toggle=self._remember_view_open)
        self.view_sec.pack(fill="x", pady=(4, 0))
        view = ttk.Frame(self.view_sec.body, padding=(4, 0, 4, 4))
        view.pack(fill="x")
        vleft = ttk.Frame(view)
        vleft.pack(side="left", fill="both", expand=True)
        r1 = ttk.Frame(vleft)
        r1.pack(fill="x")
        ttk.Button(r1, text="Bands (b)", command=self.cycle_view_mode).pack(side="left", padx=2)
        self.compare_btn = ttk.Button(r1, text="Other date (t)", command=self.toggle_compare)
        self.compare_btn.pack(side="left", padx=2)
        ttk.Button(r1, text="Side by side (w)", command=self.open_compare_window).pack(side="left", padx=2)
        self.view_mode_label = ttk.Label(vleft, text="", style="Muted.TLabel", wraplength=330)
        self.view_mode_label.pack(anchor="w", padx=2, pady=(2, 0))
        r2 = ttk.Frame(vleft)
        r2.pack(fill="x", pady=(2, 0))
        self.brightness = tk.DoubleVar(value=1.0)
        self.contrast = tk.DoubleVar(value=1.0)
        ttk.Label(r2, text="Bright").pack(side="left")
        ttk.Scale(r2, from_=0.4, to=2.5, variable=self.brightness, length=90,
                  command=lambda v: self._schedule_redraw()).pack(side="left", padx=2)
        ttk.Label(r2, text="Contrast").pack(side="left", padx=(6, 0))
        ttk.Scale(r2, from_=0.4, to=2.5, variable=self.contrast, length=90,
                  command=lambda v: self._schedule_redraw()).pack(side="left", padx=2)
        ttk.Button(r2, text="Reset", width=6, command=self._reset_enhance).pack(side="left", padx=2)
        r3 = ttk.Frame(vleft)
        r3.pack(fill="x", pady=(4, 0))
        ttk.Label(r3, text="Order").pack(side="left")
        self.order_var = tk.StringVar(value="Image order")
        cb = ttk.Combobox(r3, textvariable=self.order_var, width=17, state="readonly",
                          values=["Image order", "Lowest score first", "Highest score first", "Suggested first",
                                  "Hold-out first"])
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_order())
        ttk.Label(r3, text="Source").pack(side="left", padx=(6, 0))
        self.source_var = tk.StringVar(value="All sources")
        cb2 = ttk.Combobox(r3, textvariable=self.source_var, width=11, state="readonly",
                           values=["All sources"] + self.sources)
        cb2.pack(side="left", padx=2)
        cb2.bind("<<ComboboxSelected>>", lambda e: self._apply_order())
        self.minimap = tk.Canvas(view, width=150, height=150, bg="#222", highlightthickness=0, cursor="hand2")
        self.minimap.pack(side="right", padx=(6, 0))
        self.minimap.bind("<ButtonRelease-1>", self._on_minimap_click)

        self.status_label = ttk.Label(side, text="", foreground="#666", wraplength=520)
        self.status_label.pack(anchor="w", pady=(4, 2))

        tabs = ttk.Notebook(side)
        tabs.pack(fill="both", expand=True)
        tf = ttk.Frame(tabs)
        tabs.add(tf, text="Images (click to jump)")
        self.img_tree = ttk.Treeview(tf, columns=("img", "prog", "added", "scene"), show="headings",
                                     selectmode="browse", height=5)
        for c, w, t in (("img", 200, "Image"), ("prog", 70, "Decided"), ("added", 60, "Added"),
                        ("scene", 90, "Scene check")):
            self.img_tree.heading(c, text=t)
            self.img_tree.column(c, width=w, anchor="w" if c == "img" else "center")
        self.img_tree.tag_configure("complete", foreground="#1a7c4c")
        self.img_tree.tag_configure("unreadable", foreground="#999")
        sb = ttk.Scrollbar(tf, orient="vertical", command=self.img_tree.yview)
        self.img_tree.configure(yscrollcommand=sb.set)
        self.img_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.img_tree.bind("<ButtonRelease-1>", self._on_image_click)
        for k in self.image_order:
            e = self.entries.get(k, {})
            tag = "  [H]" if k in self.holdout else "  [S]" if k in self.suggested else ""   # see Help
            self.img_tree.insert("", "end", iid=k, values=(f"{e.get('site', k.split('|')[0])}  {e.get('date') or ''}{tag}",
                                                           "", "", ""))
        self._refresh_image_rows()
        hf = ttk.Frame(tabs)
        tabs.add(hf, text="Help")
        self.instr_label = tk.Label(hf, text="", justify="left", anchor="nw", bg="#f3f2ee", fg="#222",
                                    padx=8, pady=6, font=("TkDefaultFont", 9), wraplength=520)
        self.instr_label.pack(fill="both", expand=True)
        self.instr_label.bind("<Configure>", lambda e: self.instr_label.config(wraplength=max(200, e.width - 16)))

    def _bind_keys(self):
        root = self.app.root
        keys = {
            "v": self.mark_valid, "x": self.mark_invalid, "s": self.mark_skip, "a": self.toggle_add_mode,
            "e": self.toggle_edit_mode, "r": self.toggle_review_mode, "d": self.delete_current_addition,
            "<Left>": self.go_previous, "<Right>": self.go_next, "-": self.zoom_out, "=": self.zoom_in,
            "+": self.zoom_in, "c": self.recenter, "<Escape>": self.back, UNDO_KEYS[0]: self.undo,
            "u": lambda: self.jump_to_next_unfinished("detection"),
            "n": lambda: self.jump_to_next_unfinished("scene"),
            "X": self.reject_rest_of_image, "b": self.cycle_view_mode, "t": self.toggle_compare,
            "w": self.open_compare_window,
        }
        for k in range(10):
            keys[str(k)] = (lambda k=k: self.class_key(k))
        for seq, fn in keys.items():
            root.bind(seq, lambda e, fn=fn: fn())
            self.bindings.append(seq)
        for seq, d in (("<Shift-Left>", (-1, 0)), ("<Shift-Right>", (1, 0)),
                       ("<Shift-Up>", (0, 1)), ("<Shift-Down>", (0, -1))):
            root.bind(seq, lambda e, d=d: self.pan_step(*d))
            self.bindings.append(seq)

    def _remember_view_open(self, is_open):
        self.project.settings.setdefault("ui", {})["view_open"] = is_open
        self.project.save_meta()

    def deactivate(self):
        for seq in self.bindings:
            self.app.root.unbind(seq)
        self.bindings = []
        if self.compare_win is not None:
            self.compare_win.close()

    def back(self):
        self.app.show_project()

    # -- queue order / filter (confidence) --------------------------------------
    def _apply_order(self, show=True):
        cur = self.queue[self.index]["id"] if self.queue and self.mode == "detections" else None
        items = list(self.base_queue)
        src = self.source_var.get()
        if src != "All sources":
            items = [it for it in items if it["kind"] == "scene" or (it.get("source") or "detector") == src]
        order = self.order_var.get()
        if order in ("Suggested first", "Hold-out first"):
            first = self.suggested if order == "Suggested first" else [k for k in self.image_order if k in self.holdout]
            rank = {k: i for i, k in enumerate(first)}
            items.sort(key=lambda it: rank.get(it["image_key"], len(rank)))   # stable: image order within
        elif order != "Image order":
            dets = [it for it in items if it["kind"] != "scene"]
            scenes = [it for it in items if it["kind"] == "scene"]
            sign = 1 if order.startswith("Lowest") else -1
            # unscored detections go last either way
            dets.sort(key=lambda it: (it.get("score") is None, sign * (it.get("score") or 0)))
            items = dets + scenes
        self.queue = items or list(self.base_queue)
        ids = [it["id"] for it in self.queue]
        self.index = ids.index(cur) if cur in ids else 0
        if order != "Image order" and cur not in ids:
            self.index = 0
        self.view_key = None
        self.status_label.config(text=f"queue: {order.lower()}, {src.lower()} -- {len(self.queue)} items",
                                 foreground="#666")
        if show:
            self.show_current()
            self.canvas.focus_set()

    def _queue_index_of(self, item_id):
        ids = [it["id"] for it in self.queue]
        if item_id not in ids:   # filtered out: drop the filter rather than refuse to show it
            self.source_var.set("All sources")
            self.order_var.set("Image order")
            self.queue = list(self.base_queue)
            ids = [it["id"] for it in self.queue]
        return ids.index(item_id) if item_id in ids else None

    # -- image list --------------------------------------------------------
    def _refresh_image_rows(self, keys=None):
        prog = self.project.image_progress(self.base_queue)
        for k in keys or self.image_order:
            p = prog.get(k)
            if p is None or not self.img_tree.exists(k):
                continue
            decided = p["valid"] + p["invalid"] + p["skipped"]
            scene = {"valid": "done", "invalid": "unreadable", None: "pending"}.get(p["scene"], "") \
                if p["has_scene"] else "-"
            tag = "complete" if p["complete"] else "unreadable" if p["unreadable"] else ""
            vals = list(self.img_tree.item(k, "values"))
            self.img_tree.item(k, values=(vals[0], f"{decided}/{p['n_det']}", p["added"], scene),
                               tags=(tag,) if tag else ())

    def _on_image_click(self, event):
        k = self.img_tree.identify_row(event.y)
        if k:
            if self.mode != "detections":
                self.toggle_review_mode()
            self.jump_to_image(k)
        self.canvas.focus_set()   # keep arrow keys away from the tree

    def jump_to_image(self, key):
        idx = [i for i, it in enumerate(self.queue) if it["image_key"] == key]
        if not idx:
            self._queue_index_of(next((it["id"] for it in self.base_queue if it["image_key"] == key), None))
            idx = [i for i, it in enumerate(self.queue) if it["image_key"] == key]
            if not idx:
                return
        for i in idx:
            if not self.state.get(self.queue[i]["id"], {}).get("status"):
                self.index = i
                break
        else:
            self.index = idx[0]
        self.show_current()

    # -- navigation --------------------------------------------------------
    def jump_to_first_unreviewed(self):
        if self.recheck_ids:
            self.index = 0
            self.show_current()
            return
        for i, item in enumerate(self.queue):
            if self.state.get(item["id"], {}).get("status") is None:
                self.index = i
                break
        else:
            self.index = 0
        self.show_current()

    def jump_to_next_unfinished(self, kind):
        """Next item (wrapping) of this kind with no v/x/s decision."""
        if self.mode != "detections":
            return
        n = len(self.queue)
        for step in range(1, n + 1):
            i = (self.index + step) % n
            it = self.queue[i]
            if (it["kind"] == "scene") == (kind == "scene") and not self.state.get(it["id"], {}).get("status"):
                self.index = i
                self.show_current()
                return
        what = "scene checks" if kind == "scene" else "detections"
        messagebox.showinfo("All done", f"No unfinished {what} left.")

    def go_next(self):
        if self.mode == "additions":
            if self.addition_index < len(self.additions["features"]) - 1:
                self.addition_index += 1
                self.show_current()
            else:
                messagebox.showinfo("Done", "That's the last added box.")
            return
        if self.index < len(self.queue) - 1:
            self.index += 1
            self.show_current()
        else:
            left = sum(1 for it in self.queue if not self.state.get(it["id"], {}).get("status"))
            messagebox.showinfo("End of queue", "That's the last item in the queue." +
                                (f"\n\n{left} item(s) are still undecided — use u / n to find them." if left
                                 else "\n\nEverything is decided. Export Labels from the project screen."))

    def go_previous(self):
        if self.mode == "additions":
            if self.addition_index > 0:
                self.addition_index -= 1
                self.show_current()
            return
        if self.index > 0:
            self.index -= 1
            self.show_current()

    # -- added-box review mode ---------------------------------------------
    def toggle_review_mode(self):
        if self.mode == "detections":
            if not self.additions["features"]:
                messagebox.showinfo("No added boxes", "No manually-added boxes to review yet.")
                return
            self.mode = "additions"
            self.addition_index = min(self.addition_index, len(self.additions["features"]) - 1)
            self.review_additions_btn.config(text="Back to Detections (r)")
        else:
            self.mode = "detections"
            self.review_additions_btn.config(text="Review Added Boxes (r)")
        self.show_current()

    def delete_current_addition(self):
        if self.mode != "additions" or not self.additions["features"]:
            return
        idx = self.addition_index
        feat = self.additions["features"].pop(idx)
        self.undo_stack.append(("del_add", feat, idx))
        self.project.save_additions()
        self.project.log_action("delete_addition", feature=feat)
        if self.addition_index >= len(self.additions["features"]):
            self.addition_index = max(0, len(self.additions["features"]) - 1)
        if not self.additions["features"]:
            self.mode = "detections"
            self.review_additions_btn.config(text="Review Added Boxes (r)")
        self._refresh_image_rows()
        self.show_current()

    # -- undo (every action) -------------------------------------------------
    def _record(self, item):
        """Snapshot an item's record before changing it (None = no record yet)."""
        before = self.state.get(item["id"])
        return (item["id"], json.loads(json.dumps(before)) if before is not None else None)

    def _restore(self, item_id, before):
        if before is None:
            self.state.pop(item_id, None)
        else:
            self.state[item_id] = before

    def undo(self):
        if not self.undo_stack:
            self.status_label.config(text="nothing to undo in this session", foreground="#666")
            return
        op = self.undo_stack.pop()
        kind = op[0]
        goto, msg = None, "undone"
        if kind == "state":
            _, item_id, before, what = op
            self._restore(item_id, before)
            self.project.save_state()
            goto, msg = item_id, f"undid: {what}"
        elif kind == "bulk":
            _, snaps = op
            for item_id, before in snaps:
                self._restore(item_id, before)
            self.project.save_state()
            goto, msg = snaps[0][0] if snaps else None, f"undid: rejected {len(snaps)} detections"
        elif kind == "add":
            feat = op[1]
            if feat in self.additions["features"]:
                self.additions["features"].remove(feat)
            self.project.save_additions()
            goto, msg = feat["properties"].get("review_item_id"), "undid: added box"
        elif kind == "del_add":
            _, feat, idx = op
            self.additions["features"].insert(min(idx, len(self.additions["features"])), feat)
            self.project.save_additions()
            msg = "undid: deleted added box"
            if self.mode == "additions":
                self.addition_index = self.additions["features"].index(feat)
        elif kind == "add_class":
            _, feat, old = op
            if old is None:
                feat["properties"].pop(pj.CLASS_KEY, None)
            else:
                feat["properties"][pj.CLASS_KEY] = old
            self.project.save_additions()
            msg = "undid: class change on added box"
        self.project.log_action("undo", undone=kind, item_id=goto)
        if self.mode == "additions" and not self.additions["features"]:
            self.mode = "detections"
            self.review_additions_btn.config(text="Review Added Boxes (r)")
        if goto and self.mode == "detections":
            i = self._queue_index_of(goto)
            if i is not None:
                self.index = i
        self._refresh_image_rows()
        self.show_current()
        self.status_label.config(text=msg, foreground="#c7403a")

    # -- zoom / movement ---------------------------------------------------
    def zoom_out(self):
        self.crop_radius_m = min(self.crop_radius_m * 1.5, 1500 * max(1.0, self.default_radius / pj.CROP_RADIUS_M))
        self.show_current()

    def zoom_in(self):
        self.crop_radius_m = max(self.crop_radius_m / 1.5, 40 * self.default_radius / pj.CROP_RADIUS_M)
        self.show_current()

    def _set_pan(self, east, north):
        lim = max(self.PAN_LIMIT_M, self._view_radius() * 2)
        self.pan_m = [max(-lim, min(lim, east)), max(-lim, min(lim, north))]
        self.show_current()

    def _view_radius(self):
        item = self.queue[self.index] if self.mode == "detections" else None
        if item is not None and item["kind"] == "scene":
            return item["view_radius_m"] * self.crop_radius_m / self.default_radius
        return self.crop_radius_m

    def pan_step(self, dx, dy):
        step = self._view_radius() / 2
        self._set_pan(self.pan_m[0] + dx * step, self.pan_m[1] + dy * step)

    def recenter(self):
        self._set_pan(0.0, 0.0)

    def _m_per_canvas_px(self):
        return abs(self.view["transform"].a) / self.scale

    def _pan_press(self, event):
        if self.view is None:
            return
        self.pan_start = (event.x, event.y)
        self.pan_last = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _pan_drag(self, event):
        if self.pan_start is None:
            return
        self.canvas.move("all", event.x - self.pan_last[0], event.y - self.pan_last[1])
        self.pan_last = (event.x, event.y)

    def _pan_release(self, event):
        if self.pan_start is None:
            return
        dx_px, dy_px = event.x - self.pan_start[0], event.y - self.pan_start[1]
        self.pan_start = None
        self.canvas.config(cursor="tcross" if (self.add_mode or self.edit_mode) else "crosshair")
        if abs(dx_px) < 3 and abs(dy_px) < 3:
            if event.num == 1:
                self._click_at(event.x, event.y)   # a click, not a drag
            return
        res = self._m_per_canvas_px()
        self._set_pan(self.pan_m[0] - dx_px * res, self.pan_m[1] + dy_px * res)

    def _view_text(self):
        t = f"view: ±{int(self._view_radius())}m"
        e, n = self.pan_m
        if abs(e) >= 1 or abs(n) >= 1:
            t += f", moved {abs(e):.0f}m {'E' if e >= 0 else 'W'} {abs(n):.0f}m {'N' if n >= 0 else 'S'} (c)"
        return t

    # -- click a box to jump to it ---------------------------------------------
    def _click_at(self, x, y):
        hits = [h for h in self.hit_boxes if h[0] - 3 <= x <= h[2] + 3 and h[1] - 3 <= y <= h[3] + 3]
        if not hits:
            return
        x0, y0, x1, y1, kind, ref = min(hits, key=lambda h: (h[2] - h[0]) * (h[3] - h[1]))   # smallest wins
        if kind == "det":
            i = self._queue_index_of(ref)
            if i is not None:
                if self.mode != "detections":
                    self.mode = "detections"
                    self.review_additions_btn.config(text="Review Added Boxes (r)")
                self.index = i
                self.show_current()
        elif kind == "add" and ref in self.additions["features"]:
            self.mode = "additions"
            self.review_additions_btn.config(text="Back to Detections (r)")
            self.addition_index = self.additions["features"].index(ref)
            self.show_current()
            self.status_label.config(text="added box selected: number key = class, d = delete, r = back",
                                     foreground="#3fd17a")

    def _on_motion(self, event):
        if self.add_mode or self.edit_mode or self.pan_start is not None or self.handle_drag is not None:
            return
        if self._handle_at(event.x, event.y) is not None:
            self.canvas.config(cursor="sizing")
        elif any(h[0] - 3 <= event.x <= h[2] + 3 and h[1] - 3 <= event.y <= h[3] + 3 for h in self.hit_boxes):
            self.canvas.config(cursor="hand2")
        else:
            self.canvas.config(cursor="crosshair")

    # -- progress text ------------------------------------------------------
    def _scene_progress(self, item):
        key = item["image_key"]
        dets = [it for it in self.base_queue if it["image_key"] == key and it["kind"] != "scene"]
        scene = [it for it in self.base_queue if it["image_key"] == key and it["kind"] == "scene"]
        n_rev = sum(1 for it in dets if self.state.get(it["id"], {}).get("status"))
        n_added = sum(1 for f in self.additions["features"]
                      if self.pass_to_image.get((f["properties"].get("site"), f["properties"].get("item_id"))) == key)
        parts = [f"Image {self.image_order.index(key) + 1}/{len(self.image_order)}  ({item['site']}  {item['date']})"]
        if item["kind"] == "scene":
            parts.append(f"{len(dets)} detection{'s' if len(dets) != 1 else ''} in this image, {n_rev} reviewed")
        else:
            pos = next(i for i, it in enumerate(dets) if it["id"] == item["id"]) + 1
            parts.append(f"detection {pos} of {len(dets)} in this image  ({n_rev} reviewed)")
        if scene:
            st = self.state.get(scene[0]["id"], {}).get("status")
            parts.append({"valid": "scene check: done", "invalid": "scene check: unreadable"}.get(
                st, "scene check: pending"))
        parts.append(f"{n_added} missed object{'s' if n_added != 1 else ''} added")
        left_d = sum(1 for it in self.queue if it["kind"] != "scene" and not self.state.get(it["id"], {}).get("status"))
        left_s = sum(1 for it in self.queue if it["kind"] == "scene" and not self.state.get(it["id"], {}).get("status"))
        parts.append(f"left: {left_d} undecided (u), {left_s} scene checks (n)")
        return "   |   ".join(parts)

    # -- display -----------------------------------------------------------
    def _reset_modes(self):
        self.add_mode = False
        self.edit_mode = False
        self.add_btn.state(["!pressed"])
        self.edit_btn.state(["!pressed"])

    def show_current(self):
        key = (self.mode, self.addition_index if self.mode == "additions" else self.index)
        if key != self.view_key:
            self.view_key = key
            self.pan_m = [0.0, 0.0]
            self._reset_modes()   # draw modes persist across pan/zoom of the same item, not across items
            if self.mode == "detections":
                # widen the view if this detection's box wouldn't fit (otherwise a box larger than the
                # default view is drawn entirely off-canvas)
                fit = self._box_half_extent_m(self.queue[self.index]) * 1.3
                if fit > self.crop_radius_m:
                    self.crop_radius_m = fit
        if self.mode == "additions":
            self._show_addition()
        else:
            self._show_detection()

    def _schedule_redraw(self):
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(120, self._redraw_now)

    def _redraw_now(self):
        self._redraw_job = None
        self.show_current()

    def _reset_enhance(self):
        self.brightness.set(1.0)
        self.contrast.set(1.0)
        self.show_current()

    def _enhance(self, img):
        b, c = self.brightness.get(), self.contrast.get()
        if abs(b - 1) > 0.02:
            img = ImageEnhance.Brightness(img).enhance(b)
        if abs(c - 1) > 0.02:
            img = ImageEnhance.Contrast(img).enhance(c)
        return img

    # -- band views ------------------------------------------------------------
    def _views_for(self, path):
        if path not in self._views_cache:
            self._views_cache[path] = pj.available_views(path) if path and Path(path).exists() else ["true"]
        return self._views_cache[path]

    def cycle_view_mode(self):
        path = self._current_path()
        views = self._views_for(path)
        if len(views) == 1:
            self.status_label.config(text="this image only has red/green/blue bands -- band views need NIR/SWIR "
                                          "(e.g. the 9-band review_demo images)", foreground="#c7403a")
            return
        i = views.index(self.view_mode) if self.view_mode in views else -1
        self.view_mode = views[(i + 1) % len(views)]
        self.show_current()

    def _current_path(self):
        if self.mode == "additions" and self.additions["features"]:
            p = self.additions["features"][self.addition_index]["properties"]
            k = self.pass_to_image.get((p.get("site"), p.get("item_id")))
            return self.entries[k]["path"] if k in self.entries else None
        return self.queue[self.index]["image_path"] if self.queue else None

    def _current_image_key(self):
        if self.mode == "additions" and self.additions["features"]:
            p = self.additions["features"][self.addition_index]["properties"]
            return self.pass_to_image.get((p.get("site"), p.get("item_id")))
        return self.queue[self.index]["image_key"]

    # -- compare with another date ------------------------------------------------
    def _other_dates(self, key):
        """[(image_key, label)] for the same area, nearest date first."""
        e = self.entries.get(key)
        if e is None:
            return []
        same = [(k, o) for k, o in self.entries.items() if o["site"] == e["site"] and k != key]

        def gap(o):
            try:
                return abs((datetime.fromisoformat(o["date"]) - datetime.fromisoformat(e["date"])).days)
            except (TypeError, ValueError):
                return 10 ** 6
        same.sort(key=lambda ko: gap(ko[1]))
        return [(k, f"{o.get('date')}  ({gap(o)} days {'earlier' if (o.get('date') or '') < (e.get('date') or '') else 'later'})")
                for k, o in same]

    def _compare_target(self, key):
        others = self._other_dates(key)
        keys = [k for k, _ in others]
        target = self.compare_key if self.compare_key in keys else (keys[0] if keys else None)
        return others, target

    def _set_compare_key(self, key):
        self.compare_key = key
        self.show_current()

    def toggle_compare(self):
        others, target = self._compare_target(self._current_image_key())
        if not target:
            self.status_label.config(text="no other date of this area in the project to compare with",
                                     foreground="#c7403a")
            return
        self.compare_on = not self.compare_on
        if self.compare_on:
            self._reset_modes()
        self.compare_btn.state(["pressed" if self.compare_on else "!pressed"])
        self.show_current()

    def open_compare_window(self):
        if self.compare_win is None:
            self.compare_win = CompareWindow(self)
        self.compare_win.lift()
        self.show_current()

    # -- minimap -----------------------------------------------------------------
    def _draw_minimap(self, path):
        self.minimap.delete("all")
        self._mm = None
        if not path or not Path(path).exists() or self.view is None:
            return
        if path not in self._thumb_cache:
            img = pj.render_thumbnail(path, 150, self.rgb)
            self._thumb_cache[path] = ImageTk.PhotoImage(img) if img is not None else None
        th = self._thumb_cache[path]
        if th is None:
            return
        self.minimap.create_image(0, 0, anchor="nw", image=th)
        crs, (left, bottom, right, top) = pj.image_bounds(path)
        tw, tht = th.width(), th.height()
        self._mm = (left, top, (right - left) / tw, (top - bottom) / tht)

        def to_mm(x, y):
            return (x - left) / self._mm[2], (top - y) / self._mm[3]
        w, h = self.view["img"].size
        x0, y0 = to_mm(*(self.view["transform"] * (0, 0)))
        x1, y1 = to_mm(*(self.view["transform"] * (w, h)))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(tw, x1), min(tht, y1)
        if x1 - x0 < 4:
            x0, x1 = (x0 + x1) / 2 - 2, (x0 + x1) / 2 + 2
        if y1 - y0 < 4:
            y0, y1 = (y0 + y1) / 2 - 2, (y0 + y1) / 2 + 2
        self.minimap.create_rectangle(x0, y0, x1, y1, outline="#ffd23f", width=2)

    def _on_minimap_click(self, event):
        if not getattr(self, "_mm", None) or self.to_view is None:
            return
        left, top, rx, ry = self._mm
        mx, my = left + event.x * rx, top - event.y * ry
        cx, cy = self.to_view.transform(*self._view_center)
        self._set_pan(mx - cx, my - cy)
        self.canvas.focus_set()

    # -- rendering -------------------------------------------------------------
    def _render(self, path, lon, lat, radius, image_key):
        self._view_center = (lon, lat)
        self.hit_boxes = []
        self.cur_box = None
        show_path, banner = path, None
        others, target = self._compare_target(image_key)
        if self.compare_on and target:
            show_path = self.entries[target]["path"]
            banner = f"OTHER DATE: {self.entries[target].get('date')}  (t = back to {self.entries[image_key].get('date')})"
        elif self.compare_on:
            self.compare_on = False
            self.compare_btn.state(["!pressed"])
        self.view = pj.render_view(show_path, lon, lat, radius, offset_m=tuple(self.pan_m), rgb_bands=self.rgb,
                                   max_px=max(self.canvas_size * 2, 800), stretch_mode=self.stretch_mode,
                                   view_mode=self.view_mode)
        self.canvas.delete("all")
        if self.view is None:
            self.canvas.create_text(self.canvas_size // 2, self.canvas_size // 2,
                                    text="source image not found", fill="white")
            self.to_view = None
            self._draw_minimap(None)
            return False
        img = self._enhance(self.view["img"])
        self.to_view = Transformer.from_crs("EPSG:4326", self.view["crs"], always_xy=True)
        self.scale = self.canvas_size / max(img.size)
        disp = img.resize((max(1, int(img.width * self.scale)), max(1, int(img.height * self.scale))), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        if banner:
            t = self.canvas.create_text(10, 10, anchor="nw", text=banner, fill="#111", font=("TkDefaultFont", 11, "bold"))
            bb = self.canvas.bbox(t)
            self.canvas.create_rectangle(bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4, fill="#ffd23f", outline="")
            self.canvas.tag_raise(t)
        names = dict(pj.VIEW_MODES)
        mode_name = names.get(self.view.get("view_mode", "true"), ("True colour",))[0]
        avail = self._views_for(path)
        note = "" if self.view.get("view_mode") == self.view_mode else f"  ({self.view_mode} not available here)"
        self.view_mode_label.config(text=f"bands: {mode_name}{note}" + (f"   [{len(avail)} views, b cycles]"
                                                                        if len(avail) > 1 else ""))
        self._draw_minimap(show_path)
        return True

    def _sync_compare_window(self, image_key, lon, lat, radius, boxes):
        if self.compare_win is None:
            return
        others, target = self._compare_target(image_key)
        path = self.entries[target]["path"] if target else None
        self.compare_win.render(others, target, path, lon, lat, radius, tuple(self.pan_m), boxes)

    def _class_text(self, x0, y0, cls, color):
        if cls:
            self.canvas.create_text(x0 + 2, y0 - 2, anchor="sw", text=cls, fill=color,
                                    font=("TkDefaultFont", 9, "bold"))

    def _show_detection(self):
        self.canvas.config(cursor="tcross" if (self.add_mode or self.edit_mode) else "crosshair")
        item = self.queue[self.index]
        is_scene = item["kind"] == "scene"
        rec = self.state.get(item["id"], {})
        is_recheck = is_scene and bool(rec.get("corrected_box"))
        self.instr_label.config(text=INSTRUCTIONS["recheck" if is_recheck else "scene" if is_scene else "detection"])
        prefix = ""
        if item["id"] in self.recheck_ids:
            prefix = f"RECHECK {self.recheck_ids.index(item['id']) + 1}/{len(self.recheck_ids)}" + \
                     ("" if is_recheck else " (old box cleared)") + "   |   "
        self.scene_label.config(text=prefix + self._scene_progress(item))
        n_reviewed = sum(1 for it in self.queue if self.state.get(it["id"], {}).get("status"))
        self.progress_label.config(text=f"{self.index + 1} / {len(self.queue)}   ({n_reviewed} reviewed)")
        self.zoom_label.config(text=self._view_text())
        if self.img_tree.exists(item["image_key"]):
            self.img_tree.selection_set(item["image_key"])
            self.img_tree.see(item["image_key"])

        status = rec.get("status", "unreviewed")
        radius = self._view_radius()
        if not self._render(item["image_path"], item["lon"], item["lat"], radius, item["image_key"]):
            self.info_label.config(text=f"{item['site']} | {item['date']} | source missing | status: {status}")
            self.status_label.config(text=item["image_path"])
            return

        compare_boxes = []
        corrected = rec.get("corrected_box")
        if corrected:
            self.cur_box = self._draw_polygon(corrected, outline="#ffa500")
            compare_boxes.append((corrected, "#ffa500"))
        elif not is_scene:
            self.cur_box = self._draw_detection_box(item, outline="#ff4030", width=3)
            compare_boxes.append((pj.detection_box(item), "#ff4030"))
        if self.cur_box and not is_scene:
            self._class_text(self.cur_box[0], self.cur_box[1], rec.get(pj.CLASS_KEY), "#ffd23f")
            if not self.compare_on:
                self._draw_handles()

        for feat in self.additions["features"]:
            fp = feat["properties"]
            same = self.pass_to_image.get((fp.get("site"), fp.get("item_id"))) == item["image_key"]
            if fp.get("review_item_id") == item["id"] or (is_scene and same):
                bb = self._draw_polygon(feat["geometry"], outline="#3fd17a")
                self._class_text(bb[0], bb[1], fp.get(pj.CLASS_KEY), "#3fd17a")
                self.hit_boxes.append((*bb, "add", feat))
                compare_boxes.append((feat["geometry"], "#3fd17a"))

        compare_boxes += self._draw_other_detections(item["image_key"], exclude_id=item["id"])
        self._sync_compare_window(item["image_key"], item["lon"], item["lat"], radius, compare_boxes)

        if is_scene:
            cls = self.classes[self.active_class - 1] if self.active_class else "no class"
            blind = "HOLD-OUT (blind: draw every object)  |  " if item["image_key"] in self.holdout else ""
            self.info_label.config(text=f"{blind}SCENE CHECK  {item['site']}  |  {item['date']}  |  new boxes: {cls}")
        else:
            src = f"  |  {item['source']}" if item.get("source") else ""
            score = f"  |  score {item['score']:.2f}" if item.get("score") is not None else ""
            area = f"  |  area={item['area_px']}px" if item.get("area_px") is not None else ""
            cls = f"  |  class: {rec['class']}" if rec.get(pj.CLASS_KEY) else \
                (f"  |  predicted: {item['pred_class']} (v accepts)" if item.get("pred_class") else "")
            self.info_label.config(text=f"{item['site']}  |  {item['date']}{src}{score}{area}{cls}")
        if not (self.add_mode or self.edit_mode):
            self.status_label.config(text=f"status: {status}", foreground=STATUS_COLOR.get(status, "#888"))

    def _show_addition(self):
        self.canvas.config(cursor="arrow")
        feat = self.additions["features"][self.addition_index]
        p = feat["properties"]
        coords = feat["geometry"]["coordinates"][0][:4]
        lon = sum(c[0] for c in coords) / 4
        lat = sum(c[1] for c in coords) / 4
        ikey = self.pass_to_image.get((p.get("site"), p.get("item_id")))
        path = self.entries[ikey]["path"] if ikey in self.entries else None
        self.progress_label.config(text=f"added box {self.addition_index + 1} / {len(self.additions['features'])}")
        self.instr_label.config(text=INSTRUCTIONS["additions"])
        same = [f for f in self.additions["features"]
                if (f["properties"].get("site"), f["properties"].get("item_id")) == (p.get("site"), p.get("item_id"))]
        self.scene_label.config(text=f"{p.get('site')}  {p.get('date')}   |   added box {same.index(feat) + 1} of "
                                     f"{len(same)} in this image")
        self.zoom_label.config(text=self._view_text())
        if ikey and self.img_tree.exists(ikey):
            self.img_tree.selection_set(ikey)
            self.img_tree.see(ikey)
        if not ikey or not self._render(path, lon, lat, self.crop_radius_m, ikey):
            self.info_label.config(text=f"{p.get('site')} | {p.get('date')} | source image not in this project")
            self.status_label.config(text="")
            return
        bb = self._draw_polygon(feat["geometry"], outline="#3fd17a")
        self._class_text(bb[0], bb[1], p.get(pj.CLASS_KEY), "#3fd17a")
        boxes = [(feat["geometry"], "#3fd17a")] + self._draw_other_detections(ikey)
        self._sync_compare_window(ikey, lon, lat, self.crop_radius_m, boxes)
        cls = p.get(pj.CLASS_KEY) or "no class"
        self.info_label.config(text=f"{p.get('site')}  |  {p.get('date')}  |  added {str(p.get('added_at'))[:16]}"
                                    f"  |  class: {cls}")
        self.status_label.config(text="manually-added box -- d deletes it, number key sets its class",
                                 foreground="#3fd17a")

    # -- drawing -----------------------------------------------------------
    def _lonlat_to_canvas(self, lon, lat):
        x, y = self.to_view.transform(lon, lat)
        col, row = ~self.view["transform"] * (x, y)
        return col * self.scale, row * self.scale

    def _draw_polygon(self, geometry, outline, width=3, dash=None):
        pts = [self._lonlat_to_canvas(lon, lat) for lon, lat in geometry["coordinates"][0][:4]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bb = (min(xs), min(ys), max(xs), max(ys))
        self.canvas.create_rectangle(*bb, outline=outline, width=width, dash=dash)
        return bb

    def _draw_detection_box(self, item, outline, width, dash=None):
        """Auto-detected box: its polygon if it has one, else point +/- (max(bbox,3)/2 + 2px pad)
        in the detector's pixels (detection_box() exports exactly this footprint)."""
        if item.get("det_box"):
            return self._draw_polygon(item["det_box"], outline=outline, width=width, dash=dash)
        cx, cy = self._lonlat_to_canvas(item["lon"], item["lat"])
        k = (item.get("det_res_m") or 10.0) / self._m_per_canvas_px()   # detector px -> canvas px
        half_w = (max(item.get("bbox_w_px") or 0, 3) / 2 + 2) * k
        half_h = (max(item.get("bbox_h_px") or 0, 3) / 2 + 2) * k
        bb = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        self.canvas.create_rectangle(*bb, outline=outline, width=width, dash=dash)
        return bb

    def _box_half_extent_m(self, item):
        if item["kind"] == "scene":
            return 0.0
        box = self.state.get(item["id"], {}).get("corrected_box") or item.get("det_box")
        if box:
            ring = box["coordinates"][0]
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            return max((max(lons) - min(lons)) * 111320 * math.cos(math.radians(item["lat"])),
                       (max(lats) - min(lats)) * 110570) / 2
        res = item.get("det_res_m") or 10.0
        return (max(item.get("bbox_w_px") or 0, item.get("bbox_h_px") or 0, 3) / 2 + 2) * res

    def _draw_other_detections(self, image_key, exclude_id=None):
        """Dashed boxes for every other detection in this image (clickable), so a gap near the
        current one can be read as a genuine miss rather than a object caught by a neighbour.
        Returns their geometries for the side-by-side window."""
        out = []
        for other in self.base_queue:
            if other["image_key"] != image_key or other["id"] == exclude_id or other["kind"] == "scene":
                continue
            rec = self.state.get(other["id"], {})
            color = STATUS_COLOR.get(rec.get("status"), "#4fb8ff")
            if rec.get("corrected_box"):
                bb = self._draw_polygon(rec["corrected_box"], outline=color, width=2, dash=(4, 2))
                out.append((rec["corrected_box"], color))
            else:
                bb = self._draw_detection_box(other, outline=color, width=2, dash=(4, 2))
                out.append((pj.detection_box(other), color))
            self._class_text(bb[0], bb[1], rec.get(pj.CLASS_KEY), color)
            self.hit_boxes.append((*bb, "det", other["id"]))
        return out

    # -- resize handles ------------------------------------------------------------
    def _corners(self):
        x0, y0, x1, y1 = self.cur_box
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _draw_handles(self):
        h = self.HANDLE
        for cx, cy in self._corners():
            self.canvas.create_rectangle(cx - h, cy - h, cx + h, cy + h, fill="#ffffff", outline="#111", tags="handle")

    def _handle_at(self, x, y):
        if self.cur_box is None or self.mode != "detections" or self.compare_on:
            return None
        if self.queue[self.index]["kind"] == "scene":
            return None
        for i, (cx, cy) in enumerate(self._corners()):
            if abs(x - cx) <= self.HANDLE + 3 and abs(y - cy) <= self.HANDLE + 3:
                return i
        return None

    # -- review actions ----------------------------------------------------
    def _base_record(self, item):
        return {"site": item["site"], "item_id": item["item_id"], "date": item["date"],
                "lon": item["lon"], "lat": item["lat"], "area_px": item["area_px"]}

    def _set_status(self, status, obj_class=None, advance=True):
        item = self.queue[self.index]
        _, before = self._record(item)
        # UPDATE the record, don't replace it -- replacing would discard an edited box
        rec = dict(self.state.get(item["id"]) or {})
        rec.update({**self._base_record(item), "status": status, "reviewed_at": pj.now_iso()})
        if obj_class is None and status == "valid" and not rec.get(pj.CLASS_KEY) and item.get("pred_class"):
            obj_class = item["pred_class"]   # v accepts the model's predicted class; a number key overrides
        if obj_class is not None:
            rec[pj.CLASS_KEY] = obj_class
        self.state[item["id"]] = rec
        self.project.save_state()
        self.undo_stack.append(("state", item["id"], before, f"{status} on {item['date']}"))
        self.project.log_action("status", item_id=item["id"], status=status, obj_class=rec.get(pj.CLASS_KEY),
                                before=before)
        self._refresh_image_rows([item["image_key"]])
        if advance:
            self.go_next()
        else:
            self.show_current()

    def mark_valid(self):
        if self.mode == "detections" and not self.add_mode:
            self._set_status("valid")

    def mark_invalid(self):
        if self.mode == "detections" and not self.add_mode:
            self._set_status("invalid")

    def mark_skip(self):
        if self.mode == "detections" and not self.add_mode:
            self._set_status("skipped")

    def class_key(self, k):
        """1..n: Valid as class n on a detection; on a scene check / add mode, the class for new boxes;
        on an added box, its class. 0 clears."""
        name = self.classes[k - 1] if 1 <= k <= len(self.classes) else None
        if k > len(self.classes):
            return
        if self.mode == "additions":
            if not self.additions["features"]:
                return
            feat = self.additions["features"][self.addition_index]
            old = feat["properties"].get(pj.CLASS_KEY)
            if name is None:
                feat["properties"].pop(pj.CLASS_KEY, None)
            else:
                feat["properties"][pj.CLASS_KEY] = name
            self.project.save_additions()
            self.undo_stack.append(("add_class", feat, old))
            self.project.log_action("addition_class", added_at=feat["properties"].get("added_at"), obj_class=name)
            self.show_current()
            return
        item = self.queue[self.index]
        if item["kind"] == "scene" or self.add_mode:
            self.active_class = k or None
            if item["kind"] == "scene" and not self.add_mode:
                self.toggle_add_mode()
            self.status_label.config(text=f"boxes you add now get class: {name or 'none'}", foreground="#b88a00")
            self.show_current()
            return
        if name is None:   # 0 on a detection: clear its class, keep the decision
            if self.state.get(item["id"], {}).get(pj.CLASS_KEY):
                _, before = self._record(item)
                self.state[item["id"]].pop(pj.CLASS_KEY, None)
                self.project.save_state()
                self.undo_stack.append(("state", item["id"], before, "class cleared"))
            self.show_current()
            return
        self._set_status("valid", obj_class=name)

    def reject_rest_of_image(self):
        """X: mark every still-undecided detection in this image Invalid (one undo step)."""
        if self.mode != "detections":
            return
        key = self.queue[self.index]["image_key"]
        todo = [it for it in self.base_queue if it["image_key"] == key and it["kind"] != "scene"
                and not self.state.get(it["id"], {}).get("status")]
        if not todo:
            self.status_label.config(text="no undecided detections left in this image", foreground="#666")
            return
        if not messagebox.askyesno("Reject rest of image", f"Mark the {len(todo)} undecided detection(s) in this "
                                                           f"image Invalid?\n\n(Ctrl+Z undoes all of them at once.)"):
            return
        snaps = []
        for it in todo:
            snaps.append(self._record(it))
            rec = dict(self.state.get(it["id"]) or {})
            rec.update({**self._base_record(it), "status": "invalid", "reviewed_at": pj.now_iso(), "bulk": True})
            self.state[it["id"]] = rec
        self.project.save_state()
        self.undo_stack.append(("bulk", snaps))
        self.project.log_action("bulk_invalid", item_ids=[s[0] for s in snaps], before=[s[1] for s in snaps])
        self._refresh_image_rows([key])
        scene = next((i for i, it in enumerate(self.queue) if it["image_key"] == key and it["kind"] == "scene"), None)
        if scene is not None:
            self.index = scene
        self.show_current()
        self.status_label.config(text=f"rejected {len(todo)} detections -- now on this image's scene check",
                                 foreground="#c7403a")

    def toggle_add_mode(self):
        if self.mode != "detections" or self.view is None:
            return
        if self.compare_on:
            self.status_label.config(text="showing another date -- press t to go back before drawing",
                                     foreground="#c7403a")
            return
        self.add_mode = not self.add_mode
        if self.add_mode:
            self.edit_mode = False
            self.edit_btn.state(["!pressed"])
        self.add_btn.state(["pressed" if self.add_mode else "!pressed"])
        self.canvas.config(cursor="tcross" if self.add_mode else "crosshair")
        cls = self.classes[self.active_class - 1] if self.active_class else "no class (press 1-9 to set)"
        self.status_label.config(text=f"ADD MODE: drag a box around each missed object (a again to stop). Class: {cls}"
                                 if self.add_mode else "", foreground="#b88a00")

    def toggle_edit_mode(self):
        if self.mode != "detections" or self.view is None:
            return
        if self.compare_on:
            self.status_label.config(text="showing another date -- press t to go back before drawing",
                                     foreground="#c7403a")
            return
        if self.queue[self.index]["kind"] == "scene":
            self.status_label.config(
                text="Redraw Box fixes ONE detection's box -- not available on a scene check. Use  a  to flag a "
                     "missed object here, or click that detection's dashed box to go to it.", foreground="#c7403a")
            return
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.add_mode = False
            self.add_btn.state(["!pressed"])
        self.edit_btn.state(["pressed" if self.edit_mode else "!pressed"])
        self.canvas.config(cursor="tcross" if self.edit_mode else "crosshair")
        self.status_label.config(text="EDIT MODE: draw a corrected box over the object (or just drag the white "
                                      "corner handles)" if self.edit_mode else "", foreground="#ffa500")

    def reset_box(self):
        if self.mode != "detections":
            return
        item = self.queue[self.index]
        rec = self.state.get(item["id"])
        if rec and rec.get("corrected_box") is not None:
            _, before = self._record(item)
            for k in ("corrected_box", "corrected_bbox_w_px", "corrected_bbox_h_px", "box_edited_at"):
                rec.pop(k, None)
            self.project.save_state()
            self.undo_stack.append(("state", item["id"], before, "reset box"))
            self.project.log_action("reset_box", item_id=item["id"], before=before)
            self._refresh_image_rows([item["image_key"]])
        self.show_current()

    def _on_press(self, event):
        if not (self.add_mode or self.edit_mode):
            h = self._handle_at(event.x, event.y)
            if h is not None:   # resize: drag a corner of the current detection's box
                self.handle_drag = h
                self.rect_id = self.canvas.create_rectangle(*self.cur_box, outline="#ffa500", width=2)
                return
            self._pan_press(event)
            return
        if self.mode != "detections" or self.view is None:
            return
        self.rect_start = (event.x, event.y)
        color = "#ffd23f" if self.add_mode else "#ffa500"
        self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=color, width=2)

    def _on_drag(self, event):
        if self.handle_drag is not None:
            x0, y0, x1, y1 = self.cur_box
            if self.handle_drag in (0, 3):
                x0 = event.x
            else:
                x1 = event.x
            if self.handle_drag in (0, 1):
                y0 = event.y
            else:
                y1 = event.y
            self.canvas.coords(self.rect_id, x0, y0, x1, y1)
            return
        if self.pan_start is not None:
            self._pan_drag(event)
            return
        if not (self.add_mode or self.edit_mode) or self.rect_id is None:
            return
        x0, y0 = self.rect_start
        self.canvas.coords(self.rect_id, x0, y0, event.x, event.y)

    def _on_release(self, event):
        if self.handle_drag is not None:
            x0, y0, x1, y1 = self.canvas.coords(self.rect_id)
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.handle_drag = None
            if abs(x1 - x0) >= 4 and abs(y1 - y0) >= 4:
                self._save_corrected_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            else:
                self.show_current()
            return
        if self.pan_start is not None:
            self._pan_release(event)
            return
        if not (self.add_mode or self.edit_mode) or self.rect_id is None:
            return
        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y
        self.canvas.delete(self.rect_id)
        self.rect_id = None
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return  # accidental click
        box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if self.add_mode:
            self._save_manual_box(*box)
        else:
            self._save_corrected_box(*box)

    def _canvas_to_map(self, cx, cy):
        return self.view["transform"] * (cx / self.scale, cy / self.scale)

    def _canvas_box_to_lonlat(self, cx0, cy0, cx1, cy1):
        inv = Transformer.from_crs(self.view["crs"], "EPSG:4326", always_xy=True)
        ring = [list(inv.transform(*self._canvas_to_map(cx, cy)))
                for cx, cy in ((cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1))]
        return ring + [ring[0]]

    def _canvas_box_to_source_px(self, cx0, cy0, cx1, cy1):
        """Box in the source image's own pixel grid (col0, row0, col1, row1), for cropping chips."""
        inv = ~self.view["native_transform"]
        to_native = Transformer.from_crs(self.view["crs"], self.view["native_crs"], always_xy=True)
        c0, r0 = inv * to_native.transform(*self._canvas_to_map(cx0, cy0))
        c1, r1 = inv * to_native.transform(*self._canvas_to_map(cx1, cy1))
        return [round(min(c0, c1), 1), round(min(r0, r1), 1), round(max(c0, c1), 1), round(max(r0, r1), 1)]

    def _save_manual_box(self, cx0, cy0, cx1, cy1):
        item = self.queue[self.index]
        props = {"site": item["site"], "item_id": item["item_id"], "date": item["date"],
                 "review_item_id": item["id"], "added_at": pj.now_iso(),
                 "pixel_box": self._canvas_box_to_source_px(cx0, cy0, cx1, cy1), "image_path": item["image_path"]}
        if self.active_class:
            props[pj.CLASS_KEY] = self.classes[self.active_class - 1]
        feat = {"type": "Feature", "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [self._canvas_box_to_lonlat(cx0, cy0, cx1, cy1)]}}
        self.additions["features"].append(feat)
        self.undo_stack.append(("add", feat))
        self.project.save_additions()
        self.project.log_action("add_box", feature=feat)
        bb = self._draw_polygon(feat["geometry"], outline="#3fd17a")
        self._class_text(bb[0], bb[1], props.get(pj.CLASS_KEY), "#3fd17a")
        self.hit_boxes.append((*bb, "add", feat))
        self.status_label.config(text=f"saved missed-object box ({len(self.additions['features'])} total added; "
                                      f"Ctrl+Z undoes)", foreground="#3fd17a")
        self.scene_label.config(text=self._scene_progress(item))
        self._refresh_image_rows([item["image_key"]])

    def _save_corrected_box(self, cx0, cy0, cx1, cy1):
        item = self.queue[self.index]
        _, before = self._record(item)
        rec = dict(self.state.get(item["id"]) or {**self._base_record(item), "status": None, "reviewed_at": None})
        k = self._m_per_canvas_px() / self.view["native_res"]   # canvas px -> source px
        rec["corrected_box"] = {"type": "Polygon", "coordinates": [self._canvas_box_to_lonlat(cx0, cy0, cx1, cy1)]}
        rec["corrected_bbox_w_px"] = round(abs(cx1 - cx0) * k, 1)
        rec["corrected_bbox_h_px"] = round(abs(cy1 - cy0) * k, 1)
        rec["box_edited_at"] = pj.now_iso()
        auto_valid = self.project.settings.get("edit_marks_valid", True) and not rec.get("status")
        if auto_valid:   # fixing a box means it's a vessel: don't leave it undecided
            rec["status"] = "valid"
            rec["reviewed_at"] = pj.now_iso()
        self.state[item["id"]] = rec
        self.project.save_state()
        self.undo_stack.append(("state", item["id"], before, "box edit"))
        self.project.log_action("edit_box", item_id=item["id"], box=rec["corrected_box"], auto_valid=auto_valid,
                                before=before)
        self.edit_mode = False
        self.edit_btn.state(["!pressed"])
        self._refresh_image_rows([item["image_key"]])
        self.show_current()
        self.status_label.config(
            text=("box saved and marked VALID -- → for next, x if it isn't a object, Ctrl+Z to undo" if auto_valid
                  else f"box saved (status stays {rec.get('status') or 'undecided'})"), foreground="#ffa500")
