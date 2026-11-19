"""Shared UI pieces: help texts, colours, threading helper, collapsible sections, platform keys."""
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .. import config

STATUS_COLOR = {"valid": "#1a9c5c", "invalid": "#c7403a", "skipped": "#a98a1a"}


INSTRUCTIONS = {
    "detection": (
        "DETECTION REVIEW: is the RED box a object on the water (object, boat, barge, tug)?\n"
        "  v Valid    x Invalid (land, pier, cloud, glint, buoy, reef)    s Skip (can't tell)\n"
        "  1-9  Valid AS that object class (one key does both)    0 clear class\n"
        "  drag a white corner handle (or e, then draw) to fix the box -- that also marks it Valid\n"
        "  a  Flag Missed Object: drag a box around any unboxed object in this view\n"
        "  X  reject every undecided detection left in this image    Ctrl+Z undo anything\n"
        "Dashed boxes = this image's other detections (green valid, red invalid, blue unreviewed): CLICK one "
        "to jump to it. Score/source are in the top bar; reorder by score in View.\n"
        "LOOK: b band views (NIR/SWIR/water index)   t flick to another date   w side-by-side dates   "
        "sliders = brightness/contrast   minimap: click to move.\n"
        "MOVE: drag the image (right-drag while drawing), Shift+arrows, c re-centre, -/+/wheel zoom, "
        "u next undecided, n next scene check.\n"
        "MODEL: source model_vN = pre-labels from your trained model (v also accepts its predicted class). "
        "Image list: [H] hold-out (blind), [S] suggested next."),
    "scene": (
        "SCENE CHECK: the whole image. Look for every object WITHOUT a box.\n"
        "  a  then drag a box around each missed object (one per object; stays on until a again)\n"
        "  1-9  choose the class boxes get as you add them (turns add mode on)    0 no class\n"
        "  v  Valid when you've flagged every missed object: marks the image fully checked\n"
        "  x  Invalid if cloud/haze makes the image unreadable (excluded from export/scoring)\n"
        "Dashed boxes = detections (click to jump); solid green = objects you added (click to select / "
        "set class / delete); orange = an old edit saved on this scene check. Ctrl+Z undoes anything.\n"
        "LOOK: t / w compare with another date (a pier is there every time, a object usually isn't), b band "
        "views, minimap to move around the image.\n"
        "MOVE: drag the image (right-drag while drawing), Shift+arrows, c re-centre, -/+/wheel zoom."),
    "recheck": (
        "RECHECK: the ORANGE box is an old Edit Box saved on this scene check, so it isn't linked to any "
        "detection and won't be counted.\n"
        "  If it marks a object no dashed box covers: press  a  and drag a new box over it (counts as a missed object)\n"
        "  Then press  Reset Box  to clear the old orange box, then  v  when this image is fully checked\n"
        "  If the orange box isn't a object: just  Reset Box  and carry on with the scene check\n"
        "MOVE: drag the image, Shift+arrows step, c re-centres; -/+ or wheel zoom."),
    "additions": (
        "ADDED-BOX REVIEW: each box you drew for a missed object, one at a time.\n"
        "  d  delete this box (misclick or not a object)     1-9 set its class, 0 clear\n"
        "  ←/→ previous/next     r back to the detection queue     Ctrl+Z undo\n"
        "MOVE: drag the image, Shift+arrows step, c re-centres; -/+ or wheel zoom."),
}


SHORTCUTS = """Review screen
  v / x / s        Valid / Invalid / Skip (then advance)
  1-9              Valid as that object class (detection); class for new boxes (scene check);
                   class of the shown added box (added-box review).   0 clears
  corner handles   drag to resize the detection's box (also marks it Valid)
  e                Redraw Box from scratch (also marks it Valid)
  Reset Box        clear an edited box
  a                Flag Missed Object (drag a box; repeat for each)
  X (shift+x)      reject every undecided detection left in this image
  Ctrl+Z           undo the last action (any kind), repeatable
  click a box      jump to that detection / select that added box
  r                Review Added Boxes  /  back to detections
  d                delete the shown added box (in added-box review)
  u / n            next undecided detection / next unfinished scene check
  b                cycle band views (true colour, NIR, SWIR, water index)
  t                flick to the same place on another date of this port
  w                side-by-side window with another date
  ← / →            previous / next item
  - / + / wheel    zoom out / in
  drag, right-drag pan the image (right-drag also works while drawing)
  Shift+arrows     pan a quarter view
  c                re-centre
  Esc              back to the project dashboard"""


IS_MAC = sys.platform == "darwin"
# the "secondary" mouse button is Button-2 on macOS, Button-3 elsewhere
RIGHT = "2" if IS_MAC else "3"
UNDO_KEYS = ("<Command-z>",) if IS_MAC else ("<Control-z>",)


def open_folder(path):
    """Show a folder in the system file manager (Explorer / Finder / xdg-open)."""
    try:
        if sys.platform.startswith("win"):
            import os
            os.startfile(str(path))  # noqa: S606 -- local folder chosen by the app
        elif IS_MAC:
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        messagebox.showinfo("Folder", str(path))


def presets():
    """Recently used (imagery, detections) folders, offered in the wizard and Add images."""
    return [(f"{Path(i).name}  —  {i}" + ("  (+ detections)" if d else ""), Path(i), Path(d) if d else None)
            for i, d in config.recent_sources()]


def run_in_thread(widget, fn, on_done, on_error=None):
    """Run fn() off the Tk thread; call on_done(result) back on it (Tk isn't thread-safe)."""
    box = {}

    def work():
        try:
            box["result"] = fn()
        except Exception as exc:  # surfaced to the user below
            box["error"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()

    def poll():
        try:
            if not widget.winfo_exists():
                return   # dialog closed while the work ran (e.g. wizard cancelled mid-scan): drop the result
        except tk.TclError:
            return
        if t.is_alive():
            widget.after(80, poll)
        elif "error" in box:
            (on_error or (lambda e: messagebox.showerror("Error", f"{e.__class__.__name__}: {e}")))(box["error"])
        else:
            on_done(box.get("result"))
    widget.after(80, poll)


class Collapsible(ttk.Frame):
    """A section that starts folded: '+ title' button; click to show/hide its body."""

    def __init__(self, master, title, open_=False, on_toggle=None):
        super().__init__(master)
        self.title = title
        self.is_open = open_
        self.on_toggle = on_toggle
        self.btn = ttk.Button(self, text="", style="Toolbutton", command=self.toggle)
        self.btn.pack(anchor="w")
        self.body = ttk.Frame(self)
        self._sync()

    def _sync(self):
        self.btn.config(text=("−  " if self.is_open else "+  ") + self.title)
        if self.is_open:
            self.body.pack(fill="both", expand=True)
        else:
            self.body.pack_forget()

    def toggle(self):
        self.is_open = not self.is_open
        self._sync()
        if self.on_toggle:
            self.on_toggle(self.is_open)


def menu_button(parent, text, items):
    """ttk.Menubutton with (label, command) items; None = separator."""
    mb = ttk.Menubutton(parent, text=text)
    m = tk.Menu(mb, tearoff=0)
    for it in items:
        if it is None:
            m.add_separator()
        else:
            m.add_command(label=it[0], command=it[1])
    mb["menu"] = m
    return mb
