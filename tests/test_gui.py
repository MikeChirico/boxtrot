"""Drives the real Tk app (windows appear briefly) and FAILS on any exception raised inside a Tk
callback, not just on wrong results. Covers the setup wizard (incl. cancel mid-scan), every review
action, dialogs, a degree-based (EPSG:4326) image, and Model-tab lifecycle. Needs a display
(CI uses xvfb-run)."""
import tkinter as tk
import tkinter.messagebox
import traceback

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from boxtrot.core import assist
from boxtrot.core import project as pj
from boxtrot.testing import make_detections, make_scene_set
from boxtrot.ui.dialogs import ExportDialog, SettingsDialog
from boxtrot.ui.wizard import SetupWizard

pytestmark = pytest.mark.gui
FAILS, TK_ERRORS = [], []
W = {}


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


class Ev:
    def __init__(self, x, y, num=1):
        self.x, self.y, self.num = x, y, num




def pump(n=3):
    for _ in range(n):
        root.update()


def wait(cond, then, t=0):
    if cond():
        root.after(10, then)
    elif t > 900:
        FAILS.append(f"timeout waiting in {then.__name__}")
        finish()
    else:
        root.after(100, lambda: wait(cond, then, t + 1))


def drag(rv, pts):
    rv._on_press(Ev(*pts[0]))
    for p in pts[1:]:
        rv._on_drag(Ev(*p))
    rv._on_release(Ev(*pts[-1]))


W = {}


def step_wizard_cancel():
    print("wizard: cancel while scanning")
    w = SetupWizard(root)
    w.name.set("cancelled")
    w.parent.set(str(DATA.parent))
    w.next()
    w.img_dir.set(str(DATA / "images"))
    w._scan()
    w.destroy()    # the scan thread finishes later and must not touch the dead dialog
    root.after(4000, step_wizard_review)


def step_wizard_review():
    check(not TK_ERRORS, "no callback errors after cancelling mid-scan")
    print("wizard: detections -> review")
    w = SetupWizard(root)
    W["w"] = w
    w.next()
    check(w.step == 0, "step 1 refuses an empty name")
    w.name.set("Review run")
    w.parent.set(str(DATA.parent))
    w.next()
    w.img_dir.set(str(DATA / "images"))
    w._scan()
    wait(lambda: w.cands, step_wizard_review2)


def step_wizard_review2():
    w = W["w"]
    w.site_on["south_harbour"].set(False)
    w.d_from.set("2025-03-01")
    pump()
    n_sel = len(w.selected())
    check(0 < n_sel < 15, f"area + date filters applied ({n_sel} images)")
    w.next()
    w.has_dets.set("yes")
    w._sync_dets()
    w.det_dir.set(str(DATA / "detections"))
    w.next()
    wait(lambda: w.step == 3, step_wizard_review3)


def step_wizard_review3():
    w = W["w"]
    w.object_name.set("boat")
    w.use_classes.set(True)
    w._sync_classes()
    w.classes.set("big, small")
    w.next()
    check(w.step == 4 and w.start.get() == "review", "start step recommends reviewing detections")
    w.next()
    wait(lambda: w.result is not None, step_review)


def step_review():
    w = W["w"]
    proj = w.result
    check(all(e["detections"] for e in proj.imagery) and proj.classes == ["big", "small"], "project created with detections + types")
    app.project = proj
    app.show_project(welcome="review")
    pump()
    print("review screen")
    app.start_review()
    rv = app.view
    it = rv.queue[rv.index]
    x0, y0, x1, y1 = rv.cur_box
    drag(rv, [(x1, y1), (x1 - 25, y1 - 20)])
    check(proj.state[it["id"]].get("status") == "valid" and proj.state[it["id"]].get("corrected_box"),
          "corner handle resizes the box and marks it Valid")
    rv.undo()
    check(it["id"] not in proj.state, "undo reverts the resize")
    rv.class_key(2)
    check(proj.state[it["id"]].get(pj.CLASS_KEY) == rv.classes[1], "number key = Valid as that type")
    rv.undo()
    rv.mark_invalid()
    rv.mark_skip()
    rv.order_var.set("Lowest score first")
    rv._apply_order()
    sc = [q["score"] for q in rv.queue if q["kind"] != "scene" and q.get("score") is not None]
    check(sc == sorted(sc), "lowest-score-first order")
    rv.source_var.set(rv.sources[-1])
    rv._apply_order()
    check(all(q["kind"] == "scene" or (q.get("source") or "detector") == rv.sources[-1] for q in rv.queue),
          "source filter")
    rv.source_var.set("All sources")
    rv.order_var.set("Image order")
    rv._apply_order()
    key = next(k for k in rv.image_order
               if sum(1 for q in rv.base_queue if q["image_key"] == k and q["kind"] != "scene") >= 3)
    rv.jump_to_image(key)
    hits = [h for h in rv.hit_boxes if h[4] == "det"]
    if hits:
        h = hits[0]
        cx, cy = (h[0] + h[2]) / 2, (h[1] + h[3]) / 2
        rv._on_press(Ev(cx, cy))
        rv._on_release(Ev(cx, cy))
        check(rv.queue[rv.index]["id"] == h[5], "click a dashed box jumps to it")
    n0 = len(proj.state)
    rv.reject_rest_of_image()
    check(rv.queue[rv.index]["kind"] == "scene", "reject rest of image -> its scene check")
    rv.undo()
    check(len(proj.state) == n0, "one undo restores the whole bulk reject")
    rv.index = next(i for i, q in enumerate(rv.queue) if q["image_key"] == key and q["kind"] == "scene")
    rv.show_current()
    rv.class_key(1)
    check(rv.add_mode, "type key on a scene check turns add mode on")
    drag(rv, [(300, 300), (330, 322)])
    check(proj.additions["features"][-1]["properties"].get(pj.CLASS_KEY) == rv.classes[0], "added box gets the type")
    rv.toggle_add_mode()
    rv.cycle_view_mode()
    check(rv.view_mode == "nir", "b cycles to the NIR view")
    rv.view_mode = "true"
    rv.brightness.set(1.5)
    rv._redraw_now()
    rv._on_minimap_click(Ev(15, 15))
    check(rv.pan_m != [0.0, 0.0], "minimap click pans")
    rv.recenter()
    rv.toggle_compare()
    check(rv.compare_on, "t flicks to another date")
    rv.toggle_add_mode()
    check(not rv.add_mode, "drawing blocked while showing another date")
    rv.toggle_compare()
    rv.open_compare_window()
    pump()
    check(rv.compare_win is not None and rv.compare_win.winfo_exists(), "side-by-side window opens")
    rv.toggle_review_mode()
    check(rv.mode == "additions", "added-box review")
    rv.class_key(3)
    rv.undo()
    rv.toggle_review_mode()
    rv.view_sec.toggle()
    rv.view_sec.toggle()
    rv.back()
    pump()
    check(rv.compare_win is None, "side-by-side window closed when leaving review")
    for dlg in (SettingsDialog(root, app.project), ExportDialog(root, app.project)):
        pump()
        dlg.destroy()
    step_wizard_assist()


def step_wizard_assist():
    print("wizard: no detections -> model-assisted")
    w = SetupWizard(root)
    W["w"] = w
    w.name.set("Assist run")
    w.parent.set(str(DATA.parent))
    w.next()
    w.img_dir.set(str(DATA / "images"))
    w._scan()
    wait(lambda: w.cands, step_wizard_assist2)


def step_wizard_assist2():
    w = W["w"]
    w.next()
    w.next()
    w.object_name.set("boat")
    w.next()
    check(w.step == 4 and w.start.get() == "assist", "no detections + 30 images -> model-assisted recommended")
    w.next()
    wait(lambda: w.result is not None, step_model_tab)


def step_model_tab():
    proj = W["w"].result
    st = assist.load_assist(proj)
    check(st["holdout"] and len(st["suggested"]) == 20 and not proj.classes, "hold-out + 20 suggestions, no types")
    app.project = proj
    app.show_project(tab="Model", welcome="assist")
    pump()
    mp = app.view.model_panel
    check("hold-out" in mp.next_btn.cget("text"), f"next-step button: {mp.next_btn.cget('text')!r}")
    mp._poll_job = mp.after(60000, lambda: None)   # stand-in timer
    app.start_review()   # destroys the dashboard (and the Model tab)
    pump()
    check(mp._poll_job is None, "Model tab timer cancelled when the tab is destroyed")
    rv = app.view
    n_before = len(app.project.state)
    rv.class_key(1)   # no types: a number key must be a harmless no-op
    pump()
    check(rv.classes == [] and len(app.project.state) == n_before, "review works with no types (1 is a no-op)")
    rv.back()
    step_geo()


def step_geo():
    print("degree-based (EPSG:4326) image")
    p = DATA.parent / "geo" / "siteA" / "img_20250101.tif"
    p.parent.mkdir(parents=True)
    d = np.full((3, 600, 600), 400, np.uint16)
    d[:, 300:306, 300:310] = 4000
    tfm = from_origin(-72.40, 18.60, 0.00009, 0.00009)
    with rasterio.open(p, "w", driver="GTiff", width=600, height=600, count=3, dtype="uint16", crs="EPSG:4326",
                       transform=tfm) as dst:
        dst.write(d)
        dst.descriptions = ("red", "green", "blue")
    proj = pj.Project.create(DATA.parent / "projects", "Geo")
    proj.add_imagery(pj.scan_imagery(DATA.parent / "geo"))
    app.project = proj
    app.show_project()
    pump()
    app.start_review()
    rv = app.view
    pump()
    check(rv.queue[rv.index]["kind"] == "scene" and rv.view is not None, "scene check renders (no hang)")
    rv.toggle_add_mode()
    drag(rv, [(100, 100), (140, 130)])
    f = proj.additions["features"][-1]["properties"]
    pb = f["pixel_box"]
    check(0 <= pb[0] < pb[2] <= 600 and 0 <= pb[1] < pb[3] <= 600, f"pixel_box in source pixels {pb}")
    rv._on_minimap_click(Ev(75, 75))
    rv.zoom_in()
    rv.back()
    finish()


def finish():
    check(not TK_ERRORS, f"no exceptions inside Tk callbacks ({len(TK_ERRORS)})")
    for e in TK_ERRORS[:5]:
        print(e)
    root.destroy()


def test_gui(tmp_path, monkeypatch):
    global root, app, DATA
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    from boxtrot.ui import app as app_mod
    g = globals()
    monkeypatch.setattr(tkinter.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(tkinter.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(tkinter.messagebox, "showerror", lambda *a, **k: TK_ERRORS.append(f"showerror: {a[:2]}"))
    DATA = tmp_path / "sample"
    truth = make_scene_set(DATA / "images", n=30, seed=3)
    make_detections(DATA / "detections", DATA / "images", truth)
    FAILS.clear()
    TK_ERRORS.clear()
    root.report_callback_exception = lambda *exc: TK_ERRORS.append("".join(traceback.format_exception(*exc)))
    app = app_mod.App(root)
    for name in [n for n in list(g) if n.startswith("step_")]:
        g[name] = guarded(g[name]) if not getattr(g[name], "_guarded", False) else g[name]
    root.after(300, g["step_wizard_cancel"])
    root.mainloop()
    assert not FAILS, FAILS


def guarded(fn):
    def g():
        try:
            fn()
        except Exception:
            FAILS.append(f"{fn.__name__}: " + traceback.format_exc(limit=4))
            traceback.print_exc()
            finish()
    g.__name__ = fn.__name__
    g._guarded = True
    return g
