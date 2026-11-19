"""
Model-assisted labelling: build a dataset from scratch on any project, train a detector on the
labels so far, and pre-label the rest so review becomes correcting boxes instead of drawing on
blank images.

This module is backend-agnostic and has no deep-learning dependency: training and inference go
through a plugin (core/backends.py; reference implementation addons/boxtrot-ultralytics).
Everything except Train/Predict -- hold-out, suggestions, recommendations, dataset export,
scoring -- works without one.

Loop (the Model tab drives it):
    1. hold-out   pick_holdout(): a fixed set of images labelled BLIND (no detector or model
                  boxes shown -- project.build_queue hides them), used only to score models
    2. seed       suggest_images(): which unlabelled images to label next (diversity first,
                  then model uncertainty once a model exists); label them completely
    3. train      `python -m boxtrot.core.assist train <project> <version>`: tiles the complete
                  non-hold-out images, trains via the backend, scores on the hold-out
    4. predict    `python -m boxtrot.core.assist predict <project> <version>`: tiled inference
                  on untouched images -> models/vN/detections/, recorded in models/assist.json as
                  pre-labels (source "model_vN", with scores)
    5. repeat     recommend() says when to retrain and when more labels stop paying off

Storage: <project>/models/assist.json (hold-out, suggestions, pre-labels, version records) and
<project>/models/vN/ (dataset/, weights, metrics.json, status.json, log.txt).

Method notes:
  - Images are rendered exactly as the reviewer sees them (project.pick_bands + whole-image
    2-98% stretch -> 8-bit RGB), for training AND inference, so the model sees what you saw.
  - Tiles of tile_px native pixels (overlap tile_overlap) are fed at imgsz = tile_px x upscale;
    upscale "auto" makes the median labelled box >= ~16 px at network scale (small objects
    otherwise vanish in a detector's stride-8 grid).
  - Tile-overlap duplicates are merged with class-agnostic NMS (IoU 0.5).
  - Hold-out matching: greedy by score, IoU >= 0.25 (loose on purpose: a 4-px box offset by
    one pixel is still the same object).

Caveats -- read before trusting the numbers:
  - Hold-out metrics only mean something if the hold-out was labelled blind and completely
    (scene check done). Incomplete hold-out images are skipped from scoring.
  - The label-count recommendation is a starting rule of thumb (150 objects, x1.5 for small
    objects); the learning curve over versions is the real answer -- trust it over the rule.
  - Pre-labels bias reviewers toward accepting model boxes and missing what it missed. Scene
    checks stay mandatory, and recall on the blind hold-out is the honest check.
  - Predictions are only written to images nobody has touched, so a new model never replaces
    boxes someone already reviewed.
"""
import argparse
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window

from . import backends
from . import project as pj

ASSIST_DEFAULTS = {
    "base_weights": "",         # "" = the backend's default (e.g. yolov8n.pt, downloaded on first use)
    "epochs": 60, "tile_px": 320, "tile_overlap": 48, "upscale": "auto", "conf": 0.25,
    "min_per_class": 50,        # types with fewer confirmed examples go to the catch-all class
}
MATCH_IOU = 0.25
NMS_IOU = 0.5
SEED_BOXES = 150                # first-model rule of thumb (x1.5 for small objects)
USEFUL_RECALL, USEFUL_PRECISION = 0.6, 0.5
PLATEAU = 0.03
OTHER_CLASS = "other"           # multi-class catch-all for types with too few examples / no type                  # two retrains improving both P and R by less than this = plateau


# ------------------------------------------------------------------ state ---

def assist_dir(project):
    return project.root / "models"


def load_assist(project):
    try:
        st = json.loads((assist_dir(project) / "assist.json").read_text())
    except (OSError, ValueError):
        st = {}
    st.setdefault("holdout", [])
    st.setdefault("suggested", [])
    st.setdefault("versions", [])
    return st


def save_assist(project, st):
    pj.write_json_atomic(assist_dir(project) / "assist.json", st)


def settings(project):
    return {**ASSIST_DEFAULTS, **project.settings.get("assist", {})}


def is_model_file(path):
    """True for a detections file written by Predict (<project>/models/vN/detections/...)."""
    parts = Path(path).parts
    return any(a == "models" and b.startswith("v") and b[1:].isdigit() for a, b in zip(parts, parts[1:]))


def version_dir(project, v):
    return assist_dir(project) / f"v{v}"


def next_version(project):
    st = load_assist(project)
    done = [r["version"] for r in st["versions"]]
    existing = [int(p.name[1:]) for p in assist_dir(project).glob("v*") if p.name[1:].isdigit()]
    return max(done + existing + [0]) + 1


def write_status(project, v, **kw):
    pj.write_json_atomic(version_dir(project, v) / "status.json", {"at": pj.now_iso(), **kw})


def read_status(project, v):
    try:
        return json.loads((version_dir(project, v) / "status.json").read_text())
    except (OSError, ValueError):
        return {}


# -------------------------------------------------------------- inventory ---

def _touched(p):
    return bool(p["valid"] + p["invalid"] + p["skipped"] + p["edited_only"] + p["added"]) or bool(p["scene"])


def _box_size_px(geom, lat, res_m):
    ring = geom["coordinates"][0]
    lons = [q[0] for q in ring]
    lats = [q[1] for q in ring]
    w = (max(lons) - min(lons)) * 111320 * math.cos(math.radians(lat)) / res_m
    h = (max(lats) - min(lats)) * 110570 / res_m
    return math.sqrt(max(w, 0.1) * max(h, 0.1))


def inventory(project, queue=None):
    """What the project has to train on and score with."""
    labels, _, prog, entries = pj.gather_labels(project, queue)
    st = load_assist(project)
    hold = [k for k in st["holdout"] if k in entries]
    train_keys = [k for k in entries if prog[k]["complete"] and k not in hold]
    hold_done = [k for k in hold if prog[k]["complete"]]
    classes, sizes = {}, []
    for k in train_keys:
        e = entries[k]
        for f in labels[k]:
            c = f["properties"].get(pj.CLASS_KEY) or pj.UNCLASSIFIED
            classes[c] = classes.get(c, 0) + 1
            sizes.append(_box_size_px(f["geometry"], e["center"][1] if e.get("center") else 0,
                                      e.get("res_m") or 10.0))
    return {
        "n_images": len(entries),
        "train_images": train_keys,
        "train_boxes": sum(len(labels[k]) for k in train_keys),
        "train_empty": sum(1 for k in train_keys if not labels[k]),
        "holdout": hold, "holdout_complete": hold_done,
        "holdout_boxes": sum(len(labels[k]) for k in hold_done),
        "untouched": [k for k in entries if not _touched(prog[k]) and k not in hold],
        "classes": classes,
        "median_box_px": float(np.median(sizes)) if sizes else None,
        "versions": st["versions"],
    }


def holdout_target(n_images):
    return int(min(20, max(5, round(0.15 * n_images))))


def recommend(project, inv=None):
    """Adaptive next-step advice: {headline, steps[{name, status, detail}], ready_to_train,
    train_classes, need_boxes}. status: done / now / todo / warn."""
    inv = inv or inventory(project)
    cfg = settings(project)
    small = inv["median_box_px"] is not None and inv["median_box_px"] < 12
    need_boxes = int(SEED_BOXES * (1.5 if small else 1))
    need_images = 20 if inv["n_images"] >= 40 else max(5, inv["n_images"] // 2)
    h_need = holdout_target(inv["n_images"])
    steps = []

    # 1. hold-out
    if len(inv["holdout"]) < min(h_need, inv["n_images"] // 4 or 1):
        steps.append({"name": "Hold-out set", "status": "now",
                      "detail": f"Pick ~{h_need} hold-out images (Pick hold-out). They're labelled blind -- no "
                                f"detector or model boxes shown -- and only used to score models."})
    elif len(inv["holdout_complete"]) < len(inv["holdout"]):
        steps.append({"name": "Hold-out set", "status": "now",
                      "detail": f"Label the hold-out completely: {len(inv['holdout_complete'])}/"
                                f"{len(inv['holdout'])} images done ({inv['holdout_boxes']} objects so far). "
                                f"Do every scene check -- a missed object here makes every model look worse."})
    else:
        steps.append({"name": "Hold-out set", "status": "done",
                      "detail": f"{len(inv['holdout'])} images, {inv['holdout_boxes']} objects, all complete."
                                + ("  (Few objects: scores will be noisy; add hold-out images if you can.)"
                                   if inv["holdout_boxes"] < 30 else "")})

    # 2. seed labels
    n_img, n_box = len(inv["train_images"]), inv["train_boxes"]
    size_note = (f" Objects are small (median {inv['median_box_px']:.0f} px), so the target is 1.5x and "
                 f"tiles are upscaled.") if small else ""
    if n_box >= need_boxes and n_img >= need_images:
        steps.append({"name": "Seed labels", "status": "done",
                      "detail": f"{n_box} objects in {n_img} complete images (target {need_boxes} / {need_images})."
                                + size_note})
    else:
        steps.append({"name": "Seed labels", "status": "now" if steps[0]["status"] == "done" else "todo",
                      "detail": f"{n_box}/{need_boxes} objects in {n_img}/{need_images} complete images "
                                f"(not counting the hold-out). Use Suggest images to pick diverse ones, and "
                                f"include some images with nothing in them.{size_note}"})
    if n_img and inv["train_empty"] == 0 and n_img >= 10:
        steps.append({"name": "Empty images", "status": "warn",
                      "detail": "None of your complete images is empty -- add a few with no objects so the model "
                                "learns what isn't one (reduces false alarms)."})

    # classes
    min_pc = cfg["min_per_class"]
    named = {c: n for c, n in inv["classes"].items() if c != pj.UNCLASSIFIED}
    train_classes = sorted(c for c, n in named.items() if n >= min_pc)
    if len(train_classes) < 2:
        train_classes = []   # single-class model
    if named:
        short = {c: min_pc - n for c, n in named.items() if n < min_pc}
        steps.append({"name": "Classes", "status": "done" if not short else "warn",
                      "detail": ("Training one class per type: " + ", ".join(train_classes)
                                 + f" (+ '{OTHER_CLASS}' for everything else)") if train_classes
                      else ("Training a single class until each class has >= "
                            f"{min_pc} examples (" + ", ".join(f"{c} needs {n} more" for c, n in short.items()) + ")")})

    # 3/4. models
    versions = [r for r in inv["versions"] if r.get("status") == "done"]
    ready = n_box >= 10 and n_img >= 2
    if not versions:
        ok = n_box >= need_boxes and n_img >= need_images and steps[0]["status"] == "done"
        steps.append({"name": "First model", "status": "now" if ok else "todo",
                      "detail": "Ready: press Train model." if ok else
                      "Train once the hold-out and seed labels are done (you can train earlier, but expect "
                      "weak pre-labels)."})
        headline = next((s["detail"] for s in steps if s["status"] == "now"), "Label more images.")
    else:
        last = versions[-1]
        m = (last.get("metrics") or {}).get("best") or {}
        new_boxes = n_box - last.get("n_train_boxes", 0)
        retrain_at = max(50, int(0.25 * last.get("n_train_boxes", 0)))
        useful = m.get("r", 0) >= USEFUL_RECALL and m.get("p", 0) >= USEFUL_PRECISION
        plateau = False
        if len(versions) >= 3:
            ms = [((v.get("metrics") or {}).get("best") or {}) for v in versions[-3:]]
            plateau = all(abs(ms[i + 1].get("r", 0) - ms[i].get("r", 0)) < PLATEAU and
                          abs(ms[i + 1].get("p", 0) - ms[i].get("p", 0)) < PLATEAU for i in range(2))
        steps.append({"name": f"Model v{last['version']}", "status": "done",
                      "detail": (f"hold-out P {m.get('p', 0):.2f} / R {m.get('r', 0):.2f} / F1 {m.get('f1', 0):.2f} at "
                                 f"score >= {m.get('thr', 0):.2f}, trained on {last.get('n_train_boxes')} objects. "
                                 + ("Good enough to pre-label: Predict, then correct." if useful else
                                    "Not yet reliable as pre-labels; keep labelling.")) if m else
                      "No hold-out score (hold-out incomplete) -- complete it to measure this model."})
        if plateau:
            steps.append({"name": "Plateau", "status": "warn",
                          "detail": f"The last two retrains improved precision and recall by < {PLATEAU:.2f}. More "
                                    f"labels of the same kind are paying off little; consider more diverse images, "
                                    f"a different base model, or larger tiles/upscale."})
        steps.append({"name": "Next model", "status": "now" if new_boxes >= retrain_at else "todo",
                      "detail": f"{new_boxes}/{retrain_at} new objects since v{last['version']} -- retrain when this "
                                f"reaches the target." + ("" if inv["untouched"] else "  (No untouched images left "
                                                                                       "to pre-label.)")})
        predicted = bool(last.get("predicted"))
        if useful and inv["untouched"] and not predicted:
            headline = (f"Model v{last['version']} is good enough to pre-label (R {m.get('r', 0):.2f}, "
                        f"P {m.get('p', 0):.2f}): press Predict, then review its boxes lowest-score first.")
        elif useful:
            headline = (f"Pre-labels are on. Correct them, finish scene checks, and retrain after "
                        f"{retrain_at} new objects ({new_boxes} so far).")
        else:
            headline = next((s["detail"] for s in steps if s["status"] == "now"), steps[-1]["detail"])
    return {"headline": headline, "steps": steps, "ready_to_train": ready, "train_classes": train_classes,
            "need_boxes": need_boxes, "need_images": need_images, "holdout_target": h_need}


# -------------------------------------------------------------- selection ---

def _image_features(project, keys):
    """Cheap per-image descriptor for diversity sampling: band means/stds of a 32-px thumbnail,
    date (fraction of the project's span), and site one-hot."""
    entries = {pj.image_key(e): e for e in project.imagery}
    sites = sorted({entries[k]["site"] for k in keys})
    dates = [entries[k].get("date") for k in keys if entries[k].get("date")]
    d0 = min(dates) if dates else None
    span = max(1, (datetime.fromisoformat(max(dates)) - datetime.fromisoformat(d0)).days) if dates else 1
    feats = []
    for k in keys:
        e = entries[k]
        img = pj.render_thumbnail(e["path"], 32, project.settings.get("rgb_bands", "auto"))
        a = np.asarray(img, dtype=np.float32) / 255 if img is not None else np.zeros((1, 1, 3))
        f = list(a.reshape(-1, 3).mean(0)) + list(a.reshape(-1, 3).std(0))
        f.append((datetime.fromisoformat(e["date"]) - datetime.fromisoformat(d0)).days / span
                 if e.get("date") and d0 else 0.5)
        f += [1.0 if e["site"] == s else 0.0 for s in sites]
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def _k_center(feats, n, seed=0):
    """Greedy farthest-point selection: each pick is the image least like anything picked so far."""
    if n <= 0:
        return []
    if len(feats) <= n:
        return list(range(len(feats)))
    rng = random.Random(seed)
    picked = [rng.randrange(len(feats))]
    d = np.linalg.norm(feats - feats[picked[0]], axis=1)
    while len(picked) < n:
        i = int(np.argmax(d))
        picked.append(i)
        d = np.minimum(d, np.linalg.norm(feats - feats[i], axis=1))
    return picked


def pick_holdout(project, n=None, seed=1):
    """Diverse hold-out drawn from untouched images (nobody has seen boxes on them yet)."""
    inv = inventory(project)
    n = n or holdout_target(inv["n_images"])
    st = load_assist(project)
    keep = [k for k in st["holdout"] if k in {pj.image_key(e) for e in project.imagery}]
    pool = [k for k in inv["untouched"] if k not in keep]
    want = max(0, n - len(keep))
    picks = [pool[i] for i in _k_center(_image_features(project, pool), want, seed)] if pool and want else []
    st["holdout"] = keep + picks
    st["suggested"] = [k for k in st["suggested"] if k not in st["holdout"]]
    save_assist(project, st)
    return st["holdout"]


def suggest_images(project, n=20, seed=0):
    """Next images to label. Before any model: most diverse untouched images. With a model's
    predictions: images with the most uncertain predictions (score 0.25-0.6), diversity as tiebreak."""
    inv = inventory(project)
    pool = inv["untouched"]
    if not pool:
        return []
    st = load_assist(project)
    entries = {pj.image_key(e): e for e in project.imagery}
    unc = {}
    pre = st.get("prelabels", {})
    for k in pool:
        for gj in [pre[k]] if k in pre else [d for d in entries[k].get("detections") or [] if is_model_file(d)]:
            try:
                for f in json.loads(Path(gj).read_text())["features"]:
                    s = f["properties"].get("score") or 0
                    unc[k] = unc.get(k, 0) + (1 if 0.25 <= s <= 0.6 else 0)
            except (OSError, ValueError, KeyError):
                pass
    feats = _image_features(project, pool)
    if unc:
        order = sorted(range(len(pool)), key=lambda i: -unc.get(pool[i], 0))
        top = [i for i in order if unc.get(pool[i], 0) > 0][: n * 2]
        div = [top[j] for j in _k_center(feats[top], min(n, len(top)), seed)] if top else []
        rest = [i for i in _k_center(feats, n, seed) if i not in div]
        picks = (div + rest)[:n]
    else:
        picks = _k_center(feats, n, seed)
    st["suggested"] = [pool[i] for i in picks]
    save_assist(project, st)
    return st["suggested"]


# ---------------------------------------------------------------- imagery ---

class RasterReader:
    """Reads an image exactly as the review canvas shows it (bands + whole-image stretch)."""

    def __init__(self, path, rgb_bands="auto"):
        self.src = rasterio.open(path)
        self.bands = pj.pick_bands(self.src, rgb_bands)
        self.db = pj._is_db(self.src, self.bands)
        self.limits = pj.image_limits(self.src, self.bands, self.db)

    def read(self, col, row, w, h):
        arr = self.src.read(self.bands, window=Window(col, row, w, h), boundless=True, fill_value=0)
        u8 = pj.stretch(arr, grayscale_db=self.db, limits=self.limits)
        if u8.shape[0] == 1:
            u8 = np.repeat(u8, 3, axis=0)
        return np.ascontiguousarray(np.transpose(u8[:3], (1, 2, 0)))

    def close(self):
        self.src.close()


def tile_origins(width, height, tile, overlap):
    step = max(1, tile - overlap)
    xs = list(range(0, max(1, width - tile) + 1, step))
    ys = list(range(0, max(1, height - tile) + 1, step))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    return [(x, y) for y in ys for x in xs]


def boxes_px(src, feats):
    """Lon/lat label polygons -> native pixel boxes (c0, r0, c1, r1, class) in this raster."""
    to = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    inv = ~src.transform
    out = []
    for f in feats:
        pts = [inv * to.transform(lon, lat) for lon, lat in f["geometry"]["coordinates"][0][:4]]
        cs = [p[0] for p in pts]
        rs = [p[1] for p in pts]
        out.append((min(cs), min(rs), max(cs), max(rs), f["properties"].get(pj.CLASS_KEY) or pj.UNCLASSIFIED))
    return out


def auto_upscale(median_box_px):
    if not median_box_px:
        return 1
    return int(min(4, max(1, math.ceil(16 / median_box_px))))


def net_size(tile, upscale):
    return int(min(1280, max(320, round(tile * upscale / 32) * 32)))


# ---------------------------------------------------------------- dataset ---

def build_dataset(project, out_dir, class_names, tile, overlap, val_frac=0.15, seed=0, log=print):
    """YOLO tiles from complete non-hold-out images. Validation split is by IMAGE (tiles of
    one image never straddle train/val). Empty tiles are kept up to 1:1 with object tiles."""
    labels, _, prog, entries = pj.gather_labels(project)
    hold = set(load_assist(project)["holdout"])
    keys = [k for k in entries if prog[k]["complete"] and k not in hold and Path(entries[k]["path"]).exists()]
    if not keys:
        raise RuntimeError("no complete (non-hold-out) images to train on")
    rng = random.Random(seed)
    keys_shuf = keys[:]
    rng.shuffle(keys_shuf)
    n_val = max(1, round(val_frac * len(keys))) if len(keys) >= 4 else 0
    val_keys = set(keys_shuf[:n_val])
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts = {"train": [0, 0], "val": [0, 0]}   # [object tiles, empty tiles]
    empties = {"train": [], "val": []}
    single = len(class_names) == 1
    for k in keys:
        split = "val" if k in val_keys else "train"
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        rd = RasterReader(entries[k]["path"], project.settings.get("rgb_bands", "auto"))
        try:
            bxs = boxes_px(rd.src, labels[k])
            W, H = rd.src.width, rd.src.height
            for (x, y) in tile_origins(W, H, tile, overlap):
                lines = []
                for c0, r0, c1, r1, cls in bxs:
                    ic0, ir0, ic1, ir1 = max(c0, x), max(r0, y), min(c1, x + tile), min(r1, y + tile)
                    if ic1 - ic0 <= 0 or ir1 - ir0 <= 0:
                        continue
                    if (ic1 - ic0) * (ir1 - ir0) < 0.5 * max((c1 - c0) * (r1 - r0), 1e-6):
                        continue   # mostly outside this tile: the neighbouring tile has it
                    # multi-class: a type with too few examples, or no type, goes to the catch-all
                    # last class -- dropping it would teach the model that it's background
                    ci = 0 if single else (class_names.index(cls) if cls in class_names else len(class_names) - 1)
                    lines.append(f"{ci} {((ic0 + ic1) / 2 - x) / tile:.6f} {((ir0 + ir1) / 2 - y) / tile:.6f} "
                                 f"{(ic1 - ic0) / tile:.6f} {(ir1 - ir0) / tile:.6f}")
                stem = f"{pj.slugify(k)}__{x}_{y}"
                if not lines:
                    empties[split].append((k, x, y, stem))
                    continue
                Image.fromarray(rd.read(x, y, tile, tile)).save(out_dir / split / "images" / f"{stem}.png")
                (out_dir / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                counts[split][0] += 1
        finally:
            rd.close()
    for split in ("train", "val"):
        pool = empties[split]
        rng.shuffle(pool)
        for k, x, y, stem in pool[: max(counts[split][0], 2)]:
            rd = RasterReader(entries[k]["path"], project.settings.get("rgb_bands", "auto"))
            try:
                Image.fromarray(rd.read(x, y, tile, tile)).save(out_dir / split / "images" / f"{stem}.png")
            finally:
                rd.close()
            (out_dir / split / "labels" / f"{stem}.txt").write_text("")
            counts[split][1] += 1
    if counts["val"][0] + counts["val"][1] == 0:   # too few images to hold any out: validate on train
        val_line = "val: train/images"
    else:
        val_line = "val: val/images"
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: train/images\n{val_line}\nnames:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names)))
    log(f"dataset: {len(keys)} images ({len(val_keys)} val) -> train {counts['train'][0]} object tiles + "
        f"{counts['train'][1]} empty, val {counts['val'][0]} + {counts['val'][1]}")
    return out_dir / "data.yaml", {"images": len(keys), "val_images": len(val_keys), "tiles": counts}


# ---------------------------------------------------------------- predict ---

def nms(boxes, scores, iou):
    if not len(boxes):
        return []
    b = np.asarray(boxes, dtype=np.float32)
    order = np.argsort(-np.asarray(scores))
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(b[i, 0], b[order[1:], 0])
        yy0 = np.maximum(b[i, 1], b[order[1:], 1])
        xx1 = np.minimum(b[i, 2], b[order[1:], 2])
        yy1 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        ov = inter / np.maximum(area[i] + area[order[1:]] - inter, 1e-9)
        order = order[1:][ov < iou]
    return keep


def predict_image(model, path, cfg, rgb_bands="auto", conf=0.05, batch=8):
    """Tiled inference with a backend predictor -> [(c0, r0, c1, r1, score, class_index)] in
    native pixels. Tiles are passed as RGB uint8 arrays; backends convert if they need to."""
    rd = RasterReader(path, rgb_bands)
    try:
        W, H = rd.src.width, rd.src.height
        origins = tile_origins(W, H, cfg["tile"], cfg["overlap"])
        boxes, scores, classes = [], [], []
        for i in range(0, len(origins), batch):
            chunk = origins[i:i + batch]
            tiles = [rd.read(x, y, cfg["tile"], cfg["tile"]) for x, y in chunk]
            for (x, y), (bxs, scs, cls) in zip(chunk, model.predict(tiles, imgsz=cfg["imgsz"], conf=conf)):
                for (b0, b1, b2, b3), s, c in zip(np.asarray(bxs).reshape(-1, 4), np.asarray(scs).ravel(),
                                                  np.asarray(cls).ravel()):
                    boxes.append((b0 + x, b1 + y, b2 + x, b3 + y))
                    scores.append(float(s))
                    classes.append(int(c))
        keep = nms(boxes, scores, NMS_IOU)
        return [(*boxes[i], scores[i], classes[i]) for i in keep]
    finally:
        rd.close()


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def match(preds, gts, thr):
    """Greedy by score at IoU >= MATCH_IOU -> (tp, fp, fn) for predictions scoring >= thr."""
    used = set()
    tp = fp = 0
    for p in sorted([p for p in preds if p[4] >= thr], key=lambda p: -p[4]):
        best, bi = 0.0, None
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = _iou(p, g)
            if v > best:
                best, bi = v, j
        if bi is not None and best >= MATCH_IOU:
            used.add(bi)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gts) - len(used)


def evaluate(project, model, cfg, log=print):
    """Score on complete hold-out images: P/R/F1 at the default threshold and the best-F1
    threshold, plus per-image count correlation at that threshold."""
    labels, _, prog, entries = pj.gather_labels(project)
    keys = [k for k in load_assist(project)["holdout"] if k in entries and prog[k]["complete"]]
    missing = [k for k in keys if not Path(entries[k]["path"]).exists()]
    if missing:
        log(f"evaluate: {len(missing)} hold-out image file(s) not found, skipped: " + ", ".join(missing[:5]))
        keys = [k for k in keys if k not in missing]
    if not keys:
        log("evaluate: no complete hold-out images -- skipped")
        return None
    per = []
    for k in keys:
        with rasterio.open(entries[k]["path"]) as src:
            gts = [b[:4] for b in boxes_px(src, labels[k])]
        preds = predict_image(model, entries[k]["path"], cfg, project.settings.get("rgb_bands", "auto"))
        per.append((preds, gts))

    def at(thr):
        tp = fp = fn = 0
        counts = []
        for preds, gts in per:
            a, b, c = match(preds, gts, thr)
            tp, fp, fn = tp + a, fp + b, fn + c
            counts.append((sum(1 for p in preds if p[4] >= thr), len(gts)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"thr": round(thr, 2), "p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3),
                "tp": tp, "fp": fp, "fn": fn, "counts": counts}
    curve = [at(t) for t in np.arange(0.05, 0.951, 0.05)]
    best = max(curve, key=lambda m: (m["f1"], m["r"]))
    pc, gc = zip(*best["counts"]) if best["counts"] else ((), ())
    count_r = float(np.corrcoef(pc, gc)[0, 1]) if len(pc) > 2 and np.std(pc) > 0 and np.std(gc) > 0 else None
    default = min(curve, key=lambda m: abs(m["thr"] - cfg["conf"]))
    metrics = {"n_images": len(keys), "n_objects": sum(len(g) for _, g in per),
               "default": {k: default[k] for k in ("thr", "p", "r", "f1")},
               "best": {k: best[k] for k in ("thr", "p", "r", "f1", "tp", "fp", "fn")},
               "count_r": None if count_r is None else round(count_r, 3),
               "curve": [{k: m[k] for k in ("thr", "p", "r", "f1")} for m in curve]}
    log(f"hold-out: {metrics['n_objects']} objects in {len(keys)} images -> best F1 {best['f1']:.2f} "
        f"(P {best['p']:.2f} R {best['r']:.2f} at score >= {best['thr']:.2f})")
    return metrics


def write_predictions(project, v, model, cfg, class_names, log=print, progress=None):
    """Predict on untouched non-hold-out images and attach the files as a detection source."""
    inv = inventory(project)
    entries = {pj.image_key(e): e for e in project.imagery}
    todo = inv["untouched"]
    out = version_dir(project, v) / "detections"
    n_det = 0
    written = {}
    for i, k in enumerate(todo):
        if progress:
            progress(i, len(todo))
        e = entries[k]
        if not Path(e["path"]).exists():
            continue
        preds = predict_image(model, e["path"], cfg, project.settings.get("rgb_bands", "auto"), conf=cfg["conf"])
        with rasterio.open(e["path"]) as src:
            to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            feats = []
            for c0, r0, c1, r1, s, ci in preds:
                ring = [list(to_ll.transform(*(src.transform * (c, r))))
                        for c, r in ((c0, r0), (c1, r0), (c1, r1), (c0, r1))]
                feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
                              "properties": {"site": e["site"], "item_id": e["item_id"], "date": e.get("date"),
                                             "source": f"model_v{v}", "score": round(s, 4),
                                             "pred_class": class_names[ci] if len(class_names) > 1 else None}})
        path = out / e["site"] / f"{e['item_id']}_model.geojson"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        written[k] = str(path)
        n_det += len(feats)
    # Record in assist.json (re-read right before saving, so GUI changes made meanwhile survive).
    # project.json is NOT rewritten: the GUI may save it concurrently and would drop the change.
    st = load_assist(project)
    st.setdefault("prelabels", {}).update(written)
    save_assist(project, st)
    log(f"predicted {n_det} objects on {len(todo)} untouched images (source model_v{v}, score >= {cfg['conf']})")
    return len(todo), n_det


# ------------------------------------------------------------- train (CLI) ---

def run_train(project_dir, v):
    project = pj.Project(project_dir)
    cfg_all = settings(project)
    vdir = version_dir(project, v)
    vdir.mkdir(parents=True, exist_ok=True)
    logf = open(vdir / "log.txt", "a")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    st = load_assist(project)
    try:
        backend = backends.get_backend(cfg_all.get("backend"))
    except backends.BackendMissing as exc:
        write_status(project, v, stage="failed", message="no training add-on installed")
        log("FAILED: " + str(exc))
        logf.close()
        return
    base = cfg_all["base_weights"] or backend.default_weights
    rec = {"version": v, "created": pj.now_iso(), "status": "running", "backend": backend.name, "base": base}
    st["versions"] = [r for r in st["versions"] if r["version"] != v] + [rec]
    save_assist(project, st)
    try:
        write_status(project, v, stage="dataset", message="building tiles")
        inv = inventory(project)
        adv = recommend(project, inv)
        class_names = (adv["train_classes"] + [OTHER_CLASS]) if adv["train_classes"] else [project.object_name]
        tile = int(cfg_all["tile_px"])
        up = auto_upscale(inv["median_box_px"]) if cfg_all["upscale"] == "auto" else int(cfg_all["upscale"])
        imgsz = net_size(tile, up)
        cfg = {"tile": tile, "overlap": int(cfg_all["tile_overlap"]), "imgsz": imgsz, "conf": float(cfg_all["conf"])}
        log(f"v{v}: {inv['train_boxes']} objects in {len(inv['train_images'])} images; classes {class_names}; "
            f"median box {inv['median_box_px'] or 0:.1f} px -> tile {tile} px at {imgsz} px (x{up})")
        data_yaml, ds = build_dataset(project, vdir / "dataset", class_names, tile, cfg["overlap"], log=log)

        epochs = int(cfg_all["epochs"])
        batch = 8 if imgsz <= 640 else 4

        def on_epoch(epoch, total):
            write_status(project, v, stage="training", epoch=int(epoch), epochs=int(total),
                         message=f"epoch {int(epoch)}/{int(total)}")
        write_status(project, v, stage="training", epoch=0, epochs=epochs, message="starting")
        log(f"training with {backend.label}: base {base}, {epochs} epochs, batch {batch}")
        weights = backend.train(data_yaml=str(data_yaml), base_weights=base, epochs=epochs, imgsz=imgsz,
                                batch=batch, out_dir=str(vdir), on_epoch=on_epoch)
        model = backend.load(str(weights))
        write_status(project, v, stage="evaluating", message="scoring on the hold-out")
        metrics = evaluate(project, model, cfg, log=log)
        rec.update({"status": "done", "finished": pj.now_iso(), "weights": str(weights), "classes": class_names,
                    "cfg": cfg, "n_train_images": len(inv["train_images"]), "n_train_boxes": inv["train_boxes"],
                    "dataset": ds, "metrics": metrics})
        if metrics:
            pj.write_json_atomic(vdir / "metrics.json", metrics)
        write_status(project, v, stage="done", message="trained" + (
            f": hold-out F1 {metrics['best']['f1']:.2f}" if metrics else " (no hold-out score)"))
        log("done")
    except Exception as exc:
        import traceback
        log("FAILED: " + "".join(traceback.format_exception(exc)))
        rec.update({"status": "failed", "error": f"{exc.__class__.__name__}: {exc}"})
        write_status(project, v, stage="failed", message=f"{exc.__class__.__name__}: {exc}")
    finally:
        st = load_assist(project)
        st["versions"] = [r for r in st["versions"] if r["version"] != v] + [rec]
        st["versions"].sort(key=lambda r: r["version"])
        save_assist(project, st)
        logf.close()


def run_predict(project_dir, v):
    project = pj.Project(project_dir)
    st = load_assist(project)
    rec = next((r for r in st["versions"] if r["version"] == v and r.get("status") == "done"), None)
    vdir = version_dir(project, v)
    logf = open(vdir / "log.txt", "a")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    try:
        if rec is None:
            raise RuntimeError(f"model v{v} isn't trained")
        model = backends.get_backend(rec.get("backend")).load(rec["weights"])
        write_status(project, v, stage="predicting", message="starting")

        def prog(i, n):
            write_status(project, v, stage="predicting", done=i, total=n, message=f"image {i + 1}/{n}")
        n_img, n_det = write_predictions(project, v, model, rec["cfg"], rec["classes"], log=log, progress=prog)
        rec["predicted"] = {"at": pj.now_iso(), "images": n_img, "objects": n_det}
        st = load_assist(project)   # re-read: write_predictions just added the pre-label files
        st["versions"] = [r if r["version"] != v else rec for r in st["versions"]]
        save_assist(project, st)
        write_status(project, v, stage="done", message=f"pre-labelled {n_img} images ({n_det} objects)")
    except Exception as exc:
        import traceback
        log("FAILED: " + "".join(traceback.format_exception(exc)))
        write_status(project, v, stage="failed", message=f"{exc.__class__.__name__}: {exc}")
    finally:
        logf.close()


def _safe_tmpdir():
    """Dataloader worker processes (e.g. PyTorch) talk over AF_UNIX sockets created in the temp
    dir; a temp path longer than ~100 bytes makes them die with 'AF_UNIX path too long' and
    training then hangs. On POSIX, fall back to /tmp when the inherited temp dir is long."""
    import os
    import tempfile
    if os.name == "posix" and len(tempfile.gettempdir()) > 60 and Path("/tmp").is_dir():
        os.environ["TMPDIR"] = "/tmp"
        tempfile.tempdir = None


def main():
    _safe_tmpdir()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "predict"])
    ap.add_argument("project")
    ap.add_argument("version", type=int)
    a = ap.parse_args()
    (run_train if a.command == "train" else run_predict)(a.project, a.version)


if __name__ == "__main__":
    main()
