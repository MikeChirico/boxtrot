"""Synthetic data and a dependency-free "oracle" backend, for tests, demos and the quickstart.

Nothing here is used by the application at runtime.

  make_scene_set(root, n)      georeferenced 4-band scenes (red, green, blue, nir) with bright
                               elongated objects on dark water, a land strip, known truth boxes
  make_detections(dir, truth)  a simulated detector's GeoJSON output for those scenes: misses,
                               false alarms, scores, two "sources" -- so there is real review work
  label_completely(project, image_key, truth)   label an image as a careful reviewer would
  OracleBackend                a training backend with no ML: "training" writes a marker file and
                               the predictor finds bright blobs, so the whole assisted-labelling
                               pipeline (tiling, georeferencing, NMS, scoring, pre-labels) can be
                               tested in seconds. Use with BOXTROT_BACKEND=boxtrot.testing:OracleBackend
"""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from .core import project as pj

UTM18N = "EPSG:32618"


def make_scene(path, rng, n_objects, size=384, origin=(770000.0, 2060000.0), res=10.0):
    """Write one scene; returns (truth pixel boxes [(c0, r0, c1, r1)], transform)."""
    water = rng.normal(300, 25, (4, size, size))
    water[3] *= 0.5                                         # water is dark in NIR
    land_w = size // 6
    water[:3, :, :land_w] = rng.normal(1400, 200, (3, size, land_w))
    water[3, :, :land_w] = rng.normal(2600, 300, (size, land_w))   # vegetation: bright NIR
    boxes = []
    while len(boxes) < n_objects:
        w, h = int(rng.integers(3, 5)), int(rng.integers(8, 12))
        if rng.random() < 0.5:
            w, h = h, w
        c, r = int(rng.integers(land_w + 10, size - w - 10)), int(rng.integers(10, size - h - 10))
        if any(not (c + w + 4 < b[0] or b[2] + 4 < c or r + h + 4 < b[1] or b[3] + 4 < r) for b in boxes):
            continue
        water[:, r:r + h, c:c + w] = rng.normal(2600, 150, (4, h, w))
        boxes.append((c, r, c + w, r + h))
    tfm = from_origin(origin[0], origin[1], res, res)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=4, dtype="uint16", crs=UTM18N,
                       transform=tfm, compress="deflate") as dst:
        dst.write(water.clip(1, 65535).astype(np.uint16))
        dst.descriptions = ("red", "green", "blue", "nir")
    return boxes, tfm


def make_scene_set(root, n=30, seed=0, areas=("north_harbour", "south_harbour"), empty_every=5):
    """n scenes spread over areas and dates: root/<area>/scene_YYYYMMDD.tif.
    Returns truth {item_id: (boxes, transform)}; every empty_every-th scene has no objects."""
    rng = np.random.default_rng(seed)
    truth = {}
    d0 = date(2025, 1, 1)
    for i in range(n):
        area = areas[i % len(areas)]
        day = d0 + timedelta(days=12 * i)
        stem = f"scene_{day:%Y%m%d}"
        n_obj = 0 if i % empty_every == 0 else int(rng.integers(2, 7))
        origin = (770000.0 + 20000 * areas.index(area), 2060000.0)
        truth[stem] = make_scene(Path(root) / area / f"{stem}.tif", rng, n_obj, origin=origin)
    return truth


def make_detections(det_root, img_root, truth, seed=1, miss=0.2, false_alarms=1):
    """Simulated detector output: each true object found with probability 1-miss (score 0.5-0.98),
    plus false alarms (score 0.25-0.6), split between sources "detector_a"/"detector_b"."""
    rng = np.random.default_rng(seed)
    to_ll = Transformer.from_crs(UTM18N, "EPSG:4326", always_xy=True)
    n = 0
    for tif in sorted(Path(img_root).rglob("*.tif")):
        boxes, tfm = truth[tif.stem]
        feats = []
        cands = [(b, True) for b in boxes if rng.random() > miss]
        for _ in range(int(rng.poisson(false_alarms))):
            c, r = int(rng.integers(70, 370)), int(rng.integers(10, 370))
            cands.append(((c, r, c + 4, r + 4), False))
        for (c0, r0, c1, r1), real in cands:
            lon, lat = to_ll.transform(*(tfm * ((c0 + c1) / 2, (r0 + r1) / 2)))
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                          "properties": {"score": round(float(rng.uniform(0.5, 0.98) if real else rng.uniform(0.25, 0.6)), 3),
                                         "source": "detector_a" if rng.random() < 0.6 else "detector_b",
                                         "bbox_w_px": int(c1 - c0), "bbox_h_px": int(r1 - r0)}})
        out = Path(det_root) / tif.parent.name / f"{tif.stem}_detections.geojson"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1))
        n += len(feats)
    return n


def label_completely(project, image_key, truth):
    """Draw every true object as an added box and mark the image's scene check Valid."""
    e = next(x for x in project.imagery if pj.image_key(x) == image_key)
    boxes, tfm = truth[e["item_id"]]
    to_ll = Transformer.from_crs(UTM18N, "EPSG:4326", always_xy=True)
    for c0, r0, c1, r1 in boxes:
        ring = [list(to_ll.transform(*(tfm * (c, r)))) for c, r in ((c0, r0), (c1, r0), (c1, r1), (c0, r1))]
        project.additions["features"].append({
            "type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
            "properties": {"site": e["site"], "item_id": e["item_id"], "date": e["date"], "added_at": pj.now_iso()}})
    scene = next(it for it in pj.build_queue(project) if it["image_key"] == image_key and it["kind"] == "scene")
    project.state[scene["id"]] = {"status": "valid"}
    project.save_state()
    project.save_additions()


# ------------------------------------------------------------ oracle backend ---

def _blobs(gray, thr):
    """4-connected components of gray > thr -> list of (c0, r0, c1, r1). Pure numpy/python."""
    mask = gray > thr
    seen = np.zeros_like(mask)
    out = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        stack, rs, cs = [(r0, c0)], [], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            rs.append(r)
            cs.append(c)
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if len(rs) >= 12:   # synthetic objects are 24-44 px; smaller specks are stretched land noise
            out.append((min(cs), min(rs), max(cs) + 1, max(rs) + 1))
    return out


class OracleBackend:
    """Test backend: no learning; finds bright compact blobs (the synthetic objects)."""
    name = "oracle"
    label = "Oracle (test backend, no ML)"
    default_weights = "none"

    def train(self, data_yaml, base_weights, epochs, imgsz, batch, out_dir, on_epoch=None):
        for e in range(1, int(epochs) + 1):
            if on_epoch:
                on_epoch(e, int(epochs))
        w = Path(out_dir) / "oracle_weights.json"
        w.write_text(json.dumps({"data": str(data_yaml)}))
        return w

    def load(self, weights):
        return self

    def predict(self, tiles, imgsz, conf):
        out = []
        for t in tiles:
            g = t.astype(np.float32).mean(axis=2)
            bx = [b for b in _blobs(g, 240) if (b[2] - b[0]) * (b[3] - b[1]) <= 400]   # skip land
            out.append((np.array(bx, dtype=np.float32).reshape(-1, 4), np.full(len(bx), 0.9), np.zeros(len(bx))))
        return out
