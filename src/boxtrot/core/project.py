"""
Project model: everything that isn't GUI -- project create/open/save, imagery scanning and
detection matching, the review queue, view rendering, statistics and label export.

A project is a folder:
    <project>/project.json                     name, settings, the selected imagery
    <project>/review_data/review_state.json    one record per reviewed item
    <project>/review_data/manual_additions.geojson   boxes drawn for objects the detector missed
    <project>/review_data/history.jsonl        append-only log of every action
    <project>/models/                          model-assisted labelling (core/assist.py)
    <project>/exports/<timestamp>/             label exports

Inputs:
  - imagery: any raster GDAL can read with georeferencing (GeoTIFF, COG, JP2, ...). Projected or
    geographic CRS; geographic rasters are viewed in the local UTM zone.
  - detections (optional): GeoJSON FeatureCollections, matched to images by file name (see
    normalise_key) and area folder. Points, boxes/polygons, lines -- any geometry. Optional
    feature properties: score, source, bbox_w_px/bbox_h_px (box size in image pixels for points),
    kind="scene" + view_radius_m (a whole-area check item).

Caveats -- read before trusting the output:
  - The "area" of an image is its first sub-folder under the scanned folder (or the folder's own
    name); review ids embed it, so keep the folder layout stable once you start reviewing.
  - "0 detections" on an image usually means the detection file name didn't match the image.
  - Auto RGB: band names (red/green/blue, or Sentinel-2 B04/B03/B02) if present, else 1-3 for
    8-bit, else 3,2,1. Override per project in Settings if colours look wrong.
  - Only "complete" images (every detection decided + scene check done) are safe to train on:
    an unchecked image silently contains unlabelled objects.
"""
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .. import PROJECT_FORMAT_VERSION
from ..config import user_config_dir

RECENT_PATH = None   # None = <user config dir>/recent_projects.json (tests point this elsewhere)

CROP_RADIUS_M = 150          # default detection view radius (m)
DEFAULT_SETTINGS = {
    "crop_radius_m": CROP_RADIUS_M,
    "canvas_size": 680,
    "scene_checks": True,        # add a whole-image scene check for images that have none
    "rgb_bands": "auto",         # "auto", or e.g. "3,2,1" / "1,2,3" / "1" (grayscale)
    "stretch": "image",          # "image" = whole-image contrast (stable), "view" = per view
    "lenient_edited": True,      # count an edited-but-unmarked detection as a object on export
    "edit_marks_valid": True,    # drawing/resizing a detection's box also marks it Valid
    "object_name": "object",     # what is being labelled ("ship", "building", ...): UI wording, model class
    "classes": [],               # optional object types, number keys 1..9
    "holdout_blind": True,       # model-assist hold-out images show no detections (unbiased scoring)
}
UNCLASSIFIED = "unclassified"
CLASS_KEY = "class"          # object type on review records / added boxes / exported labels
IMAGE_EXTS = (".tif", ".tiff", ".jp2", ".vrt", ".img")   # anything else GDAL reads: add it here
S2_ID_RE = re.compile(r"S2[ABCD]_\d{2}[A-Z]{3}_\d{8}_\d+_L[12][AC]")
STRIP_SUFFIXES = ("_detections", "_dets", "_predictions", "_preds", "_labels", "_docked", "_visual", "_vv_sigma0_geocoded", "_vh_sigma0_geocoded",
                  "_sigma0_geocoded", "_sigma0")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, obj):
    """Write-then-rename, so a crash mid-save can never leave a truncated state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def slugify(name):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")
    return s or "project"


# ------------------------------------------------------------ recent list ---

def _recent_path():
    return Path(RECENT_PATH) if RECENT_PATH else user_config_dir() / "recent_projects.json"


def load_recent():
    try:
        return [p for p in json.loads(_recent_path().read_text()) if isinstance(p, str)]
    except (OSError, ValueError):
        return []


def add_recent(path):
    path = str(Path(path).resolve())  # absolute: a relative path breaks from another launch dir
    recent = [path] + [p for p in load_recent() if p != path]
    write_json_atomic(_recent_path(), recent[:20])


def remove_recent(path):
    write_json_atomic(_recent_path(), [p for p in load_recent() if p != str(path)])


# ------------------------------------------------------- name/site parsing ---

def normalise_key(name):
    """'harbour_2026-03_S2B_18QYG_20260303_0_L2A_visual' -> 'S2B_18QYG_20260303_0_L2A' (Sentinel-2 ids are extracted)."""
    m = S2_ID_RE.search(name)
    if m:
        return m.group(0)
    for suf in STRIP_SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name


def parse_date(name):
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", name)
    if not m:
        return None
    y, mo, d = m.groups()
    if not (1 <= int(mo) <= 12 and 1 <= int(d) <= 31):
        return None
    return f"{y}-{mo}-{d}"


def infer_site(path, root):
    """The image's "area": its first sub-folder below the scanned folder, else that folder's name.
    (e.g. scanning imagery/ finds imagery/harbour_north/2024/img.tif -> "harbour_north")."""
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        rel = path.relative_to(root).parts
    except ValueError:
        rel = ()
    return rel[0] if len(rel) > 1 else (root.name or "default")


# --------------------------------------------------------- imagery scanning ---

def metric_crs(src):
    """CRS to measure/view this raster in: its own if projected, else the local UTM zone
    (a raster georeferenced in degrees, e.g. EPSG:4326, can't be windowed by metres)."""
    if not src.crs.is_geographic:
        return src.crs
    b = src.bounds
    lon, lat = (b.left + b.right) / 2, (b.bottom + b.top) / 2
    zone = int((lon + 180) / 6) + 1
    return rasterio.crs.CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def raster_info(path):
    """Footprint + centre + band layout for one raster; error string if unusable.
    res_m / extent_m are always metres (degree-based rasters are measured in local UTM)."""
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return {"error": "no CRS (not geocoded)"}
            b = src.bounds
            tfm = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            clon, clat = tfm.transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)
            mcrs = metric_crs(src)
            mb = transform_bounds(src.crs, mcrs, *b) if mcrs != src.crs else tuple(b)
            extent = [mb[2] - mb[0], mb[3] - mb[1]]
            return {
                "width": src.width, "height": src.height, "count": src.count, "dtype": src.dtypes[0],
                "res_m": float(extent[0] / src.width), "crs": src.crs.to_string(),
                "center": [clon, clat], "extent_m": extent,
                "descriptions": [d or "" for d in src.descriptions],
            }
    except Exception as exc:  # unreadable file: report, don't crash the scan
        return {"error": f"unreadable: {exc.__class__.__name__}"}


def index_detections(det_dir):
    """{(site, key): [geojson paths]} and {key: [...]} for every geojson under det_dir."""
    by_site, by_key = {}, {}
    if not det_dir:
        return by_site, by_key
    for gj in sorted(Path(det_dir).rglob("*.geojson")):
        try:
            fc = json.loads(gj.read_text())
        except (OSError, ValueError):
            continue
        feats = fc.get("features", [])
        # site/item_id may be on each feature or on the FeatureCollection itself
        props = {**{k: fc[k] for k in ("site", "item_id") if k in fc}, **(feats[0]["properties"] if feats else {})}
        key = normalise_key(props.get("item_id") or gj.stem)
        site = props.get("site") or infer_site(gj, det_dir)
        by_site.setdefault((site, key), []).append(str(gj))
        by_key.setdefault(key, []).append(str(gj))
    return by_site, by_key


def scan_imagery(image_dir, det_dir=None, progress=None):
    """Candidate imagery entries for the Add Imagery dialog (nothing is added yet)."""
    by_site, by_key = index_detections(det_dir)
    files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    out = []
    for i, p in enumerate(files):
        if progress:
            progress(i, len(files))
        info = raster_info(p)
        key = normalise_key(p.stem)
        site = infer_site(p, image_dir)
        dets = by_site.get((site, key)) or (by_key.get(key) if len(by_key.get(key, [])) == 1 else None) or []
        n_det = 0
        for d in dets:
            try:
                n_det += sum(1 for f in json.loads(Path(d).read_text())["features"]
                             if f["properties"].get("kind") != "scene")
            except (OSError, ValueError, KeyError):
                pass
        out.append({
            "path": str(p.resolve()), "site": site, "item_id": key, "date": parse_date(p.stem),
            "detections": [str(Path(d).resolve()) for d in dets], "n_detections": n_det, **info,
        })
    return out


# ------------------------------------------------------------- rendering ---

BAND_ALIASES = {
    "red": ("red", "b04", "b4"), "green": ("green", "b03", "b3"), "blue": ("blue", "b02", "b2"),
    "nir": ("nir", "nir08", "b08", "b8", "b8a"), "swir16": ("swir16", "swir1", "b11"),
    "swir22": ("swir22", "swir2", "b12"),
}
# band view modes, cycled with b in the review screen; each needs the named band roles
VIEW_MODES = {
    "true": ("True colour", ("red", "green", "blue")),
    "nir": ("False colour (NIR, red, green): vegetation red, hulls/metal bright", ("nir", "red", "green")),
    "swir": ("SWIR (B12, NIR, red): haze-cutting, metal/concrete bright", ("swir22", "nir", "red")),
    "ndwi": ("Water index NDWI: water bright, objects/land dark", ("green", "nir")),
}


def band_roles(src):
    """{role: 1-based band index} from band descriptions (S2 names B04 etc. or red/green/...)."""
    desc = [(d or "").lower() for d in src.descriptions]
    roles = {}
    for role, aliases in BAND_ALIASES.items():
        for a in aliases:
            if a in desc:
                roles[role] = desc.index(a) + 1
                break
    return roles


def available_views(path):
    """View modes this raster supports (always includes "true")."""
    try:
        with rasterio.open(path) as src:
            roles = band_roles(src)
    except Exception:
        return ["true"]
    return ["true"] + [k for k, (_, need) in VIEW_MODES.items() if k != "true" and all(r in roles for r in need)]


def pick_bands(src, override="auto"):
    if override and override != "auto":
        idx = [int(x) for x in re.split(r"[,\s]+", override.strip()) if x]
        return [i for i in idx if 1 <= i <= src.count] or [1]
    roles = band_roles(src)
    if all(c in roles for c in ("red", "green", "blue")):
        return [roles["red"], roles["green"], roles["blue"]]
    if src.count >= 3:
        return [1, 2, 3] if src.dtypes[0] == "uint8" else [3, 2, 1]
    return [1]


def stretch(arr, grayscale_db=False, limits=None):
    """2-98 percentile stretch per band over valid (>0, finite) pixels. limits=[(lo, hi), ...]
    uses fixed per-band limits instead (the whole image's, see image_limits): stretching each view
    separately blows the noise of a small, uniform crop (e.g. open water) up to full range."""
    arr = arr.astype(np.float32)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for b in range(arr.shape[0]):
        band = arr[b]
        valid = np.isfinite(band) & (band > 0)
        if grayscale_db:
            band = np.where(valid, 10 * np.log10(np.where(valid, band, 1)), 0)
        if limits is not None:
            p2, p98 = limits[b]
        elif valid.any():
            p2, p98 = np.percentile(band[valid], [2, 98])
        else:
            p2, p98 = 0, 1
        out[b] = np.where(valid, np.clip((band - p2) / max(p98 - p2, 1e-6) * 255, 0, 255), 0).astype(np.uint8)
    return out


_LIMITS_CACHE = {}


def image_limits(src, bands, grayscale_db):
    """Per-band 2-98 percentiles of the whole image (decimated read), cached per file+bands."""
    key = (src.name, tuple(bands), grayscale_db)
    if key not in _LIMITS_CACHE:
        f = max(1.0, max(src.width, src.height) / 1024)
        arr = src.read(bands, out_shape=(len(bands), max(1, int(src.height / f)), max(1, int(src.width / f))),
                       resampling=Resampling.nearest).astype(np.float32)
        lims = []
        for band in arr:
            valid = np.isfinite(band) & (band > 0)
            if grayscale_db:
                band = 10 * np.log10(np.where(valid, band, 1))
            lims.append(tuple(np.percentile(band[valid], [2, 98])) if valid.any() else (0, 1))
        _LIMITS_CACHE[key] = lims
    return _LIMITS_CACHE[key]


def _is_db(src, bands):
    """Single float band with small values = linear sigma0 -> show in dB."""
    if len(bands) != 1 or np.dtype(src.dtypes[0]).kind != "f":
        return False
    # small decimated read, not src.statistics(): that can write .aux.xml next to the imagery
    sample = src.read(bands[0], out_shape=(64, 64), resampling=Resampling.nearest)
    return float(np.nanmax(sample)) < 50 if np.isfinite(sample).any() else False


def _ndwi_u8(arr):
    """(green - nir) / (green + nir) on [-0.6, 0.6] -> 0..255 (water bright)."""
    g, n = arr[0].astype(np.float32), arr[1].astype(np.float32)
    valid = (g > 0) & (n > 0)
    idx = np.where(valid, (g - n) / np.maximum(g + n, 1e-6), -1)
    return np.where(valid, np.clip((idx + 0.6) / 1.2 * 255, 0, 255), 0).astype(np.uint8)


def render_view(path, lon, lat, radius_m, offset_m=(0.0, 0.0), rgb_bands="auto", max_px=1400,
                stretch_mode="image", view_mode="true"):
    """Square view of +/-radius_m around lon/lat (+pan offset, metres E/N).
    view_mode: a VIEW_MODES key; falls back to true colour if the raster lacks its bands.

    Returns {img, transform, crs, native_res, native_transform} or None if the file is missing.
    Views wider than max_px native pixels are read decimated (so a whole-scene check of a
    big raster stays fast); view_transform is the transform of the RETURNED pixels."""
    if not path or not Path(path).exists():
        return None
    with rasterio.open(path) as raw:
        mcrs = metric_crs(raw)
        native_crs, native_tfm0 = raw.crs, raw.transform
        src = raw
        tfm = Transformer.from_crs("EPSG:4326", mcrs, always_xy=True)
        x, y = tfm.transform(lon, lat)
        x, y = x + offset_m[0], y + offset_m[1]
        if mcrs == raw.crs:
            window = rasterio.windows.from_bounds(x - radius_m, y - radius_m, x + radius_m, y + radius_m,
                                                  raw.transform).round_offsets().round_lengths()
            w, h = int(window.width), int(window.height)
        else:
            # degree-based raster: warp straight onto the requested metre grid (WarpedVRT can't read
            # boundless, so the VRT itself IS the view window; outside the image reads as 0)
            with WarpedVRT(raw, crs=mcrs) as probe:
                res0 = float(abs(probe.transform.a))
            w = h = max(1, int(round(2 * radius_m / res0)))
        f = max(1.0, max(w, h) / max_px)
        ow, oh = max(1, int(round(w / f))), max(1, int(round(h / f)))
        if mcrs != raw.crs:
            grid = rasterio.transform.from_origin(x - radius_m, y + radius_m, 2 * radius_m / ow, 2 * radius_m / oh)
            src = WarpedVRT(raw, crs=mcrs, transform=grid, width=ow, height=oh,
                            resampling=Resampling.average if f > 1 else Resampling.nearest)
        roles = band_roles(src)
        need = VIEW_MODES.get(view_mode, VIEW_MODES["true"])[1]
        if view_mode != "true" and all(r in roles for r in need):
            bands = [roles[r] for r in need]
        else:
            view_mode = "true"
            bands = pick_bands(src, rgb_bands)
        if src is raw:
            arr = src.read(bands, window=window, out_shape=(len(bands), oh, ow), boundless=True, fill_value=0,
                           resampling=Resampling.average if f > 1 else Resampling.nearest)
            view_tfm = src.window_transform(window) * Affine.scale(w / ow, h / oh)
            native_res = float(abs(raw.transform.a))
        else:
            arr = src.read(bands)
            view_tfm = src.transform
            native_res = 2 * radius_m / w
            src.close()
        crs = mcrs
        db = view_mode == "true" and _is_db(raw, bands)
        limits = image_limits(raw, bands, db) if stretch_mode == "image" and view_mode != "ndwi" else None
    gray = len(bands) == 1
    if view_mode == "ndwi":
        out = _ndwi_u8(arr)[None]
        gray = True
    else:
        out = stretch(arr, grayscale_db=db, limits=limits)
    if gray:
        img = Image.fromarray(out[0]).convert("RGB")
    else:
        img = Image.fromarray(np.transpose(out[:3], (1, 2, 0))).convert("RGB")
    # native_* = the SOURCE file's own grid (for pixel_box), which differs from the view's CRS
    # when a degree-based raster was warped to UTM for viewing
    return {"img": img, "transform": view_tfm, "crs": crs, "native_res": native_res,
            "native_transform": native_tfm0, "native_crs": native_crs, "view_mode": view_mode}


_BOUNDS_CACHE = {}


def image_bounds(path):
    """(crs, (left, bottom, right, top)) of a raster in its VIEW crs (metric_crs), cached --
    for the review minimap, whose rectangle is computed from view coordinates."""
    if path not in _BOUNDS_CACHE:
        with rasterio.open(path) as src:
            mcrs = metric_crs(src)
            b = tuple(src.bounds) if mcrs == src.crs else transform_bounds(src.crs, mcrs, *src.bounds)
            _BOUNDS_CACHE[path] = (mcrs, tuple(b))
    return _BOUNDS_CACHE[path]


def render_thumbnail(path, size=280, rgb_bands="auto"):
    """Whole-image preview for the imagery browser / dashboard."""
    try:
        with rasterio.open(path) as src:
            f = max(1.0, max(src.width, src.height) / size)
            ow, oh = max(1, int(src.width / f)), max(1, int(src.height / f))
            bands = pick_bands(src, rgb_bands)
            arr = src.read(bands, out_shape=(len(bands), oh, ow), resampling=Resampling.average)
            db = _is_db(src, bands)
        gray = len(bands) == 1
        out = stretch(arr, grayscale_db=db)
        return Image.fromarray(out[0] if gray else np.transpose(out[:3], (1, 2, 0))).convert("RGB")
    except Exception:
        return None


# ----------------------------------------------------------------- queue ---

ID_TAG = "optical"   # fixed id component, kept so review data from earlier versions still matches


def detection_id(site, item_id, lon, lat):
    """Location-based id (area | tag | image | lon_lat rounded to 1e-4 deg): stable across detector
    reruns, unlike an index into the file. Near-duplicates get a #n suffix in build_queue."""
    return f"{site}|{ID_TAG}|{item_id}|{round(lon, 4)}_{round(lat, 4)}"


def image_key(entry):
    return f"{entry['site']}|{entry['item_id']}"


def _flat_coords(geom):
    """Every [x, y] in any GeoJSON geometry (Point ... GeometryCollection)."""
    if not geom:
        return []
    if geom.get("type") == "GeometryCollection":
        return [c for g in geom.get("geometries", []) for c in _flat_coords(g)]
    out, stack = [], [geom.get("coordinates")]
    while stack:
        c = stack.pop()
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            out.append([float(c[0]), float(c[1])])
        elif isinstance(c, (list, tuple)):
            stack.extend(c)
    return out


def _geom_center(geom):
    """Point: itself; polygon: mean of its outer ring; anything else: mean of all its vertices."""
    if geom["type"] == "Point":
        return [float(geom["coordinates"][0]), float(geom["coordinates"][1])]
    if geom["type"] in ("Polygon", "MultiPolygon"):
        ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
        pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    else:
        pts = _flat_coords(geom)
    return [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]


def _bbox_polygon(geom):
    """Axis-aligned lon/lat box around a geometry; None if it has no extent (a single point)."""
    pts = _flat_coords(geom)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 <= 0 and y1 - y0 <= 0:
        return None
    return {"type": "Polygon", "coordinates": [[[x0, y1], [x1, y1], [x1, y0], [x0, y0], [x0, y1]]]}


def build_queue(project):
    """Flat review queue over the project's imagery: per image, its detections (from the
    matched geojson files, in file order) then its scene check (from the file, or a
    synthesised whole-image one when scene_checks is on and the file has none)."""
    items = []
    scene_checks = project.settings.get("scene_checks", True)
    blind = holdout_keys(project) if project.settings.get("holdout_blind", True) else set()
    prelabels = prelabel_files(project)
    for entry in project.imagery:
        site, ikey = entry["site"], entry["item_id"]
        has_scene = False
        seen = {}   # id -> (lon, lat)
        is_blind = image_key(entry) in blind   # hold-out: labelled from a blank image, no detector boxes
        files = list(entry.get("detections") or [])
        if prelabels.get(image_key(entry)) and prelabels[image_key(entry)] not in files:
            files.append(prelabels[image_key(entry)])   # model pre-labels (assist.py writes these)
        for gj in ([] if is_blind else files):
            try:
                fc = json.loads(Path(gj).read_text())
            except (OSError, ValueError):
                continue
            coll = {k: fc[k] for k in ("item_id", "date") if k in fc}
            for feat in fc.get("features", []):
                p, geom = {**coll, **(feat.get("properties") or {})}, feat.get("geometry")
                if not geom or not _flat_coords(geom):
                    continue
                lon, lat = _geom_center(geom)
                item_id = p.get("item_id") or ikey
                is_scene = p.get("kind") == "scene"
                did = detection_id(site, item_id, lon, lat) + ("|scene" if is_scene else "")
                if did in seen:
                    if seen[did] == (lon, lat):
                        continue   # the very same feature listed twice (e.g. two files): one item
                    # a DIFFERENT detection within ~11 m (ids round to 1e-4 deg): keep it, distinct id.
                    # Ids of non-colliding detections are unchanged.
                    n = 2
                    while f"{did}#{n}" in seen:
                        n += 1
                    did = f"{did}#{n}"
                seen[did] = (lon, lat)
                has_scene = has_scene or is_scene
                items.append({
                    "id": did, "kind": "scene" if is_scene else "detection",
                    "view_radius_m": p.get("view_radius_m"), "source": p.get("source"),
                    "site": site, "item_id": item_id, "date": p.get("date") or entry.get("date"),
                    "area_px": p.get("area_px"), "lon": lon, "lat": lat,
                    "bbox_w_px": p.get("bbox_w_px"), "bbox_h_px": p.get("bbox_h_px"),
                    "score": p.get("score"), "pred_class": p.get("pred_class"),
                    "det_box": _bbox_polygon(geom) if geom["type"] != "Point" else None,
                    "image_key": image_key(entry), "image_path": entry["path"],
                    "det_res_m": entry.get("res_m") or 10.0,
                })
        if (scene_checks or is_blind) and not has_scene and entry.get("center"):
            lon, lat = entry["center"]
            radius = max(entry.get("extent_m") or [3000, 3000]) / 2
            items.append({
                "id": detection_id(site, ikey, lon, lat) + "|scene", "kind": "scene",
                "view_radius_m": radius, "source": None, "site": site, "item_id": ikey,
                "date": entry.get("date"), "area_px": None, "lon": lon, "lat": lat,
                "bbox_w_px": None, "bbox_h_px": None, "det_box": None, "score": None,
                "image_key": image_key(entry), "image_path": entry["path"],
                "det_res_m": entry.get("res_m") or 10.0, "synthesised": True,
            })
    return items


def _assist_state(project):
    try:
        return json.loads((project.root / "models" / "assist.json").read_text())
    except (OSError, ValueError):
        return {}


def prelabel_files(project):
    """{image_key: model detections geojson} -- pre-labels from the latest Predict per image (assist.py).
    Kept in models/assist.json so a background job never has to rewrite project.json."""
    return _assist_state(project).get("prelabels", {})


def holdout_keys(project):
    """Image keys of the model-assist hold-out set (models/assist.json; see assist.py)."""
    try:
        return set(json.loads((project.root / "models" / "assist.json").read_text()).get("holdout", []))
    except (OSError, ValueError):
        return set()


def detection_box(item, pad_px=2):
    """The box the reviewer saw for an un-edited detection, as a lon/lat polygon: its own
    polygon if it came with one, else the point +/- (max(bbox,3)/2 + 2px pad), as drawn."""
    if item.get("det_box"):
        return item["det_box"]
    res = item.get("det_res_m") or 10.0
    hw = (max(item.get("bbox_w_px") or 0, 3) / 2 + pad_px) * res
    hh = (max(item.get("bbox_h_px") or 0, 3) / 2 + pad_px) * res
    zone = int((item["lon"] + 180) / 6) + 1
    utm = f"EPSG:{(32600 if item['lat'] >= 0 else 32700) + zone}"
    fwd = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    inv = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    x, y = fwd.transform(item["lon"], item["lat"])
    ring = [list(inv.transform(cx, cy)) for cx, cy in
            ((x - hw, y + hh), (x + hw, y + hh), (x + hw, y - hh), (x - hw, y - hh))]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


# --------------------------------------------------------------- project ---

class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        meta = json.loads((self.root / "project.json").read_text())
        self.name = meta.get("name", self.root.name)
        self.description = meta.get("description", "")
        self.created_at = meta.get("created_at")
        self.format_version = meta.get("format_version", 1)
        self.settings = {**DEFAULT_SETTINGS, **meta.get("settings", {})}
        self.imagery = meta.get("imagery", [])
        self.state = self._load(self.state_path, {})
        self.additions = self._load(self.additions_path, {"type": "FeatureCollection", "features": []})
        if self.format_version < PROJECT_FORMAT_VERSION:
            self._migrate(meta)

    def _migrate(self, meta):
        """Format 1 (the original single-project tool) -> 2: settings vessel_classes -> classes,
        assist.object_name -> object_name, record/box key vessel_class -> class.
        The original files are kept next to the new ones as *.v1.bak."""
        old_classes, old_class = "vessel_classes", "vessel_class"   # format-1 key names
        st = meta.get("settings", {})
        if "classes" not in st and isinstance(st.get(old_classes), list):
            self.settings["classes"] = list(st[old_classes])
        self.settings.pop(old_classes, None)
        if "object_name" not in st:
            self.settings["object_name"] = (st.get("assist") or {}).get("object_name") or "object"
        for rec in self.state.values():
            if old_class in rec:
                rec[CLASS_KEY] = rec.pop(old_class)
        for f in self.additions["features"]:
            if old_class in f.get("properties", {}):
                f["properties"][CLASS_KEY] = f["properties"].pop(old_class)
        for path in (self.root / "project.json", self.state_path, self.additions_path):
            if path.exists():
                bak = path.with_name(path.name + ".v1.bak")
                if not bak.exists():
                    bak.write_bytes(path.read_bytes())
        self.format_version = PROJECT_FORMAT_VERSION
        self.save_meta()
        if self.state_path.exists():
            self.save_state()
        if self.additions_path.exists():
            self.save_additions()

    # -- paths / io --
    @property
    def review_dir(self):
        return self.root / "review_data"

    @property
    def state_path(self):
        return self.review_dir / "review_state.json"

    @property
    def additions_path(self):
        return self.review_dir / "manual_additions.geojson"

    @staticmethod
    def _load(path, default):
        if path.exists():
            return json.loads(path.read_text())
        return default

    @classmethod
    def create(cls, parent, name, description="", settings=None):
        root = Path(parent).expanduser().resolve() / slugify(name)
        if (root / "project.json").exists():
            raise FileExistsError(f"A project already exists at {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(root / "project.json", {
            "format_version": PROJECT_FORMAT_VERSION,
            "name": name, "description": description, "created_at": now_iso(),
            "settings": {**DEFAULT_SETTINGS, **(settings or {})}, "imagery": [],
        })
        (root / "review_data").mkdir(exist_ok=True)
        add_recent(root)
        return cls(root)

    def save_meta(self):
        write_json_atomic(self.root / "project.json", {
            "format_version": self.format_version,
            "name": self.name, "description": self.description, "created_at": self.created_at,
            "settings": self.settings, "imagery": self.imagery,
        })

    def save_state(self):
        write_json_atomic(self.state_path, self.state)

    def save_additions(self):
        write_json_atomic(self.additions_path, self.additions)

    def log_action(self, action, **data):
        """Append-only audit trail (review_data/history.jsonl): every decision and box change with
        its before/after, so lost work can be recovered by hand even across sessions."""
        self.review_dir.mkdir(parents=True, exist_ok=True)
        with open(self.review_dir / "history.jsonl", "a") as fh:
            fh.write(json.dumps({"at": now_iso(), "action": action, **data}) + "\n")

    @property
    def classes(self):
        """Object types for number keys 1-9; [] = the project labels objects without types."""
        v = self.settings.get("classes")
        return list(v) if isinstance(v, list) else []

    @property
    def object_name(self):
        return (self.settings.get("object_name") or "object").strip()

    # -- imagery --
    def add_imagery(self, entries):
        have = {image_key(e) for e in self.imagery}
        added = 0
        for e in entries:
            if e.get("error") or image_key(e) in have:
                continue
            self.imagery.append({k: e[k] for k in ("path", "site", "item_id", "date", "detections", "width",
                                                   "height", "count", "dtype", "res_m", "crs", "center",
                                                   "extent_m") if k in e})
            have.add(image_key(e))
            added += 1
        self.imagery.sort(key=lambda e: (e["site"], e.get("date") or "", e["item_id"]))
        self.save_meta()
        return added

    def remove_imagery(self, keys):
        """Removes images from the project only; their review records are kept on disk so
        re-adding the same image brings the work back."""
        keys = set(keys)
        self.imagery = [e for e in self.imagery if image_key(e) not in keys]
        self.save_meta()

    def import_review_data(self, src_dir):
        """Merge another review_data/ (another project's, or an older copy) into this project.
        Existing records here win; additions are de-duplicated on (review_item_id, added_at)."""
        src_dir = Path(src_dir)
        n_state = n_add = 0
        sp, ap = src_dir / "review_state.json", src_dir / "manual_additions.geojson"
        if sp.exists():
            for k, v in json.loads(sp.read_text()).items():
                if k not in self.state:
                    self.state[k] = v
                    n_state += 1
            self.save_state()
        if ap.exists():
            seen = {(f["properties"].get("review_item_id"), f["properties"].get("added_at"))
                    for f in self.additions["features"]}
            for f in json.loads(ap.read_text()).get("features", []):
                k = (f["properties"].get("review_item_id"), f["properties"].get("added_at"))
                if k not in seen:
                    self.additions["features"].append(f)
                    seen.add(k)
                    n_add += 1
            self.save_additions()
        return n_state, n_add

    # -- progress --
    def image_progress(self, queue=None):
        """{image_key: counts} for the dashboard and the export summary."""
        queue = queue if queue is not None else build_queue(self)
        lenient = self.settings.get("lenient_edited", True)
        prog = {image_key(e): {"n_det": 0, "valid": 0, "invalid": 0, "skipped": 0, "edited_only": 0,
                               "undecided": 0, "added": 0, "scene": None, "has_scene": False}
                for e in self.imagery}
        for it in queue:
            p = prog.get(it["image_key"])
            if p is None:
                continue
            rec = self.state.get(it["id"], {})
            st = rec.get("status")
            if it["kind"] == "scene":
                p["has_scene"] = True
                p["scene"] = st
                continue
            p["n_det"] += 1
            if st in ("valid", "invalid", "skipped"):
                p[st] += 1
            elif rec.get("corrected_box"):
                p["edited_only"] += 1
            else:
                p["undecided"] += 1
        sites_items = {(e["site"], e["item_id"]): image_key(e) for e in self.imagery}
        for f in self.additions["features"]:
            fp = f["properties"]
            k = sites_items.get((fp.get("site"), fp.get("item_id")))
            if k in prog:
                prog[k]["added"] += 1
        for p in prog.values():
            undecided = p["undecided"] + (0 if lenient else p["edited_only"])
            p["complete"] = undecided == 0 and (p["scene"] == "valid" if p["has_scene"] else p["n_det"] > 0)
            p["unreadable"] = p["scene"] == "invalid"
            p["objects"] = p["valid"] + (p["edited_only"] if lenient else 0) + p["added"]
        return prog


# ---------------------------------------------------------------- export ---

def _lonlat_poly_to_px(geom, inv_transform, to_crs):
    ring = geom["coordinates"][0][:4]
    cols, rows = [], []
    for lon, lat in ring:
        x, y = to_crs.transform(lon, lat)
        c, r = inv_transform * (x, y)
        cols.append(c); rows.append(r)
    return min(cols), min(rows), max(cols), max(rows)


def gather_labels(project, queue=None):
    """Every object label per image and every rejected detection, as lon/lat polygons.
    Returns (objects {image_key: [Feature]}, negatives [Feature], progress, entries)."""
    lenient = project.settings.get("lenient_edited", True)
    queue = queue if queue is not None else build_queue(project)
    prog = project.image_progress(queue)
    entries = {image_key(e): e for e in project.imagery}
    objects = {k: [] for k in entries}
    negatives = []
    for it in queue:
        if it["kind"] == "scene":
            continue
        rec = project.state.get(it["id"], {})
        st = rec.get("status")
        edited = bool(rec.get("corrected_box"))
        base = {"site": it["site"], "item_id": it["item_id"], "date": it["date"], "review_id": it["id"],
                "detector_source": it.get("source"), "status": st, "image_path": it["image_path"]}
        geom = rec.get("corrected_box") or detection_box(it)
        if st == "valid" or (st is None and edited and lenient):
            label = "detection_edited" if edited else "detection_confirmed"
            if st is None:
                label = "detection_edited_unmarked"
            objects[it["image_key"]].append({"type": "Feature", "geometry": geom, "properties": {
                **base, "label_source": label, CLASS_KEY: rec.get(CLASS_KEY) or UNCLASSIFIED}})
        elif st == "invalid":
            negatives.append({"type": "Feature", "geometry": geom, "properties": base})
    by_pass = {(e["site"], e["item_id"]): k for k, e in entries.items()}
    for f in project.additions["features"]:
        fp = f["properties"]
        k = by_pass.get((fp.get("site"), fp.get("item_id")))
        if k is None:
            continue
        objects[k].append({"type": "Feature", "geometry": f["geometry"],
                           "properties": {"site": fp.get("site"), "item_id": fp.get("item_id"),
                                          "date": fp.get("date"), "review_id": fp.get("review_item_id"),
                                          "status": "valid", "label_source": "manual_addition",
                                          CLASS_KEY: fp.get(CLASS_KEY) or UNCLASSIFIED,
                                          "added_at": fp.get("added_at"), "image_path": entries[k]["path"]}})

    return objects, negatives, prog, entries


def export_labels(project, out_dir=None, only_complete=True, yolo=True, max_px=4096, progress=None,
                  yolo_classes=False):
    """Writes labels.geojson (objects), negatives.geojson (rejected detections),
    image_summary.csv, and optionally a YOLO dataset (images/, labels/, data.yaml) of the
    images that pass the only_complete filter. yolo_classes=True writes one YOLO class per
    object type (+ "unclassified" last) instead of a single class named after the object.
    Returns (out_dir, summary dict)."""
    lenient = project.settings.get("lenient_edited", True)
    objects, negatives, prog, entries = gather_labels(project)
    out = Path(out_dir) if out_dir else project.root / "exports" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    all_labels = [f for k in objects for f in objects[k]]
    write_json_atomic(out / "labels.geojson", {"type": "FeatureCollection", "features": all_labels})
    write_json_atomic(out / "negatives.geojson", {"type": "FeatureCollection", "features": negatives})

    cols = ["site", "date", "item_id", "n_detections", "valid", "invalid", "skipped", "edited_unmarked",
            "undecided", "added", "scene_check", "complete", "object_count", "image_path"]
    with open(out / "image_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for k, e in entries.items():
            p = prog[k]
            w.writerow([e["site"], e.get("date"), e["item_id"], p["n_det"], p["valid"], p["invalid"],
                        p["skipped"], p["edited_only"], p["undecided"], p["added"],
                        p["scene"] or ("pending" if p["has_scene"] else "n/a"), p["complete"],
                        p["objects"], e["path"]])

    class_names = project.classes + [UNCLASSIFIED] if yolo_classes else [project.object_name]
    n_yolo = n_boxes = 0
    if yolo:
        img_dir, lbl_dir = out / "yolo" / "images", out / "yolo" / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        todo = [k for k in entries if not prog[k]["unreadable"] and (prog[k]["complete"] or not only_complete)]
        for i, k in enumerate(todo):
            if progress:
                progress(i, len(todo))
            e = entries[k]
            if not Path(e["path"]).exists():
                continue
            with rasterio.open(e["path"]) as src:
                f = max(1.0, max(src.width, src.height) / max_px)
                ow, oh = max(1, int(round(src.width / f))), max(1, int(round(src.height / f)))
                bands = pick_bands(src, project.settings.get("rgb_bands", "auto"))
                arr = src.read(bands, out_shape=(len(bands), oh, ow),
                               resampling=Resampling.average if f > 1 else Resampling.nearest)
                tfm = src.transform * Affine.scale(src.width / ow, src.height / oh)
                to_crs = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                db = _is_db(src, bands)
            gray = len(bands) == 1
            st = stretch(arr, grayscale_db=db)
            img = Image.fromarray(st[0] if gray else np.transpose(st[:3], (1, 2, 0))).convert("RGB")
            stem = f"{e['site']}__{e['item_id']}"
            img.save(img_dir / f"{stem}.png")
            lines = []
            for feat in objects[k]:
                c0, r0, c1, r1 = _lonlat_poly_to_px(feat["geometry"], ~tfm, to_crs)
                c0, c1 = max(0.0, c0), min(float(ow), c1)
                r0, r1 = max(0.0, r0), min(float(oh), r1)
                if c1 - c0 < 1 or r1 - r0 < 1:
                    continue  # box falls outside this image
                cls = class_names.index(feat["properties"][CLASS_KEY]) \
                    if yolo_classes and feat["properties"][CLASS_KEY] in class_names else \
                    (len(class_names) - 1 if yolo_classes else 0)
                lines.append(f"{cls} {(c0 + c1) / 2 / ow:.6f} {(r0 + r1) / 2 / oh:.6f} "
                             f"{(c1 - c0) / ow:.6f} {(r1 - r0) / oh:.6f}")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_yolo += 1
            n_boxes += len(lines)
        (out / "yolo" / "data.yaml").write_text(
            f"# exported by review_app {now_iso()} from {project.root}\n"
            f"path: {out / 'yolo'}\ntrain: images\nval: images\nnames:\n" +
            "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names)))

    summary = {"images": len(entries), "complete_images": sum(p["complete"] for p in prog.values()),
               "object_labels": len(all_labels), "negatives": len(negatives),
               "yolo_images": n_yolo, "yolo_boxes": n_boxes, "only_complete": only_complete,
               "yolo_classes": class_names,
               "lenient_edited": lenient, "exported_at": now_iso()}
    write_json_atomic(out / "export_summary.json", summary)
    return out, summary


# ----------------------------------------------------------------- stats ---

def compute_stats(project, queue=None):
    """Live review statistics (same lenient/strict reading as score_timeseries_review.py):
      by_source: per (site, detector source) -- confirmed / rejected / precision over decided
      by_site:   per site -- detector hits, missed objects you added, recall = hits / (hits + added)
                 over images whose scene check is done (elsewhere "missed" isn't known yet)
      series:    per site, [(date, objects, complete, started)] -- objects per image over time;
                 started=False means nobody has looked at it yet (its 0 means nothing)
      classes:   object count per class (confirmed detections + added boxes)"""
    queue = queue if queue is not None else build_queue(project)
    lenient = project.settings.get("lenient_edited", True)
    prog = project.image_progress(queue)
    entries = {image_key(e): e for e in project.imagery}
    by_source, hits, classes = {}, {}, {}
    for it in queue:
        if it["kind"] == "scene":
            continue
        rec = project.state.get(it["id"], {})
        st = rec.get("status")
        if st is None and rec.get("corrected_box") and lenient:
            st = "valid"
        row = by_source.setdefault((it["site"], it.get("source") or "detector"),
                                   {"valid": 0, "invalid": 0, "skipped": 0, "undecided": 0})
        row[st if st in ("valid", "invalid", "skipped") else "undecided"] += 1
        if st == "valid":
            c = rec.get(CLASS_KEY) or UNCLASSIFIED
            classes[c] = classes.get(c, 0) + 1
            if prog.get(it["image_key"], {}).get("scene") == "valid":
                hits[it["site"]] = hits.get(it["site"], 0) + 1
    added = {}
    by_pass = {(e["site"], e["item_id"]): k for k, e in entries.items()}
    for f in project.additions["features"]:
        fp = f["properties"]
        k = by_pass.get((fp.get("site"), fp.get("item_id")))
        if k is None:
            continue
        c = fp.get(CLASS_KEY) or UNCLASSIFIED
        classes[c] = classes.get(c, 0) + 1
        if prog[k]["scene"] == "valid":
            added[fp["site"]] = added.get(fp["site"], 0) + 1
    src_rows = []
    for (site, source), r in sorted(by_source.items()):
        decided = r["valid"] + r["invalid"]
        src_rows.append({"site": site, "source": source, **r,
                         "precision": r["valid"] / decided if decided else None})
    site_rows = []
    for site in sorted({e["site"] for e in project.imagery}):
        h, a = hits.get(site, 0), added.get(site, 0)
        ks = [k for k, e in entries.items() if e["site"] == site]
        site_rows.append({"site": site, "images": len(ks), "checked": sum(prog[k]["scene"] == "valid" for k in ks),
                          "complete": sum(prog[k]["complete"] for k in ks), "hits": h, "added": a,
                          "recall": h / (h + a) if h + a else None,
                          "objects": sum(prog[k]["objects"] for k in ks)})
    series = {}
    for k, e in entries.items():
        if prog[k]["unreadable"] or not e.get("date"):
            continue
        p = prog[k]
        started = bool(p["valid"] + p["invalid"] + p["skipped"] + p["edited_only"] + p["added"]) or p["scene"]
        series.setdefault(e["site"], []).append((e["date"], p["objects"], p["complete"], bool(started)))
    for v in series.values():
        v.sort()
    return {"by_source": src_rows, "by_site": site_rows, "series": series, "classes": classes}
