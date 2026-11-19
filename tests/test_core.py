"""Core tests: scanning, georeferencing (projected + geographic CRS), rendering, the review queue,
export (GeoJSON + YOLO), band views, statistics, and regressions from earlier debugging passes."""
import json

import numpy as np
import rasterio
from conftest import check
from pyproj import Transformer
from rasterio.transform import from_origin

from boxtrot.core import assist
from boxtrot.core import project as pj


def make_raster(path, w=3000, h=2400, target_px=(2100, 600), res=10.0):
    """uint16 blue,green,red raster in UTM 18N with a 6x6 bright target."""
    left, top = 770000.0, 2060000.0
    tfm = from_origin(left, top, res, res)
    data = np.full((3, h, w), 500, dtype=np.uint16)   # flat background: the stretch maps it to 0
    c, r = target_px
    data[:, r:r + 6, c:c + 6] = 5000
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint16",
                       crs="EPSG:32618", transform=tfm) as dst:
        dst.write(data)
        dst.descriptions = ("blue", "green", "red")
    x, y = tfm * (c + 3, r + 3)
    lon, lat = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True).transform(x, y)
    return lon, lat, tfm


def test_synthetic(tmp):
    img_dir = tmp / "imgs" / "harbour_a"
    path = img_dir / "S2A_18QYF_20250101_0_L2A.tif"
    tlon, tlat, _ = make_raster(path)

    # scanning / site / key / date inference
    cands = pj.scan_imagery(tmp / "imgs")
    check(len(cands) == 1 and cands[0]["site"] == "harbour_a", "scan finds image, site from folder")
    check(cands[0]["item_id"] == "S2A_18QYF_20250101_0_L2A" and cands[0]["date"] == "2025-01-01",
          "item id + date parsed")
    check(pj.normalise_key("harbour_2026-03_S2B_18QYG_20260303_0_L2A_visual") == "S2B_18QYG_20260303_0_L2A",
          "S2 id extracted from visual filename")

    # decimated whole-scene view: the target must land at its true location
    c = cands[0]
    view = pj.render_view(str(path), *c["center"], radius_m=max(c["extent_m"]) / 2, max_px=500)
    arr = np.asarray(view["img"]).sum(axis=2)
    check(max(view["img"].size) <= 500, f"whole scene decimated to {view['img'].size}")
    rows, cols = np.nonzero(arr > arr.max() / 2)
    x, y = view["transform"] * (cols.mean() + 0.5, rows.mean() + 0.5)   # target centroid
    lon, lat = Transformer.from_crs(view["crs"], "EPSG:4326", always_xy=True).transform(x, y)
    err_m = np.hypot((lon - tlon) * 111320 * np.cos(np.radians(tlat)), (lat - tlat) * 110570)
    check(err_m < 2 * abs(view["transform"].a), f"bright target located within {err_m:.0f} m in decimated view "
                                                 f"({abs(view['transform'].a):.0f} m/px)")

    # full-res detection view, round trip lon/lat -> px -> lon/lat
    view = pj.render_view(str(path), tlon, tlat, radius_m=150)
    fwd = Transformer.from_crs("EPSG:4326", view["crs"], always_xy=True)
    inv = Transformer.from_crs(view["crs"], "EPSG:4326", always_xy=True)
    col, row = ~view["transform"] * fwd.transform(tlon, tlat)
    check(abs(col - 15) < 1 and abs(row - 15) < 1, f"target at view centre (col {col:.1f}, row {row:.1f} of 30)")
    lon2, lat2 = inv.transform(*(view["transform"] * (col, row)))
    check(abs(lon2 - tlon) < 1e-9 and abs(lat2 - tlat) < 1e-9, "lon/lat round trip exact")

    # project + queue + export
    proj = pj.Project.create(tmp / "projects", "Synthetic test")
    check(proj.add_imagery(cands) == 1 and proj.add_imagery(cands) == 0, "add imagery once, duplicates skipped")
    q = pj.build_queue(proj)
    check(len(q) == 1 and q[0]["kind"] == "scene" and q[0].get("synthesised"),
          "image without detections gets one synthesised scene check")
    prog = proj.image_progress(q)
    check(not list(prog.values())[0]["complete"], "unchecked image is not complete")

    # manual box exactly around the target, then mark the scene check valid
    tfm = rasterio.open(path).transform
    to_ll = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    ring = [list(to_ll.transform(*(tfm * (c_, r_)))) for c_, r_ in ((2100, 600), (2106, 600), (2106, 606), (2100, 606))]
    proj.additions["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
                                       "properties": {"site": q[0]["site"], "item_id": q[0]["item_id"],
                                                      "date": q[0]["date"], "review_item_id": q[0]["id"],
                                                      "added_at": pj.now_iso()}})
    proj.save_additions()
    out, s = pj.export_labels(proj, tmp / "exp_incomplete")
    check(s["yolo_images"] == 0 and s["object_labels"] == 1, "incomplete image: label exported, no YOLO image")
    proj.state[q[0]["id"]] = {"status": "valid"}
    proj.save_state()
    out, s = pj.export_labels(proj, tmp / "exp", max_px=1000)
    lbl = (out / "yolo/labels/harbour_a__S2A_18QYF_20250101_0_L2A.txt").read_text().split()
    exp = [0, 2103 / 3000, 603 / 2400, 6 / 3000, 6 / 2400]
    check(len(lbl) == 5 and all(abs(float(a) - b) < 2e-3 for a, b in zip(lbl, exp)),
          f"YOLO box matches target (got {lbl[1:]}, want {[round(v, 4) for v in exp[1:]]})")
    from PIL import Image
    check(Image.open(out / "yolo/images/harbour_a__S2A_18QYF_20250101_0_L2A.png").width == 1000,
          "YOLO image decimated to max_px")

    # reopen from disk
    p2 = pj.Project(proj.root)
    check(len(p2.imagery) == 1 and p2.state == proj.state and len(p2.additions["features"]) == 1,
          "project reopens with imagery/state/additions")

    # point detections file with bbox: detection_box size matches review_tool's padded footprint
    det_dir = tmp / "dets" / "harbour_a"
    det_dir.mkdir(parents=True)
    (det_dir / "S2A_18QYF_20250101_0_L2A_docked.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [tlon, tlat]},
         "properties": {"site": "harbour_a", "item_id": "S2A_18QYF_20250101_0_L2A", "date": "2025-01-01",
                        "bbox_w_px": 6, "bbox_h_px": 6, "area_px": 36}}]}))
    cands = pj.scan_imagery(tmp / "imgs", tmp / "dets")
    check(cands[0]["n_detections"] == 1, "detection file matched to image")
    proj2 = pj.Project.create(tmp / "projects", "Synthetic dets")
    proj2.add_imagery(cands)
    q = pj.build_queue(proj2)
    check([it["kind"] for it in q] == ["detection", "scene"], "queue = detection then scene check")
    box = pj.detection_box(q[0])["coordinates"][0]
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    xs = [fwd.transform(*p)[0] for p in box]
    check(abs((max(xs) - min(xs)) - 100) < 0.5,
          f"detection box width {(max(xs) - min(xs)):.1f} m (2 x (6px/2 + 2px pad) x 10 m = 100 m)")


def test_new_features(tmp):
    """S2 band names, band views, scores, types, stats, multi-class YOLO."""
    w, h = 400, 300
    path = tmp / "s2" / "harbour_b" / "S2A_18QYF_20250301_0_L2A.tif"
    path.parent.mkdir(parents=True)
    data = np.full((9, h, w), 1000, dtype=np.uint16)
    data[1, :, :200] = 2000   # B03 green high, B08 nir low on the west half -> water (NDWI > 0)
    data[6, :, :200] = 300
    data[6, :, 200:] = 3000   # east half: vegetation-like (NDWI < 0)
    tfm = from_origin(770000.0, 2075000.0, 10, 10)
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=9, dtype="uint16",
                       crs="EPSG:32618", transform=tfm) as dst:
        dst.write(data)
        dst.descriptions = ("B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12")
    with rasterio.open(path) as src:
        check(pj.pick_bands(src) == [1, 2, 3], "S2 band names B04/B03/B02 picked as RGB")
        check(pj.band_roles(src).get("nir") == 7 and pj.band_roles(src).get("swir22") == 9, "NIR/SWIR roles found")
    check(pj.available_views(str(path)) == ["true", "nir", "swir", "ndwi"], "all band views available")
    lon, lat = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True).transform(772000, 2073500)
    v = pj.render_view(str(path), lon, lat, 1000, view_mode="ndwi")
    a = np.asarray(v["img"])[:, :, 0].astype(int)
    check(v["view_mode"] == "ndwi" and a[:, :90].mean() > 150 and a[:, 110:190].mean() < 80,
          f"NDWI view: water half bright ({a[:, :90].mean():.0f}), land half dark ({a[:, 110:190].mean():.0f})")
    rgb_only = tmp / "imgs" / "harbour_a" / "S2A_18QYF_20250101_0_L2A.tif"
    if rgb_only.exists():
        v = pj.render_view(str(rgb_only), lon, lat, 500, view_mode="nir")
        check(v["view_mode"] == "true", "3-band image falls back to true colour for nir view")

    det_dir = tmp / "s2dets" / "harbour_b"
    det_dir.mkdir(parents=True)
    pts = [Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True).transform(*(tfm * (c, 150)))
           for c in (50, 100, 150, 300)]
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": list(p)},
              "properties": {"source": src, "score": sc, "bbox_w_px": 6, "bbox_h_px": 6}}
             for p, src, sc in zip(pts, ("detector_a", "detector_a", "detector_b", "detector_b"), (0.9, 0.81, None, None))]
    (det_dir / "S2A_18QYF_20250301.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "site": "harbour_b", "item_id": "S2A_18QYF_20250301_0_L2A",
         "date": "2025-03-01", "features": feats}))
    proj = pj.Project.create(tmp / "projects", "Features")
    proj.settings["classes"] = ["Large vessel", "Small boat", "Tug/barge", "Other"]
    proj.add_imagery(pj.scan_imagery(tmp / "s2", tmp / "s2dets"))
    q = pj.build_queue(proj)
    check([it.get("score") for it in q[:2]] == [0.9, 0.81], "detector scores carried into the queue")
    # decisions: detector_a 1 valid (type Small boat) + 1 invalid; detector_b 1 valid + 1 invalid; 1 added box
    proj.state[q[0]["id"]] = {"status": "valid", pj.CLASS_KEY: "Small boat"}
    proj.state[q[1]["id"]] = {"status": "invalid"}
    proj.state[q[2]["id"]] = {"status": "valid"}
    proj.state[q[3]["id"]] = {"status": "invalid"}
    proj.state[q[4]["id"]] = {"status": "valid"}   # scene check done
    proj.additions["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
        [[pts[3][0] - 1e-4, pts[3][1] + 1e-4], [pts[3][0] + 1e-4, pts[3][1] + 1e-4],
         [pts[3][0] + 1e-4, pts[3][1] - 1e-4], [pts[3][0] - 1e-4, pts[3][1] - 1e-4],
         [pts[3][0] - 1e-4, pts[3][1] + 1e-4]]]},
        "properties": {"site": "harbour_b", "item_id": "S2A_18QYF_20250301_0_L2A", "date": "2025-03-01",
                       pj.CLASS_KEY: "Large vessel", "added_at": pj.now_iso()}})
    st = pj.compute_stats(proj, q)
    prec = {r["source"]: r["precision"] for r in st["by_source"]}
    check(prec == {"detector_a": 0.5, "detector_b": 0.5}, f"precision per source {prec}")
    site = st["by_site"][0]
    check(site["hits"] == 2 and site["added"] == 1 and abs(site["recall"] - 2 / 3) < 1e-9,
          f"recall = hits/(hits+added) = {site['recall']:.2f}")
    check(st["classes"] == {"Small boat": 1, "unclassified": 1, "Large vessel": 1}, f"class counts {st['classes']}")
    check(st["series"]["harbour_b"] == [("2025-03-01", 3, True, True)], "time series point")
    out, s = pj.export_labels(proj, tmp / "exp_cls", yolo_classes=True)
    ids = sorted(int(l.split()[0]) for l in (out / "yolo/labels/harbour_b__S2A_18QYF_20250301_0_L2A.txt")
                 .read_text().split("\n") if l)
    names = proj.classes + ["unclassified"]
    check(ids == sorted([names.index("Small boat"), names.index("unclassified"), names.index("Large vessel")]),
          f"multi-class YOLO ids {ids}")
    check("4: unclassified" in (out / "yolo/data.yaml").read_text(), "data.yaml lists classes")
    proj.log_action("test", x=1)
    check((proj.review_dir / "history.jsonl").read_text().count("\n") == 1, "history.jsonl appended")


def test_debug_regressions(tmp):
    """Bugs found in a debugging pass -- each was reproduced before being fixed."""
    # degree-based raster (EPSG:4326): metres must be metres, and the view must not hang
    p = tmp / "geo" / "siteA" / "img_20250101.tif"
    p.parent.mkdir(parents=True)
    d = np.full((3, 400, 400), 500, np.uint16)
    d[:, 200:206, 200:206] = 4000
    tfm = from_origin(-72.40, 18.60, 0.00009, 0.00009)
    with rasterio.open(p, "w", driver="GTiff", width=400, height=400, count=3, dtype="uint16", crs="EPSG:4326",
                       transform=tfm) as dst:
        dst.write(d)
        dst.descriptions = ("red", "green", "blue")
    c = pj.scan_imagery(tmp / "geo")[0]
    check(9 < c["res_m"] < 10.5 and 3500 < c["extent_m"][0] < 4200, f"EPSG:4326 res/extent in metres ({c['res_m']:.1f} m)")
    lon, lat = tfm * (203, 203)
    v = pj.render_view(str(p), lon, lat, 150)
    a = np.asarray(v["img"]).sum(axis=2)
    rows, cols = np.nonzero(a > a.max() / 2)
    x, y = v["transform"] * (cols.mean() + 0.5, rows.mean() + 0.5)
    tl = Transformer.from_crs(v["crs"], "EPSG:4326", always_xy=True).transform(x, y)
    err = np.hypot((tl[0] - lon) * 111320 * np.cos(np.radians(lat)), (tl[1] - lat) * 110570)
    check(25 <= v["img"].size[0] <= 40 and err < 15, f"EPSG:4326 150 m view: {v['img'].size[0]} px, target within {err:.0f} m")
    edge = pj.render_view(str(p), *(tfm * (0, 0)), 150)   # view hanging off the image corner
    check(edge is not None and np.asarray(edge["img"])[:5, :5].sum() == 0, "EPSG:4326 view off the edge reads as black")
    native = Transformer.from_crs(v["crs"], v["native_crs"], always_xy=True)
    col, row = ~v["native_transform"] * native.transform(x, y)
    check(abs(col - 203) < 1.5 and abs(row - 203) < 1.5, f"pixel_box maps back to SOURCE pixels ({col:.1f}, {row:.1f})")
    crs, b = pj.image_bounds(str(p))
    check(not crs.is_geographic and b[2] - b[0] > 3000, "minimap bounds in metres")
    # odd geometries + near-duplicate detections
    g = tmp / "geodets" / "siteA"
    g.mkdir(parents=True)
    feats = [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[lon, lat], [lon + 1e-4, lat]]}},
             {"type": "Feature", "geometry": {"type": "MultiPoint", "coordinates": [[lon - 2e-4, lat]]}},
             {"type": "Feature", "geometry": None},
             {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat + 3e-4]}},
             {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon + 3e-5, lat + 3e-4]}},
             {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat + 3e-4]}}]   # exact dup
    for f in feats:
        f["properties"] = {}
    (g / "img_20250101_docked.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    pr = pj.Project.create(tmp / "projects", "Regress")
    pr.add_imagery(pj.scan_imagery(tmp / "geo", tmp / "geodets"))
    q = [it for it in pj.build_queue(pr) if it["kind"] == "detection"]
    ids = [it["id"] for it in q]
    check(len(q) == 4 and len(set(ids)) == 4, f"line + multipoint + 2 points 3 m apart queued, exact dup merged ({len(q)})")
    check(ids[2] == pj.detection_id(q[2]["site"], "img_20250101", lon, lat + 3e-4) and ids[3] == ids[2] + "#2",
          "colliding id gets #2; the first keeps the review_tool-compatible id")
    check(assist._k_center(np.zeros((5, 3)), 0) == [], "k-center with n=0 picks nothing")




def test_migrates_format1_project(tmp):
    """A project written by the original single-project tool opens, migrates, keeps backups."""
    root = tmp / "old"
    (root / "review_data").mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({
        "name": "old", "settings": {"vessel_classes": ["Big", "Small"], "assist": {"object_name": "ship"}},
        "imagery": []}))
    (root / "review_data" / "review_state.json").write_text(json.dumps({"a|optical|x|1_2": {"status": "valid",
                                                                                           "vessel_class": "Big"}}))
    (root / "review_data" / "manual_additions.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": None, "properties": {"vessel_class": "Small"}}]}))
    p = pj.Project(root)
    check(p.classes == ["Big", "Small"] and p.object_name == "ship", "settings migrated")
    check(p.state["a|optical|x|1_2"][pj.CLASS_KEY] == "Big", "record type migrated")
    check(p.additions["features"][0]["properties"][pj.CLASS_KEY] == "Small", "added-box type migrated")
    check(json.loads((root / "project.json").read_text())["format_version"] == 2, "format_version written")
    check((root / "project.json.v1.bak").exists() and (root / "review_data" / "review_state.json.v1.bak").exists(),
          "originals backed up")
    check("vessel_classes" not in json.loads((root / "project.json").read_text())["settings"], "old key removed")


def test_area_is_first_subfolder(tmp):
    root = tmp / "imagery"
    for rel in ("north/2024/a.tif", "south/b.tif", "c.tif"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).touch()
    check(pj.infer_site(root / "north/2024/a.tif", root) == "north", "nested -> first subfolder")
    check(pj.infer_site(root / "south/b.tif", root) == "south", "subfolder")
    check(pj.infer_site(root / "c.tif", root) == "imagery", "top level -> folder's own name")


def test_core_never_imports_deep_learning():
    """The MIT core must not import the AGPL training stack (or torch) on its own."""
    import subprocess
    import sys
    code = ("import sys, boxtrot.core.assist, boxtrot.core.backends, boxtrot.ui.app; "
            "bad = [m for m in ('ultralytics', 'torch') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check(r.returncode == 0, f"imported: {r.stdout.strip()} {r.stderr[-300:]}")
