"""Model-assisted labelling, end to end, with the dependency-free oracle backend
(boxtrot.testing.OracleBackend): hold-out blindness, recommendations, suggestions, dataset
tiling, training via a backend, hold-out scoring, pre-labels landing on the true objects,
multi-class catch-all, and behaviour when no backend is installed."""
import json
from pathlib import Path

import pytest
from conftest import check
from rasterio import open as ropen

from boxtrot.core import assist, backends
from boxtrot.core import project as pj
from boxtrot.testing import label_completely, make_scene_set

ORACLE = "boxtrot.testing:OracleBackend"


@pytest.fixture
def scenes(tmp):
    truth = make_scene_set(tmp / "imgs", n=36, seed=7)
    proj = pj.Project.create(tmp / "projects", "Assist")
    proj.settings["assist"] = {"tile_px": 192, "tile_overlap": 32, "epochs": 3}
    proj.save_meta()
    proj.add_imagery(pj.scan_imagery(tmp / "imgs"))
    return proj, truth


def test_full_loop_with_oracle_backend(scenes, tmp, monkeypatch):
    monkeypatch.setenv("BOXTROT_BACKEND", ORACLE)
    proj, truth = scenes
    adv = assist.recommend(proj)
    check(adv["steps"][0]["name"] == "Hold-out set" and adv["steps"][0]["status"] == "now", "first step: hold-out")
    hold = assist.pick_holdout(proj, n=6)
    check(len(set(hold)) == 6, "6 distinct hold-out images")
    # a detection file on a hold-out image must stay hidden (blind)
    he = next(e for e in proj.imagery if pj.image_key(e) == hold[0])
    fake = tmp / "fake.geojson"
    fake.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {
        "type": "Point", "coordinates": he["center"]}, "properties": {}}]}))
    he["detections"] = [str(fake)]
    proj.save_meta()
    check([it["kind"] for it in pj.build_queue(proj) if it["image_key"] == hold[0]] == ["scene"], "hold-out is blind")
    for k in hold:
        label_completely(proj, k, truth)
    check(assist.recommend(proj)["steps"][0]["status"] == "done", "hold-out step done")
    sugg = assist.suggest_images(proj, n=24)
    check(len(sugg) == 24 and not set(sugg) & set(hold), "24 suggestions outside the hold-out")
    for k in sugg:
        label_completely(proj, k, truth)
    inv = assist.inventory(proj)
    check(len(inv["train_images"]) == 24 and len(inv["untouched"]) == 6, "24 training images, 6 untouched")
    check(assist.auto_upscale(inv["median_box_px"]) >= 2, "small objects -> tiles upscaled")

    v = assist.next_version(proj)
    assist.run_train(str(proj.root), v)
    rec = assist.load_assist(proj)["versions"][-1]
    check(rec["status"] == "done" and rec["backend"] == "oracle", f"trained: {rec.get('error')}")
    best = rec["metrics"]["best"]
    check(rec["metrics"]["n_objects"] == sum(len(truth[k.split("|")[1]][0]) for k in hold), "all hold-out objects scored")
    check(best["r"] >= 0.9 and best["p"] >= 0.9, f"oracle scores high on the hold-out: {best}")

    assist.run_predict(str(proj.root), v)
    p2 = pj.Project(proj.root)
    pre = pj.prelabel_files(p2)
    check(set(pre) == set(inv["untouched"]), "pre-labels written for exactly the untouched images")
    check(all(not any(assist.is_model_file(d) for d in e.get("detections") or []) for e in p2.imagery),
          "project.json left alone")
    hits = total = 0
    for k, f in pre.items():
        e = next(x for x in p2.imagery if pj.image_key(x) == k)
        with ropen(e["path"]) as src:
            pred = [b[:4] for b in assist.boxes_px(src, json.loads(Path(f).read_text())["features"])]
        gts = truth[e["item_id"]][0]
        tp, _, _ = assist.match([(*b, 1.0) for b in pred], gts, 0)
        hits, total = hits + tp, total + len(gts)
    check(total and hits / total >= 0.9, f"pre-labels hit {hits}/{total} true objects (georeferencing + tiling)")
    q = pj.build_queue(p2)
    check(any((it.get("source") or "").startswith("model_v") and it.get("score") for it in q), "pre-labels queued")


def test_multiclass_keeps_untyped_objects_as_other(scenes, monkeypatch):
    monkeypatch.setenv("BOXTROT_BACKEND", ORACLE)
    proj, truth = scenes
    assist.pick_holdout(proj, n=4)
    for k in assist.suggest_images(proj, n=20):
        label_completely(proj, k, truth)
    for i, f in enumerate(proj.additions["features"]):
        if i % 3 == 0:
            f["properties"][pj.CLASS_KEY] = "Large" if i % 2 == 0 else "Small"
    proj.save_additions()
    proj.settings["assist"]["min_per_class"] = 3
    proj.save_meta()
    check(len(assist.recommend(proj)["train_classes"]) == 2, "two types trained")
    v = assist.next_version(proj)
    assist.run_train(str(proj.root), v)
    rec = assist.load_assist(proj)["versions"][-1]
    check(rec["classes"][-1] == assist.OTHER_CLASS, f"catch-all class: {rec['classes']}")
    lbl = assist.version_dir(proj, v) / "dataset" / "train" / "labels"
    other = sum(1 for f in lbl.glob("*.txt") for line in f.read_text().splitlines() if line.startswith("2 "))
    check(other > 0, "untyped objects labelled 'other', not left as background")


def test_without_backend(scenes, monkeypatch):
    monkeypatch.delenv("BOXTROT_BACKEND", raising=False)
    monkeypatch.setattr(backends, "available", lambda: {})
    proj, truth = scenes
    check(not backends.is_available(), "no backend")
    with pytest.raises(backends.BackendMissing):
        backends.get_backend()
    assist.pick_holdout(proj, n=4)
    check(assist.suggest_images(proj, n=5), "suggestions work without a backend")
    v = assist.next_version(proj)
    assist.run_train(str(proj.root), v)
    check(assist.read_status(proj, v)["stage"] == "failed", "train fails cleanly, with a status, not a crash")
