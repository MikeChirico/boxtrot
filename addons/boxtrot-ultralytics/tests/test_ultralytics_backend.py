"""Real (short) training run through the boxtrot pipeline. Skipped without ultralytics.
Run:  pytest addons/boxtrot-ultralytics/tests -q   (takes a few minutes; GPU if available)"""
import pytest

pytest.importorskip("ultralytics")


def test_train_and_predict_on_synthetic_ships(tmp_path, monkeypatch):
    from boxtrot.core import assist
    from boxtrot.core import project as pj
    from boxtrot.testing import label_completely, make_scene_set

    monkeypatch.setenv("BOXTROT_BACKEND", "boxtrot_ultralytics:UltralyticsBackend")
    truth = make_scene_set(tmp_path / "imgs", n=30, seed=7)
    proj = pj.Project.create(tmp_path / "projects", "addon")
    proj.settings["assist"] = {"tile_px": 192, "tile_overlap": 32, "epochs": 20}
    proj.save_meta()
    proj.add_imagery(pj.scan_imagery(tmp_path / "imgs"))
    hold = assist.pick_holdout(proj, n=5)
    for k in hold + assist.suggest_images(proj, n=20):
        label_completely(proj, k, truth)
    v = assist.next_version(proj)
    assist.run_train(str(proj.root), v)
    rec = assist.load_assist(proj)["versions"][-1]
    assert rec["status"] == "done", rec.get("error")
    best = rec["metrics"]["best"]
    assert best["r"] >= 0.7 and best["p"] >= 0.5, best
    assist.run_predict(str(proj.root), v)
    assert pj.prelabel_files(pj.Project(proj.root))
