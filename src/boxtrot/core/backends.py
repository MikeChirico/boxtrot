"""Training backends for model-assisted labelling (plugins).

The MIT-licensed core never imports a deep-learning library. A backend is a separate,
separately-licensed package that registers itself under the entry-point group
"boxtrot.trainers"; the reference one is addons/boxtrot-ultralytics (AGPL-3.0).

A backend is an object with:
    name: str                 short id, e.g. "ultralytics"
    label: str                shown in the UI, e.g. "YOLO (Ultralytics, AGPL-3.0)"
    default_weights: str      base model used when the project doesn't name one
    train(data_yaml, base_weights, epochs, imgsz, batch, out_dir, on_epoch) -> weights path
        data_yaml: a YOLO-format dataset written by core.assist.build_dataset
        on_epoch(epoch, epochs): progress callback
    load(weights) -> predictor with
        predict(tiles, imgsz, conf) -> [(boxes Nx4 xyxy, scores N, classes N), ...] per tile
        tiles: list of HxWx3 uint8 RGB arrays (exactly as the reviewer sees them)

Discovery order: BOXTROT_BACKEND="module:attribute" (for testing/custom backends),
then installed entry points.
"""
import importlib
import os
from importlib.metadata import entry_points

GROUP = "boxtrot.trainers"
INSTALL_HINT = ("Model training is an optional add-on (separately licensed, AGPL-3.0, because it uses "
                "Ultralytics YOLO). Install it with:\n\n    pip install boxtrot-ultralytics\n\n"
                "or from a checkout:  pip install ./addons/boxtrot-ultralytics\n\n"
                "Everything else -- hold-out, image suggestions, recommendations, review, export -- works "
                "without it.")


class BackendMissing(RuntimeError):
    pass


def _from_spec(spec):
    mod, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(mod), attr or "Backend")
    return obj() if isinstance(obj, type) else obj


def available():
    """{name: loader} of installed backends (not imported until used)."""
    found = {}
    if os.environ.get("BOXTROT_BACKEND"):
        found["env"] = lambda: _from_spec(os.environ["BOXTROT_BACKEND"])
    try:
        eps = entry_points(group=GROUP)
    except TypeError:   # Python < 3.10 API
        eps = entry_points().get(GROUP, [])
    for ep in eps:
        found.setdefault(ep.name, lambda ep=ep: (lambda o: o() if isinstance(o, type) else o)(ep.load()))
    return found


def get_backend(name=None):
    found = available()
    if not found:
        raise BackendMissing(INSTALL_HINT)
    key = name if name in found else next(iter(found))
    return found[key]()


def is_available():
    return bool(available())
