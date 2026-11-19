"""Ultralytics YOLO training backend for boxtrot's model-assisted labelling.

Licensed AGPL-3.0 (see LICENSE) because it uses Ultralytics, which is AGPL-3.0. The boxtrot
application itself is MIT and never imports this package directly: it is discovered through the
"boxtrot.trainers" entry point when installed.

Base weights: anything ultralytics.YOLO() accepts. The default, yolov8n.pt, is downloaded by
Ultralytics on first use (also AGPL-3.0).
"""
__version__ = "0.1.0"


def _resolve_weights(name):
    """A bare model name (e.g. "yolov8n.pt") is downloaded once into a per-user cache instead of
    the current folder (training runs inside each model-version folder, so it would otherwise be
    re-downloaded for every version). Paths that exist are used as-is."""
    import os
    from pathlib import Path
    p = Path(name)
    if p.exists() or p.parent != Path("."):
        return str(p)
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "boxtrot-ultralytics"
    target = cache / p.name
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics.utils.downloads import attempt_download_asset
            attempt_download_asset(str(target))
        except Exception:   # offline / unknown name: let Ultralytics resolve it its own way
            return name
    return str(target) if target.exists() else name


class UltralyticsBackend:
    name = "ultralytics"
    label = "YOLO (Ultralytics, AGPL-3.0 add-on)"
    default_weights = "yolov8n.pt"

    def train(self, data_yaml, base_weights, epochs, imgsz, batch, out_dir, on_epoch=None):
        """Fine-tune on a YOLO-format dataset; returns the path of the best weights."""
        from pathlib import Path

        import torch
        from ultralytics import YOLO
        device = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(_resolve_weights(base_weights))
        if on_epoch is not None:
            model.add_callback("on_fit_epoch_end", lambda tr: on_epoch(int(tr.epoch) + 1, int(tr.epochs)))
        model.train(data=str(data_yaml), epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), device=device,
                    project=str(out_dir), name="train", exist_ok=True, patience=max(15, int(epochs) // 3),
                    workers=2, plots=False, verbose=False,
                    # overhead imagery has no "up": rotate/flip freely
                    mosaic=1.0, degrees=90, flipud=0.5, fliplr=0.5)
        best = Path(out_dir) / "train" / "weights" / "best.pt"
        return best if best.exists() else Path(out_dir) / "train" / "weights" / "last.pt"

    def load(self, weights):
        return _Predictor(weights)


class _Predictor:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))

    def predict(self, tiles, imgsz, conf):
        """tiles: RGB uint8 arrays -> [(boxes Nx4 xyxy, scores N, classes N)] per tile."""
        # Ultralytics treats numpy input as BGR (OpenCV convention); training PNGs are RGB on disk
        bgr = [t[:, :, ::-1].copy() for t in tiles]
        out = []
        for res in self.model.predict(bgr, imgsz=int(imgsz), conf=float(conf), verbose=False):
            b = res.boxes
            if b is None or not len(b):
                out.append(([], [], []))
            else:
                out.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()))
        return out
