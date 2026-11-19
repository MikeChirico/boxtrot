# boxtrot-ultralytics

Training backend for **boxtrot**'s model-assisted labelling, using
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

```
pip install ./addons/boxtrot-ultralytics     # from a boxtrot checkout
```

Once installed, boxtrot finds it automatically (entry point `boxtrot.trainers`), and
**Train model** / **Predict** become available on a project's Model tab.

## License: AGPL-3.0

This add-on is licensed under the **GNU AGPL-3.0** (see `LICENSE`), because Ultralytics is
AGPL-3.0. boxtrot itself is MIT-licensed and works without this add-on. If you distribute
this add-on, or provide it as a network service, the AGPL's obligations apply to it. Ultralytics
also offers commercial licenses. The default base model, `yolov8n.pt`, is downloaded by
Ultralytics on first use and is under the same terms.

## Writing another backend

Any package can provide a backend. Implement the interface described in
`boxtrot/core/backends.py` and register it under the `boxtrot.trainers` entry point
(see this package's `pyproject.toml`). That's how to plug in a permissively licensed
detector instead.
