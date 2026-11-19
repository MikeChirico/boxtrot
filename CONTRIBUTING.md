# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
conda env create -f environment.yml && conda activate boxtrot
pip install -e ".[dev]"
pytest -m "not gui" && ruff check src tests scripts addons
```

## Ground rules

- **The core stays MIT and ML-free.** Nothing under `src/boxtrot/` may import a deep-learning
  library or AGPL code; `tests/test_core.py::test_core_never_imports_deep_learning` enforces it.
  Training and inference go through the backend interface (`core/backends.py`).
- **`core/` never imports `ui/`.** The core must stay usable without a display.
- **Never lose review work.** Every action is saved immediately with write-then-rename, and every
  change to `project.json` / `review_data/` formats needs a migration in `Project._migrate` plus a
  test (see `test_migrates_format1_project`). Bump `PROJECT_FORMAT_VERSION` when the format changes.
- **Reproduce before fixing.** For bugs, add a failing test first. `boxtrot.testing` has synthetic
  georeferenced scenes with known objects and an oracle backend for that.
- **UI text is plain ASCII where Tk fonts may lack glyphs** (some Linux Tk builds render symbols such
  as ▾ ⚙ ✕ as `\uXXXX`).

## Tests

| command | covers |
|---|---|
| `pytest -m "not gui"` | scanning, georeferencing (projected + geographic), rendering, queue, export, stats, migration, the whole assisted-labelling pipeline with the oracle backend |
| `pytest -m gui` | drives the real app; fails on any exception inside a Tk callback |
| `pytest addons/boxtrot-ultralytics/tests` | a real YOLO training run (a few minutes; GPU if present) |

## Renaming (forks)

The user-visible name and the package name are kept in few places. To rename a fork:

1. `src/boxtrot/__init__.py`: set `APP_NAME` (shown in the UI) and `APP_SLUG` (config folder).
2. Rename the package folder `src/boxtrot` → `src/<newname>`, and replace `boxtrot` in imports,
   `pyproject.toml` (`name`, `[project.gui-scripts]`, version attribute), the module path in
   `ui/model_panel.py` (`-m boxtrot.core.assist`), and `tests/`.
3. Entry-point group `boxtrot.trainers` (`core/backends.py` and the add-on's `pyproject.toml`),
   the add-on package name, and the environment variables `BOXTROT_BACKEND` / `BOXTROT_CONFIG_DIR`.
4. `grep -ri boxtrot .` should then only show what you intend to keep.
