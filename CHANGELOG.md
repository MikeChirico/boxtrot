# Changelog

## 0.1.0 (unreleased)

First public version, generalised from a single-project ship-review tool.

- Project-based app with a 5-step setup wizard; any georeferenced raster; optional GeoJSON detections
  matched by name and area folder.
- Review screen:
  - decisions, types (number keys), corner-handle box editing, missed-object boxes
  - whole-image scene checks
  - undo for every action, plus an audit log
- Review context: band views, other-date flick and side-by-side, brightness/contrast, overview map,
  score ordering and source filter.
- Stats tab; export to GeoJSON, CSV and YOLO (single class or one class per type).
- Model-assisted labelling: blind hold-out, suggestions, learning curve, retrain/plateau advice.
  Training and inference are plugins; the Ultralytics YOLO backend is a separate AGPL-3.0 add-on.
- Geographic-CRS rasters are viewed in local UTM. Project format 2, with automatic migration and
  backups of format-1 projects.
