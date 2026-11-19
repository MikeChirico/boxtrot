"""Per-user settings that don't belong to any one project: config folder, recent projects,
default location for new projects. Plain JSON in the platform's usual config folder."""
import json
import os
import sys
from pathlib import Path

from . import APP_NAME, APP_SLUG


def user_config_dir():
    """~/.config/<slug> (Linux, honours XDG_CONFIG_HOME), %APPDATA%\\<slug> (Windows),
    ~/Library/Application Support/<slug> (macOS). Override with BOXTROT_CONFIG_DIR."""
    if os.environ.get("BOXTROT_CONFIG_DIR"):
        return Path(os.environ["BOXTROT_CONFIG_DIR"])
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_SLUG


def _settings_path():
    return user_config_dir() / "settings.json"


def load_user_settings():
    try:
        return json.loads(_settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_user_settings(s):
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def default_projects_dir():
    return Path(load_user_settings().get("projects_dir") or Path.home() / f"{APP_NAME} projects")


def remember_projects_dir(path):
    s = load_user_settings()
    s["projects_dir"] = str(Path(path))
    save_user_settings(s)


def recent_sources():
    """[(imagery_dir, detections_dir_or_None)] used recently, newest first (offered as presets)."""
    out = []
    for r in load_user_settings().get("recent_sources", []):
        if isinstance(r, list) and len(r) == 2 and r[0] and Path(r[0]).exists():
            out.append((r[0], r[1] if r[1] and Path(r[1]).exists() else None))
    return out


def remember_source(img_dir, det_dir=None):
    s = load_user_settings()
    item = [str(Path(img_dir)), str(Path(det_dir)) if det_dir else None]
    s["recent_sources"] = [item] + [r for r in s.get("recent_sources", []) if r != item][:9]
    save_user_settings(s)
