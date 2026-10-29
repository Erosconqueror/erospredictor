"""Local client location preferences, separate from shared application files."""

import json
import os
from pathlib import Path
import tempfile
from threading import RLock


DEFAULT_LOCKFILE = Path("C:/Riot Games/League of Legends/lockfile")
_settings_lock = RLock()


def settings_path():
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".config")
    return root / "Erospredictor" / "settings.json"


def _read_settings():
    try:
        value = json.loads(settings_path().read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def normalize_lockfile(path):
    candidate = Path(os.path.expandvars(str(path))).expanduser()
    if candidate.is_dir():
        candidate = candidate / "lockfile"
    return candidate.absolute()


def saved_lockfile():
    value = _read_settings().get("lcu_lockfile")
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        return None
    try:
        return normalize_lockfile(value)
    except (OSError, ValueError):
        return None


def preferred_lockfile():
    supplied = os.environ.get("EROSPREDICTOR_LCU_LOCKFILE")
    return normalize_lockfile(supplied) if supplied else saved_lockfile() or DEFAULT_LOCKFILE


def save_lockfile_path(path):
    """Atomically store only the location; tokens and ports are never persisted."""
    try:
        return update_settings({"lcu_lockfile": str(normalize_lockfile(path))})
    except (OSError, ValueError):
        return False


def update_settings(changes, *, section=None):
    """Serialize read/modify/write across GUI and client-reader threads."""
    with _settings_lock:
        return _update_settings(changes, section=section)


def _update_settings(changes, *, section=None):
    temporary = None
    try:
        destination = settings_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = _read_settings()
        if section is None:
            values.update(changes)
        else:
            current = values.get(section)
            values[section] = dict(current if isinstance(current, dict) else {}, **changes)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix="settings-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(values, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, destination)
        return True
    except (OSError, ValueError):
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
