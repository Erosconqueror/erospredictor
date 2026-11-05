"""Local-only Riot credentials; never log values or put them in checkpoints."""
import json
import os
from pathlib import Path

KEYS_PATH = Path(__file__).resolve().parents[2] / "secrets.local.json"
PROFILES = ("default", "development", "development2", "development3")


def get_api_key(profile="default", path=None):
    if profile not in PROFILES:
        raise ValueError("Unknown API key profile")
    variable = {"default": "RIOT_API_KEY", "development": "RIOT_DEV_API_KEY",
                "development2": "RIOT_DEV_API_KEY_2", "development3": "RIOT_DEV_API_KEY_3"}[profile]
    if os.environ.get(variable):
        return os.environ[variable].strip()
    try:
        document = json.loads(Path(path or KEYS_PATH).read_text(encoding="utf-8"))
        value = document.get("riot", {}).get(profile, "")
        return value.strip() if isinstance(value, str) else ""
    except (OSError, ValueError, AttributeError):
        return ""
