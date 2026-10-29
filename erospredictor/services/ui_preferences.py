"""Validated per-computer preferences; credentials and draft contents are excluded."""

from services import client_settings

BOOLEAN_KEYS = {"auto_draft", "off_meta", "simple_effects", "robust_roles",
                "history_markers", "history_service", "pool_only"}


def validated(values):
    if not isinstance(values, dict):
        return {}
    result = {}
    for key, value in values.items():
        if key in BOOLEAN_KEYS and isinstance(value, bool):
            result[key] = value
        elif key == "history_limit" and type(value) is int and value in (100, 500, 1000):
            result[key] = value
        elif key == "sync_side" and value in (None, "blue", "red"):
            result[key] = value
        elif key in ("preferred_role", "ban_role") and type(value) is int and 0 <= value < 5:
            result[key] = value
        elif key in ("active_pool", "tier") and isinstance(value, str) and 0 < len(value) <= 64 and "\0" not in value:
            result[key] = value
    return result


def load_ui_preferences():
    return validated(client_settings._read_settings().get("ui_preferences"))


def save_ui_preferences(changes):
    changes = validated(changes)
    return not changes or client_settings.update_settings(changes, section="ui_preferences")
