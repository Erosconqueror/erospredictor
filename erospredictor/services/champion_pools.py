"""Machine-local champion lists with stable Riot IDs, independent of history."""

from copy import deepcopy
import json
from pathlib import Path

from services.client_settings import settings_path
from services.update_progress import atomic_json

TAGS = {"played": "Játszom", "comfort": "Komfort", "learn": "Tanulnám"}
ALL_MINE = "Összes saját hős"


def library_path():
    return settings_path().parent / "champion_pools.json"


def empty_library():
    return {"version": 1, "champions": {}, "lists": {}}


def valid_id(value):
    return isinstance(value, str) and value.isascii() and value.isdigit() and 0 < int(value) < 100000


def validate_library(document):
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("A hőslisták formátuma érvénytelen.")
    champions, lists = document.get("champions"), document.get("lists")
    if not isinstance(champions, dict) or not isinstance(lists, dict) or len(lists) > 50:
        raise ValueError("A hőslisták formátuma érvénytelen.")
    for champion, tags in champions.items():
        if not valid_id(champion) or not isinstance(tags, list) or any(tag not in TAGS for tag in tags):
            raise ValueError("Érvénytelen hősjelölés.")
    for name, ids in lists.items():
        if (not isinstance(name, str) or not name.strip() or len(name) > 64 or name == ALL_MINE
                or not isinstance(ids, list) or any(not valid_id(champion) for champion in ids)):
            raise ValueError("Érvénytelen saját lista.")
    return document


def load_library(path=None):
    try:
        return validate_library(json.loads(Path(path or library_path()).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, UnicodeError):
        return empty_library()


def save_library(document, path=None):
    document = deepcopy(validate_library(document))
    document["champions"] = {cid: sorted(set(tags)) for cid, tags in document["champions"].items() if tags}
    document["lists"] = {name: list(dict.fromkeys(ids)) for name, ids in document["lists"].items()}
    atomic_json(Path(path or library_path()), document)


def member_ids(document, mapping, list_name=ALL_MINE):
    riot_ids = ({cid for cid, tags in document["champions"].items() if tags}
                if list_name == ALL_MINE else set(document["lists"].get(list_name, [])))
    return {int(mapping[cid]) for cid in riot_ids if cid in mapping}


def internal_tags(document, mapping):
    return {int(mapping[cid]): tuple(tag for tag in TAGS if tag in tags)
            for cid, tags in document["champions"].items() if cid in mapping and tags}


def include_history(document, history, mapping):
    """Explicit import of already-loaded history; no network or account lookup."""
    result = deepcopy(document)
    known = set(mapping)
    for game in history.games if history is not None else ():
        champion = str(game["champion"])
        if champion in known:
            tags = result["champions"].setdefault(champion, [])
            if "played" not in tags:
                tags.append("played")
    return result
