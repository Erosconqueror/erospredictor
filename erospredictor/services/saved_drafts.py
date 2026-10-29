"""Portable draft JSON with stable Riot IDs; no account data or model results."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from configs import TARGET_TIERS
from services.client_settings import settings_path
from services.role_distribution import ROLE_NAMES
from services.update_progress import atomic_json

FORMAT = "erospredictor-draft"
MAX_FILE_BYTES = 64 * 1024
SLOTS = ("blue", "red", "blue_bans", "red_bans")


class DraftFileError(ValueError):
    pass


@dataclass(frozen=True)
class SavedDraft:
    values: dict
    tier: str
    team: str
    role: int


def drafts_directory():
    return settings_path().parent / "drafts"


def _validate(values, tier, team, role, *, internal=False):
    if not isinstance(values, dict):
        raise DraftFileError("A mentett draft hőslistái hiányosak.")
    rows = {}
    for key in SLOTS:
        row = values.get(key)
        if not isinstance(row, list) or len(row) != 5:
            raise DraftFileError("Minden csapathoz és tiltáslistához pontosan öt hely szükséges.")
        if any(value is not None and (type(value) is not int or value < (0 if internal else 1)) for value in row):
            raise DraftFileError("A draft érvénytelen hősazonosítót tartalmaz.")
        rows[key] = list(row)
    picks = [value for key in ("blue", "red") for value in rows[key] if value is not None]
    if len(picks) != len(set(picks)):
        raise DraftFileError("Ugyanaz a hős nem szerepelhet kétszer a draftban.")
    for key in ("blue_bans", "red_bans"):
        bans = [value for value in rows[key] if value is not None]
        if len(bans) != len(set(bans)):
            raise DraftFileError("Egy csapat tiltáslistája nem tartalmazhat ismétlődő hőst.")
        if set(bans).intersection(picks):
            raise DraftFileError("Kiválasztott hős nem szerepelhet a tiltások között.")
    if tier not in ("MIXED", *TARGET_TIERS):
        raise DraftFileError("A draft rangbeállítása nem támogatott.")
    if team not in ("blue", "red") or type(role) is not int or not 0 <= role < 5:
        raise DraftFileError("A draft fókuszcsapata vagy pozíciója érvénytelen.")
    return rows


def _catalog(mapping):
    try:
        pairs = [(int(riot), index) for riot, index in mapping.items()]
    except (AttributeError, TypeError, ValueError):
        raise DraftFileError("A helyi hőskatalógus nem olvasható.") from None
    if (any(riot <= 0 or type(index) is not int or index < 0 for riot, index in pairs)
            or len({riot for riot, _ in pairs}) != len(pairs)
            or len({index for _, index in pairs}) != len(pairs)):
        raise DraftFileError("A helyi hőskatalógus azonosítói nem egyértelműek.")
    return dict(pairs)


def make_document(values, mapping, tier, team, role, names=None):
    if not isinstance(values, dict) or any(not isinstance(values.get(key), list) for key in SLOTS):
        raise DraftFileError("A mentendő draft hőslistái hiányosak.")
    normalized = {key: [None if value == -1 and type(value) is int else value for value in values.get(key, [])]
                  for key in SLOTS}
    normalized = _validate(normalized, tier, team, role, internal=True)
    inverse = {index: riot for riot, index in _catalog(mapping).items()}
    used = {cid for row in normalized.values() for cid in row if cid is not None}
    if not used.issubset(inverse):
        raise DraftFileError("A draft egyik hőse nincs a helyi katalógusban; mentés nem történt.")
    document = {"format": FORMAT, "version": 1, "roles": list(ROLE_NAMES),
        "saved_at": datetime.now(timezone.utc).isoformat(), "tier": tier,
        "focus": {"team": team, "role": role},
        "draft": {key: [inverse[cid] if cid is not None else None for cid in row] for key, row in normalized.items()}}
    if names:
        document["champion_names"] = {str(inverse[cid]): names[cid] for cid in sorted(used)
                                     if cid in names and isinstance(names[cid], str)}
    return document


def _document_state(document):
    if (not isinstance(document, dict) or document.get("format") != FORMAT
            or type(document.get("version")) is not int or document["version"] != 1
            or document.get("roles") != list(ROLE_NAMES)):
        raise DraftFileError("Ez nem támogatott Erospredictor-draftfájl.")
    focus = document.get("focus")
    if not isinstance(focus, dict):
        raise DraftFileError("A draft fókuszbeállítása hiányzik.")
    tier, team, role = document.get("tier"), focus.get("team"), focus.get("role")
    return _validate(document.get("draft"), tier, team, role), tier, team, role


def save_draft(path, document):
    rows, tier, team, role = _document_state(document)
    # Explicit allowlist: never export preferences, account identifiers or tokens.
    output = {"format": FORMAT, "version": 1, "roles": list(ROLE_NAMES),
              "saved_at": datetime.now(timezone.utc).isoformat(), "tier": tier,
              "focus": {"team": team, "role": role}, "draft": rows}
    used = {str(cid) for row in rows.values() for cid in row if cid is not None}
    labels = document.get("champion_names")
    if isinstance(labels, dict):
        output["champion_names"] = {cid: label for cid, label in labels.items()
                                    if cid in used and isinstance(label, str) and len(label) <= 100}
    try:
        atomic_json(Path(path), output)
    except (OSError, ValueError):
        raise DraftFileError("A draft nem menthető a kiválasztott helyre. A korábbi mentés megmaradt.") from None


def load_draft(path, mapping, *, available_ids=None):
    try:
        path = Path(path)
        with path.open("rb") as source:
            contents = source.read(MAX_FILE_BYTES + 1)
        if len(contents) > MAX_FILE_BYTES:
            raise DraftFileError("A draftfájl túl nagy; legfeljebb 64 KiB lehet.")
        document = json.loads(contents.decode("utf-8-sig"))
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        if isinstance(error, DraftFileError):
            raise
        raise DraftFileError("A kiválasztott draftfájl nem olvasható vagy hibás JSON-t tartalmaz.") from None
    rows, tier, team, role = _document_state(document)
    catalog = _catalog(mapping)
    used = {cid for row in rows.values() for cid in row if cid is not None}
    unknown = sorted(cid for cid in used if cid not in catalog
                     or (available_ids is not None and catalog[cid] not in available_ids))
    if unknown:
        raise DraftFileError("A draft hősei közül néhány nincs a jelenlegi katalógusban: " + ", ".join(map(str, unknown)) + ". Frissítsd a hősadatokat.")
    values = {key: [catalog[cid] if cid is not None else None for cid in row] for key, row in rows.items()}
    return SavedDraft(values, tier, team, role)
