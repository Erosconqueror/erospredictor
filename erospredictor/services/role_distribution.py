"""Observed P(role | champion), using Riot IDs and role-ordered match records.

No ML packages, client access or API key are needed. Unknown or damaged data
returns a uniform distribution with a zero sample size, never a fabricated role.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from services.update_progress import atomic_json


ROLE_NAMES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data/champion_roles.json"


def _champion_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value > 0 else None
    if isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value.strip()):
        return str(int(value))
    return None


def _team(value):
    if isinstance(value, str):
        value = value.strip().strip("{}").split(",")
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        return None
    result = [_champion_id(item) for item in value]
    return result if all(result) else None


def empty_role_distribution():
    return {"version": 1, "roles": list(ROLE_NAMES), "scope": {"matches": 0, "tiers": []},
            "tiers": {"MIXED": {}}, "tier_scopes": {}}


def build_role_distribution(records, *, patches=None, region=None, tiers=None, source="training_snapshot"):
    """Count both teams once; role index is the DB's stored lane order.

    Counts are unweighted observations, independent of the winner. Optional
    filters use exact patch, platform region and tier values. Duplicate IDs are
    ignored within a region, and an invalid team invalidates the whole match.
    """
    patches = list(dict.fromkeys(patches)) if patches is not None else None
    tiers = list(dict.fromkeys(tier for tier in tiers if tier != "MIXED")) if tiers is not None else None
    patch_filter, tier_filter = set(patches or []), set(tiers or [])
    counts, matched, excluded, seen = {}, Counter(), Counter(), set()
    actual_patches = set()
    for row in records:
        if not isinstance(row, dict):
            excluded["invalid_record"] += 1
            continue
        tier = row.get("tier")
        if (not isinstance(tier, str) or not tier or tier == "MIXED"
                or (tiers is not None and tier not in tier_filter)
                or (patches is not None and row.get("patch") not in patch_filter)
                or (region is not None and row.get("region") != region)):
            excluded["outside_scope"] += 1
            continue
        blue, red = _team(row.get("blue_team")), _team(row.get("red_team"))
        if blue is None or red is None or len(set(blue + red)) != 10:
            excluded["invalid_teams"] += 1
            continue
        match_id = row.get("match_id")
        key = (row.get("region"), str(match_id)) if match_id else None
        if key is not None and key in seen:
            excluded["duplicate_match"] += 1
            continue
        if key is not None:
            seen.add(key)
        if isinstance(row.get("patch"), str):
            actual_patches.add(row["patch"])
        for bucket in (tier, "MIXED"):
            by_champion = counts.setdefault(bucket, {})
            matched[bucket] += 1
            for team in (blue, red):
                for role, champion in enumerate(team):
                    by_champion.setdefault(champion, [0] * 5)[role] += 1
    requested_tiers = tiers if tiers is not None else sorted(key for key in counts if key != "MIXED")
    result = {}
    for tier in requested_tiers + ["MIXED"]:
        result[tier] = {}
        for champion, values in sorted(counts.get(tier, {}).items(), key=lambda pair: int(pair[0])):
            total = sum(values)
            result[tier][champion] = {"counts": values, "probabilities": [value / total for value in values], "total": total}
    scope = {"patches": patches if patches is not None else sorted(actual_patches), "region": region,
             "tiers": requested_tiers, "source": source, "matches": matched["MIXED"],
             "excluded_records": dict(excluded)}
    return {"version": 1, "roles": list(ROLE_NAMES), "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": scope, "tiers": result,
            "tier_scopes": {tier: dict(scope, matches=matched[tier], tiers=(requested_tiers if tier == "MIXED" else [tier]))
                            for tier in result}}


def _valid_entry(entry):
    if not isinstance(entry, dict):
        return False
    counts, probabilities, total = entry.get("counts"), entry.get("probabilities"), entry.get("total")
    if (not isinstance(counts, list) or len(counts) != 5
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
            or not isinstance(total, int) or isinstance(total, bool) or total <= 0 or sum(counts) != total
            or not isinstance(probabilities, list) or len(probabilities) != 5):
        return False
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1
               and math.isfinite(value) and abs(value - count / total) <= 1e-8
               for count, value in zip(counts, probabilities))


def _valid_document(document):
    return (isinstance(document, dict) and document.get("version") == 1
            and document.get("roles") == list(ROLE_NAMES) and isinstance(document.get("tiers"), dict)
            and all(isinstance(tier, str) and isinstance(champions, dict)
                    and all(_champion_id(champion) == champion and _valid_entry(entry)
                            for champion, entry in champions.items())
                    for tier, champions in document["tiers"].items()))


def load_role_distribution(path=None):
    """Read only; missing/corrupt files yield no observations and a safe fallback."""
    try:
        document = json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
        if _valid_document(document):
            return document
    except (OSError, ValueError, TypeError, UnicodeError):
        pass
    return empty_role_distribution()


def role_probabilities(document, riot_id, tier="MIXED"):
    """Use a tier's observations, then MIXED; uniform probabilities mean no sample."""
    champion = _champion_id(riot_id)
    if (champion is not None and isinstance(document, dict) and document.get("version") == 1
            and document.get("roles") == list(ROLE_NAMES)):
        tiers = document.get("tiers", {})
        if isinstance(tiers, dict):
            for key in dict.fromkeys((tier, "MIXED")):
                bucket = tiers.get(key, {})
                entry = bucket.get(champion) if isinstance(bucket, dict) else None
                if _valid_entry(entry):
                    return list(entry["probabilities"]), entry["total"]
    return [0.2] * 5, 0


def merge_role_distributions(previous, current, selected_tiers):
    """Preserve ranks untouched by a partial update; MIXED has the new scope."""
    if not _valid_document(current):
        raise ValueError("Invalid generated champion role distribution")
    if not _valid_document(previous):
        return current
    selected = set(selected_tiers) | {"MIXED"}
    preserved = {tier: values for tier, values in previous["tiers"].items() if tier not in selected}
    scopes = previous.get("tier_scopes", {})
    scopes = scopes if isinstance(scopes, dict) else {}
    return dict(current, tiers={**preserved, **current["tiers"]},
                scope=dict(current.get("scope", {}), preserved_tiers=sorted(preserved)),
                tier_scopes={**{tier: value for tier, value in scopes.items() if tier not in selected},
                             **current.get("tier_scopes", {})})


def save_role_distribution(path, document, *, selected_tiers=None):
    """Atomic standalone write; pipeline callers should direct this into staging."""
    if selected_tiers is not None:
        document = merge_role_distributions(load_role_distribution(path), document, selected_tiers)
    if not _valid_document(document):
        raise ValueError("Invalid champion role distribution")
    atomic_json(path, document)
    return document
