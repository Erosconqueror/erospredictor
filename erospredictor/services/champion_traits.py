"""Experimental descriptive profiles. Never imported by inference models.

Observed values are unweighted sums and valid sample counts, by patch/role/tier.
Gameplay presets are editable heuristics, not measured matchup evidence.
"""

import json
import math
from pathlib import Path
from services.update_progress import atomic_json

DATA = Path(__file__).resolve().parents[1] / "data"
LABELS = {"frontline": "Frontline", "engage": "Engage", "follow_up": "Follow-up",
          "peel": "Peel / disengage", "cc": "CC-erő", "cc_access": "CC alkalmazhatóság",
          "hard_cc": "Hard CC", "soft_cc": "Soft CC",
          "cc_use": "Mért CC-kihasználás",
          "burst": "Burst", "sustained": "Folyamatos sebzés", "percent_health": "% HP sebzés",
          "waveclear": "Waveclear", "poke": "Poke", "mobility": "Mobilitás",
          "early": "Early", "mid": "Mid game", "late": "Late game",
          "range": "Harci távolság", "reach": "Elérési távolság", "spacing": "Távolságtartás"}
# Reviewable starting points only. Empty fields remain unknown.
PRESETS = {
    "Tank": {"frontline": 4, "engage": 3, "peel": 3, "cc": 4, "range": 1},
    "Fighter": {"frontline": 3, "sustained": 4, "range": 1},
    "Assassin": {"burst": 4, "mobility": 4, "range": 2, "frontline": 1},
    "Mage": {"burst": 3, "waveclear": 4, "range": 4},
    "Marksman": {"sustained": 4, "range": 4, "frontline": 1},
    "Support": {"peel": 3},
}


def read_document(path, default=None):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else (default or {})
    except (OSError, ValueError):
        return default or {}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def extract_observations(info):
    """Extract no identities, builds or timelines. Missing values have no sample."""
    observations = []
    roles = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "UTILITY": 4}
    for player in info.get("participants", []):
        if not isinstance(player, dict) or player.get("teamPosition") not in roles:
            continue
        duration = player.get("timePlayed", info.get("gameDuration"))
        if not finite(duration) or duration < 300:
            continue
        minutes = duration / 60
        metrics = {}
        fields = {"cc_seconds_per_min": "timeCCingOthers", "damage_per_min": "totalDamageDealtToChampions",
                  "mitigated_per_min": "damageSelfMitigated", "taken_per_min": "totalDamageTaken",
                  "healing_per_min": "totalHeal", "ally_heal_per_min": "totalHealsOnTeammates",
                  "ally_shield_per_min": "totalDamageShieldedOnTeammates", "gold_per_min": "goldEarned"}
        fields.update({f"spell{slot}_casts_per_min": f"spell{slot}Casts" for slot in range(1, 5)})
        for name, field in fields.items():
            if finite(player.get(field)):
                metrics[name] = player[field] / minutes
        damage = player.get("totalDamageDealtToChampions")
        if finite(damage) and damage > 0:
            for name, field in (("physical_share", "physicalDamageDealtToChampions"),
                                ("magic_share", "magicDamageDealtToChampions"), ("true_share", "trueDamageDealtToChampions")):
                if finite(player.get(field)) and player[field] <= damage:
                    metrics[name] = player[field] / damage
        challenges = player.get("challenges") or {}
        if isinstance(challenges, dict) and finite(challenges.get("enemyChampionImmobilizations")):
            metrics["immobilizations_per_min"] = challenges["enemyChampionImmobilizations"] / minutes
        if isinstance(player.get("win"), bool):
            phase = "early" if minutes < 20 else "mid" if minutes < 30 else "late"
            metrics[f"observed_win_{phase}"] = float(player["win"])
        champion = player.get("championId")
        if isinstance(champion, int) and not isinstance(champion, bool) and champion > 0 and metrics:
            observations.append({"champion": champion, "role": roles[player["teamPosition"]],
                                 "metrics": {name: [value, 1] for name, value in metrics.items()}})
    return observations


def merge_metrics(first, second):
    result = {key: list(value) for key, value in first.items()}
    for key, (total, count) in second.items():
        previous = result.get(key, [0, 0])
        result[key] = [previous[0] + total, previous[1] + count]
    return result


def export_observations(db, patches, path, tiers=None, region=None):
    query = getattr(db, "get_trait_aggregates", None)
    if not callable(query):
        return False
    patches = sorted(set(patches), key=lambda p: tuple(map(int, p.split("."))))[-5:]
    rows = query(patches, tiers=tiers, region=region)
    atomic_json(path, {"version": 1, "patches": patches, "rows": rows,
                       "method": "unweighted_observation_sums", "region": region})
    return True


class TraitLibrary:
    def __init__(self, basics=None, observations=None, overrides=None):
        self.basics = basics if basics is not None else read_document(DATA / "champion_trait_basics.json")
        self.observations = observations if observations is not None else read_document(DATA / "champion_trait_observations.json")
        if overrides is None:
            from services.client_settings import settings_path
            overrides = read_document(settings_path().with_name("champion_trait_profiles.json"))
        self.overrides = overrides
        self.patches = sorted(set(self.observations.get("patches", [])), key=lambda p: tuple(map(int, p.split("."))))[-5:]
        self._measured = {}
        for row in self.observations.get("rows", []):
            if row.get("patch") not in self.patches:
                continue
            for tier in set((row.get("tier"), "MIXED")):
                key = (str(row.get("champion")), row.get("role"), tier)
                self._measured[key] = merge_metrics(self._measured.get(key, {}), row.get("metrics", {}))

    def measured(self, riot, role, tier):
        return self._measured.get((str(riot), role, tier), {})

    def profile(self, riot, role, tier="MIXED"):
        base = self.basics.get("champions", {}).get(str(riot), {})
        values, sources = {}, {}
        for tag in base.get("tags", []):
            for key, value in PRESETS.get(tag, {}).items():
                values[key] = max(values.get(key, 0), value)
                sources[key] = "osztályalapú kísérleti becslés"
        # Effective range is intentionally editable and distinct from attackrange.
        if base.get("name") in ("Zed", "Urgot"):
            values["range"] = 2
            sources["range"] = "kísérleti harcitáv-korrekció"
        measured = self.measured(riot, role, tier)
        cc_total, cc_samples = measured.get("cc_seconds_per_min", (0, 0))
        if cc_samples >= 30:
            values["cc_use"] = 1 + 4 * min(1, cc_total / cc_samples / 6)
            sources["cc_use"] = f"megfigyelt kontroll/perc · n={cc_samples}; kísérleti 0–6 s/perc skála, nem CC-nehézség"
        manual = self.overrides.get(str(riot), {})
        for entry in (manual.get("all", {}), manual.get(str(role), {})):
            for key, value in entry.items():
                if key in LABELS and value is None:
                    values.pop(key, None)
                    sources.pop(key, None)
                    continue
                if key in LABELS and finite(value) and 1 <= value <= 5:
                    values[key], sources[key] = float(value), "kézzel megadott"
        # Observations stay descriptive: no CC 'difficulty' or HP-damage inference.
        return {"values": values, "sources": sources, "metrics": measured,
                "samples": max((value[1] for value in measured.values()), default=0),
                "attack_range": base.get("stats", {}).get("attackrange"), "name": base.get("name", str(riot))}

    def accepts(self, riot, role, tier, tags, minimum=3, match_all=True):
        values = self.profile(riot, role, tier)["values"]
        results = [values.get(tag, 0) >= minimum for tag in tags if tag in LABELS]
        return (all(results) if match_all else any(results)) if results else True


def team_summary(profiles):
    result = {}
    for key in LABELS:
        values = sorted((p["values"][key] for p in profiles if key in p["values"]), reverse=True)
        if not values:
            result[key] = {"score": None, "known": 0}
            continue
        if key in ("frontline", "engage", "peel", "waveclear", "poke"):
            score = values[0] if len(values) == 1 else .65 * values[0] + .35 * values[1]
        elif key == "cc":
            score = min(5, values[0] + .15 * sum(max(0, value - 1) for value in values[1:]))
        else:
            score = sum(values) / len(values)
        result[key] = {"score": score, "known": len(values)}
    # Follow-up is useful in relation to the available initiation.
    if result["follow_up"]["score"] is not None and result["engage"]["score"] is not None:
        result["follow_up"]["score"] = min(result["follow_up"]["score"], result["engage"]["score"] + 1)
    result["range_mix"] = {"közel": 0, "közép": 0, "távol": 0, "ismeretlen": 0}
    for profile in profiles:
        value = profile["values"].get("range")
        result["range_mix"]["ismeretlen" if value is None else "közel" if value <= 2 else "közép" if value < 4 else "távol"] += 1
    damage, known = {key: 0 for key in ("physical", "magic", "true")}, 0
    for profile in profiles:
        metrics = profile.get("metrics", {})
        dpm, n = metrics.get("damage_per_min", (0, 0))
        if n < 20 or not all(metrics.get(key+"_share", (0, 0))[1] >= 20 for key in damage):
            continue
        known += 1
        for key in damage:
            total, count = metrics[key+"_share"]
            damage[key] += dpm/n * total/count
    total = sum(damage.values())
    result["damage_mix"] = {key: value/total for key, value in damage.items()} if total else {}
    result["damage_known"] = known
    return result


def fit_notes(profile, allies):
    summary = team_summary(allies)
    additions = [LABELS[key] for key in ("frontline", "engage", "peel", "waveclear")
                 if profile["values"].get(key, 0) >= 3 and
                 (summary[key]["score"] is None or summary[key]["score"] < 3)]
    if profile["values"].get("follow_up", 0) >= 3 and (summary["engage"]["score"] or 0) >= 3:
        additions.append("Follow-up")
    return additions[:2]
