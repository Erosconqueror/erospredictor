"""Experimental descriptive profiles. Never imported by inference models.

Observed values are unweighted sums and valid sample counts, by patch/role/tier.
Gameplay presets are editable heuristics, not measured matchup evidence.
"""

import json
import math
from pathlib import Path
from services.update_progress import atomic_json
from services.trait_metrics import DAMAGE_KEYS, PHASE_PREFIX, damage_rate, phase_profiles, tag_priority

DATA = Path(__file__).resolve().parents[1] / "data"
LABELS = {"frontline": "Frontline", "engage": "Engage", "follow_up": "Follow-up",
          "peel": "Peel / disengage", "cc": "CC-erő", "cc_access": "CC alkalmazhatóság",
          "hard_cc": "Hard CC", "soft_cc": "Soft CC",
          "cc_use": "Mért CC-kihasználás",
          "burst": "Burst", "sustained": "Folyamatos sebzés", "percent_health": "% HP sebzés",
          "waveclear": "Waveclear", "poke": "Poke", "mobility": "Harci mobilitás",
          "early": "Early WR", "mid": "Mid WR", "late": "Late WR",
          "range": "Harci távolság", "reach": "Elérési távolság", "spacing": "Távolságtartás",
          "team_sustain": "Csapatfenntartás", "map_presence": "Map presence",
          "physical_damage": "Fizikai sebzés", "magic_damage": "Mágikus sebzés"}
# Reviewable starting points only. Empty fields remain unknown.
PRESETS = {
    "Tank": {"frontline": 4, "engage": 3, "peel": 3, "cc": 4, "range": 1},
    "Fighter": {"frontline": 3, "sustained": 4, "range": 1},
    "Assassin": {"burst": 4, "range": 2, "frontline": 1},
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
        fields.update({kind+"_damage_per_min": kind+"DamageDealtToChampions" for kind in ("physical", "magic", "true")})
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
            new_phase = "early" if minutes < 23 else "mid" if minutes < 34 else "late"
            metrics[PHASE_PREFIX+new_phase] = float(player["win"])
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
    def __init__(self, basics=None, observations=None, overrides=None, capabilities=None):
        self.basics = basics if basics is not None else read_document(DATA / "champion_trait_basics.json")
        self.observations = observations if observations is not None else read_document(DATA / "champion_trait_observations.json")
        if overrides is None:
            from services.client_settings import settings_path
            overrides = read_document(settings_path().with_name("champion_trait_profiles.json"))
        self.overrides = overrides
        self.capabilities = capabilities if capabilities is not None else read_document(DATA / "champion_mobility_profiles.json")
        self.enemy_riots = ()
        self._reference_teams = {}
        self.patches = sorted(set(self.observations.get("patches", [])), key=lambda p: tuple(map(int, p.split("."))))[-5:]
        self._measured = {}
        for row in self.observations.get("rows", []):
            if row.get("patch") not in self.patches:
                continue
            for tier in set((row.get("tier"), "MIXED")):
                key = (str(row.get("champion")), row.get("role"), tier)
                self._measured[key] = merge_metrics(self._measured.get(key, {}), row.get("metrics", {}))
        self._damage_baselines = {}
        for (_, _, tier), metrics in self._measured.items():
            for kind in ("physical", "magic", "true"):
                value = damage_rate(metrics, kind)
                if value:
                    rate, count, _ = value
                    total, n = self._damage_baselines.get((tier, kind), (0, 0))
                    self._damage_baselines[tier, kind] = (total+rate*count, n+count)

    def measured(self, riot, role, tier):
        return self._measured.get((str(riot), role, tier), {})

    def for_opponents(self, riots):
        """Contextual descriptive copy; no inference model or shared mutation."""
        from copy import copy
        result = copy(self)
        result.enemy_riots = tuple(sorted(set(map(str, riots))))
        return result

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
        name = base.get("name")
        review = self.capabilities.get("champions", {}).get(str(riot), {})
        if name and review.get("name") == name:
            for key in ("mobility", "map_presence"):
                score = review.get(key)
                if finite(score) and 1 <= score <= 5:
                    values[key] = score
                    sources[key] = (f"kísérleti képességprofil · {self.capabilities.get('patch', '?')} · "
                        f"{review.get(key+'_reason', '')} · {review.get('source', '')}")
            if name == "Sylas" and "map_presence" in values:
                ultimates = self.capabilities.get("champions", {})
                candidates = [(ultimates.get(enemy, {}).get("ultimate_map_presence", 1),
                               ultimates.get(enemy, {}).get("name", enemy)) for enemy in self.enemy_riots]
                score, enemy = max([(2, "alapkit")] + candidates)
                values["map_presence"] = score
                sources["map_presence"] += f" · aktuális ellopható R: {enemy}, {score}/5; lehetőség, nem ellopott/használatra kész ulti"
        measured = self.measured(riot, role, tier)
        damage_details = {}
        for key, kind in DAMAGE_KEYS.items():
            rate = damage_rate(measured, kind)
            total, n = self._damage_baselines.get((tier, kind), (0, 0))
            if rate and n >= 100 and total > 0:
                dpm, samples, estimated = rate
                baseline = total/n
                values[key] = max(1, min(5, 1+2*dpm/baseline))
                damage_details[kind] = dict(dpm=dpm, baseline=baseline, samples=samples, estimated=estimated)
                sources[key] = (f"{dpm:.1f}/perc; rangon belüli összes szerepkör átlaga {baseline:.1f}/perc; "
                    f"n={samples}; " + ("régi arány × DPM becslés" if estimated else "mért típusonkénti sebzés/perc"))
        phases = phase_profiles(measured)
        for key, detail in phases.items():
            values[key] = detail["score"]
            sources[key] = (f"Meccsvégi {key} WR {detail['wr']:.1%}, saját alap-WR {detail['baseline']:.1%}, "
                f"eltérés {detail['delta']*100:+.1f}%; n={detail['samples']}/{detail['baseline_samples']}; "
                "23/34 perces időhatárok; nem játékon belüli erősségmérés")
        cc_total, cc_samples = measured.get("cc_seconds_per_min", (0, 0))
        if cc_samples >= 30:
            values["cc_use"] = 1 + 4 * min(1, cc_total / cc_samples / 6)
            sources["cc_use"] = f"megfigyelt kontroll/perc · n={cc_samples}; kísérleti 0–6 s/perc skála, nem CC-nehézség"
        heal, heal_n = measured.get("ally_heal_per_min", (0, 0))
        shield, shield_n = measured.get("ally_shield_per_min", (0, 0))
        if heal_n >= 30 and shield_n >= 30:
            # Sum the two observed mean rates, not self-healing or totalHeal.
            rate = heal / heal_n + shield / shield_n
            values["team_sustain"] = 1 + 4 * min(1, rate / 600)
            sources["team_sustain"] = (f"társak gyógyítása {heal/heal_n:.1f}/perc (n={heal_n}) + "
                f"pajzsolása {shield/shield_n:.1f}/perc (n={shield_n}); "
                "kísérleti 0–600/perc skála, nem winrate vagy önfenntartás")
        manual = self.overrides.get(str(riot), {})
        for entry in (manual.get("all", {}), manual.get(str(role), {})):
            for key, value in entry.items():
                if key in LABELS and value is None:
                    phases.pop(key, None)
                    values.pop(key, None)
                    sources.pop(key, None)
                    continue
                if key in LABELS and finite(value) and 1 <= value <= 5:
                    phases.pop(key, None)
                    values[key], sources[key] = float(value), "kézzel megadott"
        # Observations stay descriptive: no CC 'difficulty' or HP-damage inference.
        return {"values": values, "sources": sources, "metrics": measured,
                "damage_details": damage_details, "phase_details": phases,
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
        if key in ("frontline", "engage", "peel", "waveclear", "poke", "team_sustain", "map_presence"):
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
    result["damage_rates"] = {}
    for kind in ("physical", "magic"):
        details = [p.get("damage_details", {}).get(kind) for p in profiles]
        details = [detail for detail in details if detail]
        if details:
            result["damage_rates"][kind] = dict(dpm=sum(d["dpm"] for d in details),
                baseline=sum(d["baseline"] for d in details), known=len(details),
                estimated=any(d["estimated"] for d in details))
            key = next(key for key, value in DAMAGE_KEYS.items() if value == kind)
            info = result["damage_rates"][kind]
            result[key]["detail"] = (f"{info['dpm']:.1f}/perc; referencia {info['baseline']:.1f}/perc; "
                f"{len(details)}/{len(profiles)} ismert; " + ("részben régi arány × DPM becslés" if info['estimated'] else "mért DPM"))
    for phase in ("early", "mid", "late"):
        details = [p.get("phase_details", {}).get(phase) for p in profiles]
        details = [detail for detail in details if detail]
        if details:
            delta = sum(detail["delta"] for detail in details)/len(details)
            result[phase]["score"] = max(1, min(5, 3+delta/.05))
            result[phase]["known"] = len(details)
            result[phase]["phase_delta"] = delta
            result[phase]["phase_known"] = len(details)
            result[phase]["detail"] = (f"Hősönkénti saját alap-WR-hez viszonyított meccsvégi WR-eltérés átlaga: "
                f"{delta*100:+.1f}%; {len(details)}/{len(profiles)} ismert. 23/34 perces időhatárok; nem csapat-winrate.")
    return result


def fit_notes(profile, allies):
    summary = team_summary(allies)
    additions = [LABELS[key] for key in ("frontline", "engage", "peel", "waveclear", "team_sustain")
                 if profile["values"].get(key, 0) >= 3 and
                 (summary[key]["score"] is None or summary[key]["score"] < 3)]
    if profile["values"].get("follow_up", 0) >= 3 and (summary["engage"]["score"] or 0) >= 3:
        additions.append("Follow-up")
    return additions[:2]


def composition_outliers(library, picks, tier="MIXED", distributions=None, limit=3):
    """Compare the visible roles with reproducible, role-frequency-weighted teams.

    This is a descriptive profile reference, never a winrate model. Unknowns
    are excluded on both sides, with a minimum coverage and effect threshold.
    """
    import random
    from services.role_distribution import role_probabilities
    actual = [library.profile(riot, role, tier) for riot, role in picks]
    summary = team_summary(actual)
    if not picks:
        return {"tags": [], "picks": 0, "note": "Még nincs kiválasztott hős."}
    roles = tuple(sorted(role for _, role in picks))
    key = (tier, roles, id(distributions), library.enemy_riots)
    if not hasattr(library, "_reference_teams"):
        library._reference_teams = {}
    if key not in library._reference_teams:
        pools = {}
        riots = set(library.basics.get("champions", {})) | set(library.overrides)
        for role in roles:
            pool = []
            for riot in sorted(riots):
                probabilities, total = role_probabilities(distributions or {}, riot, tier)
                if total and probabilities[role] >= .05:
                    pool.append((riot, probabilities[role] * total))
            pools[role] = pool
        references = []
        rng = random.Random(1701)
        for _ in range(128):
            selected, profiles = set(), []
            for role in roles:
                choices = [(riot, weight) for riot, weight in pools[role] if riot not in selected]
                if not choices:
                    break
                riot = rng.choices([item[0] for item in choices], weights=[item[1] for item in choices])[0]
                selected.add(riot)
                profiles.append(library.profile(riot, role, tier))
            if len(profiles) == len(roles):
                references.append(team_summary(profiles))
        library._reference_teams[key] = references
    references = library._reference_teams[key]
    coverage = max(1, math.ceil(len(picks) * .6))
    tags = []
    eligible_metrics = 0
    descriptive = {"range": ("Hosszú harci táv", "Rövid harci táv"),
                   "reach": ("Nagy elérés", "Kis elérés"),
                   "spacing": ("Erős távolságtartás", "Kevés távolságtartás")}
    for metric, caption in LABELS.items():
        value = summary[metric]
        if "phase_delta" in value:
            if value["phase_known"] < coverage:
                continue
            shift = value["phase_delta"]
            if abs(shift) >= .03:
                tags.append(dict(key=metric, label=f"{caption} {shift*100:+.1f}%", direction="high" if shift > 0 else "low",
                    descriptive=False, score=value["score"], mean=3, delta=value["score"]-3,
                    deviation=abs(shift)/.05, known=value["phase_known"], tooltip=value["detail"]))
            continue
        reference = [item[metric]["score"] for item in references
                     if item[metric]["known"] >= coverage and item[metric]["score"] is not None]
        if metric in DAMAGE_KEYS:
            # A common absolute-DPM baseline across all roles; no 100% split
            # and no support-only normalization that would inflate tank damage.
            reference = [3]*32
        if value["known"] < coverage or len(reference) < 32:
            continue
        eligible_metrics += 1
        mean = sum(reference) / len(reference)
        spread = math.sqrt(sum((score - mean)**2 for score in reference) / len(reference))
        delta = value["score"] - mean
        z = delta / max(.35, spread)
        mild = metric in ("map_presence", "physical_damage", "magic_damage") or (delta > 0 and metric in ("frontline", "engage", "peel", "waveclear", "team_sustain"))
        if abs(delta) < (.35 if mild else .5) or abs(z) < (.75 if mild else 1):
            continue
        direction = "high" if delta > 0 else "low"
        label = (descriptive[metric][0 if delta > 0 else 1] if metric in descriptive else
                 ("Erős " if delta > 0 else "Kevés ") + caption)
        if "phase_delta" in value:
            label = f"{caption} {value['phase_delta']*100:+.1f}%"
        tags.append(dict(key=metric, label=label, direction=direction, descriptive=metric in descriptive,
                         score=value["score"], mean=mean, delta=delta, deviation=abs(z), known=value["known"],
                         tooltip=value.get("detail", "")))
    tags.sort(key=lambda item: item["deviation"], reverse=True)
    return {"tags": tags if limit is None else tags[:max(0, min(8, limit))], "picks": len(picks), "references": len(references),
            "note": "Szerepkörhöz illesztett profilreferencia; kísérleti becslés." if tags else
                    "Kevés profiladat a megbízható összehasonlításhoz." if not eligible_metrics else
                    "Nincs nagy eltérés a profilreferenciától."}


def _balanced_tags(candidates, team=False):
    """Hero cap 3/2 or 2/3; team cap 5/3, 3/5 or 4/4. No forced filling."""
    candidates = sorted(candidates, key=lambda item: (tag_priority(item), item.get("priority", 0)), reverse=True)
    groups = {tone: [item for item in candidates if (item["tone"] == "negative") == (tone == "negative")]
              for tone in ("positive", "negative")}
    dominant = max(groups, key=lambda tone: (len(groups[tone]),
                   sum(item.get("priority", 0) for item in groups[tone][:5])))
    caps = {tone: 3 if tone == dominant else 2 for tone in groups}
    if team:
        caps = {tone: 5 if tone == dominant else 3 for tone in groups}
        other = "negative" if dominant == "positive" else "positive"
        dominant_items, other_items = groups[dominant], groups[other]
        if (len(other_items) >= 4 and other_items[3].get("extreme") and
                (len(dominant_items) < 5 or other_items[3]["priority"] > dominant_items[4]["priority"])):
            caps = {"positive": 4, "negative": 4}
    selected = []
    counts = {"positive": 0, "negative": 0}
    for item in candidates:
        tone = "negative" if item["tone"] == "negative" else "positive"
        if len(selected) >= (4 if team else 3) and not item.get("extreme"):
            continue
        if counts[tone] < caps[tone]:
            selected.append(item)
            counts[tone] += 1
    return selected


def composition_summary(library, picks, tier="MIXED", distributions=None):
    """Prediction-side team summary with the same sparse 5/3–4/4–3/5 budget."""
    result = composition_outliers(library, picks, tier, distributions, limit=None)
    candidates = [dict(tag,
                       tone="descriptive" if tag["descriptive"] else
                            "positive" if tag["direction"] == "high" else "negative",
                       priority=tag["deviation"],
                       extreme=abs(tag["delta"]) >= 1 and tag["deviation"] >= 1.5)
                  for tag in result["tags"]]
    return dict(result, tags=_balanced_tags(candidates, team=True))


def recommendation_badges(library, riot, role, tier, distributions=None):
    """Salient known traits only, with a 3/2 or 2/3 cap."""
    profile = library.profile(riot, role, tier)
    outliers = composition_outliers(library, [(riot, role)], tier, distributions, limit=None)["tags"]
    candidates, used = [], set()
    descriptive = {"range", "reach", "spacing"}
    for tag in outliers:
        key = tag["key"]
        label = (f"{LABELS[key]} {profile['phase_details'][key]['delta']*100:+.1f}%"
                 if key in profile.get("phase_details", {}) else tag["label"] if tag["descriptive"] else LABELS[key])
        candidates.append({"key": key, "label": label,
                           "tone": "descriptive" if tag["descriptive"] else
                                   "positive" if tag["direction"] == "high" else "negative",
                           "priority": 10 + tag["deviation"],
                           "extreme": abs(tag["delta"]) >= 1 and tag["deviation"] >= 1.5,
                           "score": tag["score"],
                           "tooltip": f"{tag['score']:.2f}/5 · szerepköri referencia: {tag['mean']:.2f}/5 · " + profile['sources'].get(key, '')})
        used.add(key)
    extreme = sorted(profile["values"].items(), key=lambda item: abs(item[1]-3), reverse=True)
    for key, value in extreme:
        positive_min = 3.5 if key in ("map_presence", "frontline", "engage", "peel", "waveclear", "team_sustain", "physical_damage", "magic_damage") else 4
        negative_max = 2.5 if key in ("map_presence", "physical_damage", "magic_damage") else 2
        if key in used or key in descriptive or negative_max < value < positive_min:
            continue
        label = (f"{LABELS[key]} {profile['phase_details'][key]['delta']*100:+.1f}%"
                 if key in profile.get("phase_details", {}) else LABELS[key])
        candidates.append({"key": key, "label": label, "tone": "positive" if value >= positive_min else "negative",
                           "priority": abs(value-3),
                           "extreme": value >= 4.5 or value <= 1.5,
                           "score": value,
                           "tooltip": f"{value:.2f}/5 · {profile['sources'].get(key, 'kísérleti profil')}"})
    return _balanced_tags(candidates)


def recommendation_team_digest(library, picks, tier, distributions=None):
    result = composition_outliers(library, picks, tier, distributions, limit=None)
    if not picks:
        return {"character": [], "needs": [], "note": "Még nincs választott hős."}
    profiles = [library.profile(riot, role, tier) for riot, role in picks]
    summary = team_summary(profiles)
    candidates, used = [], set()
    for tag in result["tags"]:
        candidates.append({"key": tag["key"], "label": tag["label"] if tag["descriptive"] or tag["key"] in ("early", "mid", "late") else LABELS[tag["key"]],
                           "tone": "descriptive" if tag["descriptive"] else
                                   "negative" if tag["direction"] == "low" else "positive",
                           "priority": 10 + tag["deviation"],
                           "score": tag["score"],
                           "tooltip": tag.get("tooltip", ""),
                           "extreme": abs(tag["delta"]) >= 1 and tag["deviation"] >= 1.5})
        used.add(tag["key"])
    coverage = max(1, math.ceil(len(picks)*.6))
    for key in ("frontline", "engage", "peel", "waveclear", "sustained", "team_sustain", "map_presence", "physical_damage", "magic_damage", "early", "mid", "late"):
        score = summary[key]["score"]
        if summary[key]["known"] < coverage or score is None:
            continue
        positive_min = 3.5 if key in ("map_presence", "frontline", "engage", "peel", "waveclear", "team_sustain", "physical_damage", "magic_damage") else 4
        negative_max = 2.5 if key in ("map_presence", "physical_damage", "magic_damage") else 2
        if key not in used and (score >= positive_min or score <= negative_max):
            label = f"{LABELS[key]} {summary[key]['phase_delta']*100:+.1f}%" if "phase_delta" in summary[key] else LABELS[key]
            candidates.append({"key": key, "label": label, "tone": "positive" if score >= positive_min else "negative",
                               "tooltip": summary[key].get("detail", ""),
                               "score": score, "priority": abs(score-3), "extreme": score >= 4.5 or score <= 1.5})
    selected = _balanced_tags(candidates, team=True)
    return {"character": [tag["label"] for tag in selected if tag["tone"] != "negative"],
            "needs": [tag["label"] for tag in selected if tag["tone"] == "negative"],
            "character_tags": [tag for tag in selected if tag["tone"] != "negative"],
            "need_tags": [tag for tag in selected if tag["tone"] == "negative"], "note": result["note"]}
