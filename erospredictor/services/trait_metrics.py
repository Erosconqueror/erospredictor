"""Descriptive damage rates and game-end winrates; never prediction inputs."""
import math

PHASE_PREFIX = "endwin_23_34_"
PHASES = ("early", "mid", "late")
DAMAGE_KEYS = {"physical_damage": "physical", "magic_damage": "magic"}
POSITIVE_DAMAGE_KEYS = {"overall_damage", "balanced_damage"}


def balanced_damage_score(details):
    if not all(kind in details for kind in ("physical", "magic", "overall")):
        return None
    physical, magic, total = (details[kind] for kind in ("physical", "magic", "overall"))
    if min(physical["baseline"], magic["baseline"], total["baseline"], total["dpm"]) <= 0:
        return None
    ratio = min(physical["dpm"]/physical["baseline"], magic["dpm"]/magic["baseline"])
    share = min(physical["dpm"], magic["dpm"])/total["dpm"]
    if ratio < .75 or share < .25 or total["dpm"] < total["baseline"]:
        return None
    return min(5, 3.5+2*(ratio-.75))


def damage_rate(metrics, kind):
    if kind == "overall":
        total, n = metrics.get("damage_per_min", (0, 0))
        return (total/n, n, False) if n >= 30 else None
    total, n = metrics.get(kind+"_damage_per_min", (0, 0))
    if n >= 30:
        return total/n, n, False
    # Compatibility with existing exports. Mean products are an estimate,
    # not the exact mean of per-match type damage; new collection stores that.
    damage, dn = metrics.get("damage_per_min", (0, 0))
    share, sn = metrics.get(kind+"_share", (0, 0))
    if dn >= 30 and sn >= 30:
        return damage/dn * share/sn, min(dn, sn), True
    return None


def phase_profiles(metrics):
    buckets = {key: metrics.get(PHASE_PREFIX+key, (0, 0)) for key in PHASES}
    wins = sum(value[0] for value in buckets.values())
    n = sum(value[1] for value in buckets.values())
    if n < 300:
        return {}
    base = wins/n
    result = {}
    for key, (w, count) in buckets.items():
        other = n-count
        if count < 100 or other < 100:
            continue
        rate = w/count
        delta = rate-base
        # Conservative noise guard, not a formal independent-sample CI:
        # the bucket also contributes to the baseline.
        noise = 1.96*math.sqrt(rate*(1-rate)/count + base*(1-base)/n)
        if abs(delta) < max(.03, noise):
            continue
        result[key] = dict(score=max(1, min(5, 3+delta/.05)), wr=rate, baseline=base,
                           delta=delta, samples=count, baseline_samples=n)
    return result


def tag_priority(item):
    key = item.get("key")
    if key in DAMAGE_KEYS:
        return 100 if item.get("tone") == "negative" else 45
    if key in POSITIVE_DAMAGE_KEYS:
        return 50 if key == "balanced_damage" else 48
    if key in ("frontline", "engage", "peel", "waveclear"):
        return 25
    if key == "map_presence":
        return -5
    return 0
