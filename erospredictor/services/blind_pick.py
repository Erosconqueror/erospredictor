"""Pick-rate-weighted supported counter exposure; no extra model inference."""

import math

COUNTER_DROP_PP = 3.0
MIN_COVERAGE = .5
MIN_VISIBLE_RISK = 10.0


def _aggregate_risks(parts):
    if not parts:
        return {"status": "resolved", "risk_percent": 0.0, "upper_percent": 0.0,
                "coverage_percent": None}
    chance, upper = 1.0, 1.0
    for part in parts:
        chance *= 1 - part["risk_percent"] / 100
        upper *= 1 - part["upper_percent"] / 100
    return {"status": "estimated" if all(part["coverage_percent"] >= 100 * MIN_COVERAGE
                                         for part in parts) else "insufficient",
            "risk_percent": 100 * (1 - chance), "upper_percent": 100 * (1 - upper),
            "coverage_percent": sum(part["coverage_percent"] for part in parts) / len(parts)}


def blind_risk(index, div, champion, role, enemy, unavailable=()):
    result = {"status": "unavailable", "risk_percent": None, "upper_percent": None,
              "coverage_percent": None, "known_counters": [], "role_risks": [],
              "threshold_pp": COUNTER_DROP_PP, "baseline": None}
    if index is None:
        return result
    if not index.eligible(div, champion, role):
        return dict(result, status="off_meta")
    baseline = index.baseline(div, champion, role)
    if baseline is None:
        return result
    result["baseline"] = baseline
    prior = index.prior(div) or {}
    weights = prior.get("counter_weights", [[1.0] * 5 for _ in range(5)])
    excluded = {cid for cid in unavailable if cid >= 0} | {cid for cid in enemy if cid >= 0} | {champion}

    def evidence(foe, foe_role):
        pair = index.counter(div, champion, role, foe, foe_role)
        if pair is None or pair.get("matches", 0) < 20:
            return None
        delta = pair.get("noise_corrected_delta_pp", pair["adjusted_delta_pp"])
        weighted = delta * weights[role][foe_role]
        return {"id": foe, "role": foe_role, "delta_pp": pair["adjusted_delta_pp"],
                "weighted_delta_pp": weighted, "matches": pair["matches"],
                "counter": weighted <= -COUNTER_DROP_PP}

    for foe_role, selected in enumerate(enemy):
        if selected >= 0:
            pair = evidence(selected, foe_role)
            if pair is not None and pair["counter"]:
                result["known_counters"].append(pair)
            continue
        rates = {cid: probability for cid, probability in index.role_pick_rates(div, foe_role).items()
                 if cid not in excluded and probability > 0}
        total = sum(rates.values())
        covered, counter_mass, counters = 0.0, 0.0, []
        for foe, probability in rates.items():
            pair = evidence(foe, foe_role)
            if pair is None:
                continue
            covered += probability
            if pair["counter"]:
                counter_mass += probability
                counters.append(dict(pair, pick_rate_percent=100 * probability / total))
        coverage = min(1.0, covered / total) if total else 0.0
        probability = min(1.0, counter_mass / total) if total else 0.0
        unknown = 1 - coverage
        result["role_risks"].append({"role": foe_role, "risk_percent": 100 * probability,
            "upper_percent": 100 * min(1.0, probability + unknown),
            "coverage_percent": 100 * coverage, "counters": sorted(counters, key=lambda item: -item["pick_rate_percent"])})
    result.update(_aggregate_risks(result["role_risks"]))
    result["lane"] = _aggregate_risks([part for part in result["role_risks"] if part["role"] == role])
    result["others"] = _aggregate_risks([part for part in result["role_risks"] if part["role"] != role])
    result["scope"] = "all"
    if result["others"]["status"] == "insufficient" and result["lane"]["status"] == "estimated":
        # Stay in the selected tier; do not invent cross-role estimates or pool ranks.
        result.update(result["lane"])
        result["scope"] = "lane"
    return result


def annotate_blind_risk(recommendations, index, div, blue, red, bans, team, role):
    enemy = red if team == "blue" else blue
    unavailable = list(blue) + list(red) + list(bans)
    result = []
    for item in recommendations:
        try:
            evidence = blind_risk(index, div, item["id"], role, enemy, unavailable)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError):
            evidence = blind_risk(None, div, item["id"], role, enemy, unavailable)
        result.append(dict(item, blind=evidence))
    return result


def build_blind_profiles(index, tiers, mapping):
    """Pre-training baseline profiles with Riot IDs, counter mass and noise-corrected spread."""
    inverse = {int(cid): str(riot) for riot, cid in mapping.items()}
    buckets = {}
    for tier in dict.fromkeys(list(tiers) + ["MIXED"]):
        by_role = {}
        for role in range(5):
            by_champion = {}
            for champion in index.role_pick_rates(tier, role):
                profile = blind_risk(index, tier, champion, role, [-1] * 5)
                summary = {key: profile.get(key) for key in ("status", "risk_percent", "upper_percent", "coverage_percent", "baseline", "lane", "others", "scope")}
                summary["counter_count"] = sum(len(item["counters"]) for item in profile["role_risks"])
                prior = index.prior(tier) or {}
                for kind in ("counter", "synergy"):
                    observations = []
                    weights = prior.get(f"{kind}_weights", [[1.0] * 5 for _ in range(5)])
                    for other_role in range(5):
                        if kind == "synergy" and role == other_role:
                            continue
                        for other, rate in index.role_pick_rates(tier, other_role).items():
                            if other == champion:
                                continue
                            pair = getattr(index, kind)(tier, champion, role, other, other_role)
                            if pair:
                                weight = weights[role][other_role]
                                observations.append((rate, pair["adjusted_delta_pp"] * weight,
                                                     pair.get("uncertainty_pp", 0.0) * weight))
                    total = sum(rate for rate, _, _ in observations)
                    spread = None
                    if total:
                        mean = sum(rate * delta for rate, delta, _ in observations) / total
                        variance = sum(rate * ((delta - mean) ** 2 - noise ** 2)
                                       for rate, delta, noise in observations) / total
                        spread = math.sqrt(max(0.0, variance))
                    summary[f"{kind}_spread_pp"] = spread
                    summary[f"{kind}_supported_pairs"] = len(observations)
                by_champion[inverse[champion]] = summary
            by_role[str(role)] = by_champion
        buckets[tier] = by_role
    return {"version": 1, "scope": dict(getattr(index, "document", {}).get("scope", {})),
            "threshold_pp": COUNTER_DROP_PP, "minimum_coverage": MIN_COVERAGE,
            "method": "baseline-adjusted supported counter pick-rate exposure; noise-corrected weighted spread",
            "tiers": buckets}


def risk_badge(evidence):
    """Compact future-pick warning; known opponents never override this badge."""
    status, chance = evidence.get("status"), evidence.get("risk_percent")
    if status == "resolved":
        return "", None
    if status != "estimated" or chance is None or not math.isfinite(chance):
        return "! Blind: kevés adat" if status == "insufficient" else "! Blind: nincs adat", "#9E9EAE"
    upper = evidence.get("upper_percent", chance)
    upper = chance if upper is None else max(chance, upper)
    if not math.isfinite(upper):
        return "! Blind: kevés adat", "#9E9EAE"
    if upper < MIN_VISIBLE_RISK:
        return "", None
    # Foreground-only continuous yellow -> orange -> red scale.
    anchors = ((10, (212, 197, 92)), (25, (255, 183, 77)), (45, (255, 138, 80)),
               (65, (239, 83, 80)), (85, (215, 55, 79)), (100, (181, 31, 61)))
    value = min(100.0, max(MIN_VISIBLE_RISK, upper))
    for (low, start), (high, end) in zip(anchors, anchors[1:]):
        if value <= high:
            fraction = (value - low) / (high - low)
            rgb = tuple(round(a + fraction * (b - a)) for a, b in zip(start, end))
            color = "#" + "".join(f"{channel:02X}" for channel in rgb)
            break
    caption = f"! Blind ~{chance:.0f}–{upper:.0f}%" if upper - chance >= 2 else f"! Blind ~{chance:.0f}%"
    return caption, color


def risk_badges(evidence):
    """Independent lane/other-role captions for the compact line below win rate."""
    if "lane" not in evidence or "others" not in evidence:
        caption, color = risk_badge(evidence)
        return [(caption, color)] if caption else []
    badges = []
    for key, label in (("lane", "Lane"), ("others", "Más role")):
        part = evidence[key]
        if key == "others" and evidence.get("scope") == "lane":
            badges.append(("csak lane (kevés adat)", "#9E9EAE"))
            continue
        caption, color = risk_badge(part)
        if caption:
            caption = caption.replace("! Blind", label, 1)
            badges.append((caption, color))
    return badges
