"""Explain pool-specific bans from cached lane outcomes, without neural calls."""

import math

from services.role_distribution import role_probabilities

MIN_MATCHUP_SAMPLE = 20
MIN_PERSONAL_SAMPLE = 10


def _personal_matchups(history, pool, role, inverse):
    """Count unique known lane opponents on this role, within the selected pool."""
    if history is None:
        return {}
    own_riot = {int(inverse[cid]) for cid in pool}
    by_riot = {int(riot): cid for cid, riot in inverse.items()}
    results, seen = {}, set()
    for row in history.games:
        match_id = row.get("match_id")
        if (not isinstance(match_id, str) or not match_id or match_id in seen
                or type(row.get("role")) is not int or row["role"] != role
                or row.get("champion") not in own_riot or not isinstance(row.get("win"), bool)):
            continue
        opponent = by_riot.get(row.get("enemy"))
        if opponent is None:
            continue
        seen.add(match_id)
        item = results.setdefault(opponent, {"games": 0, "wins": 0})
        item["games"] += 1
        item["wins"] += int(row["win"])
    return {cid: dict(item, winrate=100 * item["wins"] / item["games"],
                      enough=item["games"] >= MIN_PERSONAL_SAMPLE)
            for cid, item in results.items()}


def recommend_bans(statistical, distributions, mapping, pool, div, role, unavailable=(), top_k=5, history=None):
    """Blend pool matchup win rates 50/50 with >=10 own games per role/opponent.

    Population rates use sample-reliability weights within the pool. Their
    effective coverage and the opponent's observed frequency scale the ranking
    score. Personal data never changes prediction or recommendation models.
    """
    if type(role) is not int or not 0 <= role < 5:
        raise ValueError("Válassz érvényes pozíciót a bansegédhez.")
    inverse = {int(index): riot for riot, index in mapping.items()}
    pool = sorted(set(pool) & set(inverse))
    if not pool:
        return []
    unavailable = set(unavailable) | set(pool)
    frequencies = {}
    for cid, riot in inverse.items():
        probabilities, total = role_probabilities(distributions, riot, tier=div)
        if total and probabilities[role] >= .05:
            frequencies[cid] = total * probabilities[role]
    total_frequency = sum(frequencies.values())
    if not total_frequency:
        return []
    personal = _personal_matchups(history, pool, role, inverse)
    results = []
    for opponent, occurrences in frequencies.items():
        if opponent in unavailable:
            continue
        matchups, weighted_rate, reliability = [], 0.0, 0.0
        for champion in pool:
            summary = statistical.get_lane_matchup(div, champion, opponent, role)
            if not summary or summary["matches"] < MIN_MATCHUP_SAMPLE:
                continue
            rate, count = summary["weighted_winrate"], summary["matches"]
            disadvantage = max(0.0, 50.0 - rate)
            weight = count / (count + MIN_MATCHUP_SAMPLE)
            weighted_rate += rate * weight
            reliability += weight
            matchups.append({"champion": champion, "winrate": rate, "matches": count, "disadvantage": disadvantage})
        if not matchups:
            continue
        general_rate = weighted_rate / reliability
        own = personal.get(opponent)
        personal_weight = .5 if own is not None and own["enough"] else 0.0
        combined_rate = (1 - personal_weight) * general_rate + personal_weight * (own["winrate"] if own else 0)
        disadvantage = max(0.0, 50.0 - combined_rate)
        if disadvantage <= 0:
            continue
        frequency = occurrences / total_frequency
        # A ranking score, never presented as a win probability or ban win rate.
        score = disadvantage * reliability / len(pool) * math.sqrt(frequency)
        results.append({"id": opponent, "score": score, "sample_frequency": frequency * 100,
            "role_occurrences": round(occurrences), "covered": len(matchups), "pool_size": len(pool),
            "global_winrate": general_rate, "global_matches": sum(item["matches"] for item in matchups),
            "personal": own, "personal_weight": personal_weight, "combined_winrate": combined_rate,
            "matchups": sorted(matchups, key=lambda item: item["winrate"])})
    results.sort(key=lambda item: (-item["score"], item["id"]))
    return results[:max(0, top_k)]
