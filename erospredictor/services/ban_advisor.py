"""Explain pool-specific bans from cached lane outcomes, without neural calls."""

import math

from services.role_distribution import role_probabilities

MIN_MATCHUP_SAMPLE = 20


def recommend_bans(statistical, distributions, mapping, pool, div, role, unavailable=(), top_k=5):
    if not isinstance(role, int) or not 0 <= role < 5:
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
    results = []
    for opponent, occurrences in frequencies.items():
        if opponent in unavailable:
            continue
        matchups, burden = [], 0.0
        for champion in pool:
            summary = statistical.get_lane_matchup(div, champion, opponent, role)
            if not summary or summary["matches"] < MIN_MATCHUP_SAMPLE:
                continue
            rate, count = summary["weighted_winrate"], summary["matches"]
            disadvantage = max(0.0, 50.0 - rate)
            burden += disadvantage * count / (count + MIN_MATCHUP_SAMPLE)
            matchups.append({"champion": champion, "winrate": rate, "matches": count, "disadvantage": disadvantage})
        if not matchups or burden <= 0:
            continue
        frequency = occurrences / total_frequency
        # A ranking score, never presented as a win probability or ban win rate.
        score = burden / len(pool) * math.sqrt(frequency)
        results.append({"id": opponent, "score": score, "sample_frequency": frequency * 100,
            "role_occurrences": round(occurrences), "covered": len(matchups), "pool_size": len(pool),
            "matchups": sorted(matchups, key=lambda item: item["winrate"])})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max(0, top_k)]
