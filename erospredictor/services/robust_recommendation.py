"""Optional reranking of five candidates over at most three enemy role layouts."""

from math import isfinite


def rerank(recommendations, predict, div, blue, red, team, role, scenarios):
    if team not in ("blue", "red") or not 0 <= role < 5:
        raise ValueError("Érvénytelen ajánlási pozíció.")
    own = blue if team == "blue" else red
    enemy = red if team == "blue" else blue
    if own[role] >= 0:
        raise ValueError("Az ajánlás helye már foglalt.")
    layouts, seen = [], set()
    for scenario in scenarios[:3]:
        b = [cid if cid is not None else -1 for cid in scenario.blue]
        r = [cid if cid is not None else -1 for cid in scenario.red]
        candidate_own, candidate_enemy = (b, r) if team == "blue" else (r, b)
        signature = (tuple(b), tuple(r))
        if (len(b) != 5 or len(r) != 5 or candidate_own != list(own)
                or sorted(cid for cid in candidate_enemy if cid >= 0) != sorted(cid for cid in enemy if cid >= 0)
                or signature in seen or not isfinite(scenario.weight) or scenario.weight <= 0):
            continue
        seen.add(signature)
        layouts.append((b, r, scenario.weight, scenario.forced))
    if len(layouts) < 2:
        return [dict(item, robust={"count": len(layouts), "note": "Nincs két különböző, használható ellenfél-kiosztás."})
                for item in recommendations]
    result = []
    for position, item in enumerate(recommendations[:5]):
        scores = []
        for b, r, weight, forced in layouts:
            primary = b == list(blue) and r == list(red)
            b, r = list(b), list(r)
            (b if team == "blue" else r)[role] = item["id"]
            if primary:
                score = item["wr"]
            else:
                score = predict(div, b, r)[f"{team}_win_prob"]
            if not isfinite(score) or not 0 <= score <= 100:
                raise ValueError("Érvénytelen predikció az alternatív kiosztásban.")
            scores.append({"blue": tuple(b), "red": tuple(r), "weight": weight, "wr": score, "forced": forced})
        worst, best = min(score["wr"] for score in scores), max(score["wr"] for score in scores)
        mean = sum(score["wr"] * score["weight"] for score in scores) / sum(score["weight"] for score in scores)
        result.append(dict(item, robust={"count": len(scores), "min": worst, "max": best, "mean": mean,
            "primary_rank": position + 1, "scenarios": scores, "forced": any(score["forced"] for score in scores)}))
    result.sort(key=lambda item: (item["robust"]["min"], item["robust"]["mean"]), reverse=True)
    return result + [dict(item) for item in recommendations[5:]]
