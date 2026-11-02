"""Normalize available model weights with a hard minimum contribution."""

import math

MIN_MODEL_WEIGHT = 0.08
MODEL_NAMES = ("gnn", "roleweighted", "roleaware", "statistical")


def ensemble_minimums(enhanced_statistical=False):
    return {"gnn": .16 if enhanced_statistical else .24,
            "statistical": .30,
            "roleweighted": .08, "roleaware": .08}


def enhanced_statistics(statistical, div, blue=None, red=None):
    name = "has_interactions" if blue is None else "uses_interactions"
    query = getattr(statistical, name, None)
    if not callable(query):
        return False
    try:
        return (query(div) if blue is None else query(div, blue, red)) is True
    except (ValueError, TypeError, AttributeError):
        return False


def bounded_weights(weights, available=None, minimum=MIN_MODEL_WEIGHT, minimums=None):
    names = list(dict.fromkeys(available if available is not None else weights))
    if not names:
        return {}
    if not 0 <= minimum <= 1 / len(names):
        raise ValueError("Infeasible ensemble weight floor")
    floors = {name: (minimums or {}).get(name, minimum) for name in names}
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in floors.values()) or sum(floors.values()) > 1 + 1e-12:
        raise ValueError("Infeasible per-model ensemble weight floors")
    raw = {}
    for name in names:
        value = weights.get(name, 0)
        raw[name] = float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else 0.0
    free, fixed = list(names), {}
    while free:
        remaining = 1.0 - sum(fixed.values())
        total = sum(raw[name] for name in free)
        proposal = {name: remaining * raw[name] / total if total else remaining / len(free) for name in free}
        below = [name for name in free if proposal[name] < floors[name]]
        if not below:
            fixed.update(proposal)
            break
        fixed.update({name: floors[name] for name in below})
        free = [name for name in free if name not in fixed]
    return {name: fixed[name] for name in names}
