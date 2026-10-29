"""Normalize available model weights with a hard minimum contribution."""

import math

MIN_MODEL_WEIGHT = 0.08
MODEL_NAMES = ("gnn", "roleweighted", "roleaware", "statistical")


def bounded_weights(weights, available=None, minimum=MIN_MODEL_WEIGHT):
    names = list(dict.fromkeys(available if available is not None else weights))
    if not names:
        return {}
    if not 0 <= minimum <= 1 / len(names):
        raise ValueError("Infeasible ensemble weight floor")
    raw = {}
    for name in names:
        value = weights.get(name, 0)
        raw[name] = float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else 0.0
    free, fixed = list(names), {}
    while free:
        remaining = 1.0 - sum(fixed.values())
        total = sum(raw[name] for name in free)
        proposal = {name: remaining * raw[name] / total if total else remaining / len(free) for name in free}
        below = [name for name in free if proposal[name] < minimum]
        if not below:
            fixed.update(proposal)
            break
        fixed.update({name: minimum for name in below})
        free = [name for name in free if name not in fixed]
    return {name: fixed[name] for name in names}
