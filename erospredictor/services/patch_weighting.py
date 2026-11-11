"""Training recency weights, newest first: 1.0, 1.0, 0.9, 0.8, ..."""

PATCH_WEIGHT_POLICY = "two_full_then_tenth_v1"


def patch_weights(patches):
    ordered = sorted(set(patches), key=lambda patch: tuple(map(int, patch.split("."))), reverse=True)
    return {patch: max(1, 10 - max(0, age - 1)) / 10
            for age, patch in enumerate(ordered)}
