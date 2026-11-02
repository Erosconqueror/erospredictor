"""Rank targets and a deterministic patch-stratified training snapshot."""

from collections import defaultdict
import random


def rank_targets(args, configured):
    targets = {tier: (args.matches_per_tier if args.matches_per_tier is not None else configured[tier])
               for tier in args.tiers}
    for value in getattr(args, "tier_target", []) or []:
        try:
            tier, count = value.split("=", 1)
            tier, count = tier.upper(), int(count)
        except (ValueError, TypeError) as exc:
            raise ValueError("--tier-target must look like GOLD=22000") from exc
        if tier not in targets or count < 1:
            raise ValueError("--tier-target must name a selected tier and a positive count")
        targets[tier] = count
    return targets


def collection_steps(targets):
    """Switch tiers after one twentieth of each rank's base target (rounded up)."""
    return {tier: max(1, (goal + 19) // 20) for tier, goal in targets.items()}


def collection_order(tiers, priority=()):
    preferred = list(dict.fromkeys(priority))
    if any(tier not in tiers for tier in preferred):
        raise ValueError("--priority-tier must name a selected tier")
    return preferred + [tier for tier in tiers if tier not in preferred]


def balanced_snapshot(records, targets, patches, seed=42):
    """Cap large historical tiers; include every available patch where possible."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["tier"], record["patch"])].append(record)
    result = []
    generator = random.Random(seed)
    for tier, target in targets.items():
        available = [patch for patch in reversed(patches) if groups[(tier, patch)]]
        if not available:
            continue
        buckets = []
        for patch in available:
            values = sorted(groups[(tier, patch)], key=lambda item: item["match_id"])
            generator.shuffle(values)
            buckets.append(values)
        selected, used = [], []
        per_patch, extra = divmod(target, len(buckets))
        for index, bucket in enumerate(buckets):
            take = min(len(bucket), per_patch + (index < extra))
            selected.extend(bucket[:take])
            used.append(take)
        remaining = target - len(selected)
        for bucket, offset in zip(buckets, used):
            take = min(len(bucket) - offset, remaining)
            selected.extend(bucket[offset:offset + take])
            remaining -= take
        result.extend(selected)
    return result
