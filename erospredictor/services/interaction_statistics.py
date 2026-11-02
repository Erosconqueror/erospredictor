"""Side- and champion-strength-adjusted role interactions from a frozen window.

All persisted champion IDs are Riot IDs. Counts include both teams; counter
records describe the first champion's team outcome, independently of its side.
The actual blue exposure is retained when computing the expected matchup rate.
No inference model or network access is needed to build or read this artifact.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from services.role_distribution import ROLE_NAMES, _team


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data/interaction_stats.json"
MIN_SAMPLES = 13
SHRINKAGE = 20.0


def _sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def _logit(value):
    value = np.clip(value, .001, .999)
    return np.log(value / (1 - value))


def _neutral_baselines(counts, wins, blue_counts, blue_rate, shrinkage):
    if blue_rate <= .001 or blue_rate >= .999:
        return np.full_like(wins, .5)  # No side-independent strength is identifiable.
    fraction = np.divide(blue_counts, counts, out=np.full_like(wins, .5), where=counts > 0)
    neutral_expected = fraction * blue_rate + (1 - fraction) * (1 - blue_rate)
    target = (wins + shrinkage * neutral_expected) / (counts + shrinkage)
    offset = _logit(blue_rate)
    low, high = np.full_like(wins, -10), np.full_like(wins, 10)
    for _ in range(30):
        middle = (low + high) / 2
        expected = fraction * _sigmoid(middle + offset) + (1 - fraction) * _sigmoid(middle - offset)
        low = np.where(expected < target, middle, low)
        high = np.where(expected >= target, middle, high)
    return _sigmoid((low + high) / 2)


def _meta_roles(counts, riots, tier, meta_data):
    total = counts.sum(axis=0)
    observed = counts >= 1
    eligible = observed & (counts >= total[np.newaxis, :] * .05)
    entry = (meta_data or {}).get(tier, {})
    strict = entry.get("strict") if isinstance(entry, dict) else None
    if isinstance(strict, dict):
        for role in range(5):
            selected = strict.get(str(role))
            if isinstance(selected, list):
                allowed = set(selected)
                eligible[role] &= np.asarray([index in allowed for index in range(len(riots))])
    return eligible


def _relative_weights(strength, evidence, diagonal=False):
    weights = np.zeros((5, 5)) if diagonal else np.ones((5, 5))
    for role in range(5):
        positions = [other for other in range(5) if not diagonal or role != other]
        values = np.asarray([strength[role, other] for other in positions])
        mean = float(values.mean())
        if mean > 1e-8:
            raw = np.asarray([1 + min(1, evidence[role, other] / 40)
                              * (strength[role, other] / mean - 1) for other in positions])
            raw = .25 + .75 * raw / raw.mean()
        else:
            raw = np.ones(len(positions))
        for other, value in zip(positions, raw):
            weights[role, other] = float(value)
    return weights.tolist()


def _bucket(rows, riots, tier, meta_data, min_samples, shrinkage):
    size = len(riots)
    teams = np.asarray([row[:10] for row in rows], dtype=np.int32)
    outcomes = np.asarray([row[10] for row in rows], dtype=float)
    blue, red = teams[:, :5], teams[:, 5:]
    blue_rate = float(outcomes.mean())
    counts, wins, blue_counts = (np.zeros((5, size), dtype=float) for _ in range(3))
    for role in range(5):
        counts[role] = np.bincount(blue[:, role], minlength=size) + np.bincount(red[:, role], minlength=size)
        wins[role] = np.bincount(blue[:, role], weights=outcomes, minlength=size) + np.bincount(red[:, role], weights=1 - outcomes, minlength=size)
        blue_counts[role] = np.bincount(blue[:, role], minlength=size)
    baseline = _neutral_baselines(counts, wins, blue_counts, blue_rate, shrinkage)
    eligible = _meta_roles(counts, riots, tier, meta_data)
    baseline_doc, population = {}, {}
    for role in range(5):
        baseline_doc[str(role)] = {riots[cid]: [int(counts[role, cid]), int(wins[role, cid]),
            int(blue_counts[role, cid]), round(float(baseline[role, cid]), 9)] for cid in range(size) if counts[role, cid]}
        population[str(role)] = {riots[cid]: int(counts[role, cid]) for cid in range(size) if eligible[role, cid]}
    tables = {"counter": {}, "synergy": {}}
    strengths = {kind: np.zeros((5, 5)) for kind in tables}
    evidence = {kind: np.zeros((5, 5), dtype=int) for kind in tables}
    offset = float(_logit(blue_rate))
    for kind in tables:
        pairs = ((first, second) for first in range(5) for second in range(5)
                 if kind == "counter" or first < second)
        for first, second in pairs:
            if kind == "counter":
                left = np.concatenate((blue[:, first], red[:, first]))
                right = np.concatenate((red[:, second], blue[:, second]))
            else:
                left = np.concatenate((blue[:, first], red[:, first]))
                right = np.concatenate((blue[:, second], red[:, second]))
            mask = eligible[first, left] & eligible[second, right]
            keys = left[mask] * size + right[mask]
            n = np.bincount(keys, minlength=size * size)
            w = np.bincount(keys, weights=np.concatenate((outcomes, 1 - outcomes))[mask], minlength=size * size)
            exposure = np.concatenate((np.ones(len(rows)), np.zeros(len(rows))))[mask]
            n_blue = np.bincount(keys, weights=exposure, minlength=size * size)
            selected = np.flatnonzero(n >= min_samples)
            if not selected.size:
                tables[kind][f"{first}:{second}"] = {}
                continue
            own, other = selected // size, selected % size
            logits = _logit(baseline[first, own]) + ((-1 if kind == "counter" else 1) * _logit(baseline[second, other]))
            expected = (n_blue[selected] * _sigmoid(logits + offset)
                        + (n[selected] - n_blue[selected]) * _sigmoid(logits - offset)) / n[selected]
            rate = w[selected] / n[selected]
            confidence = n[selected] / (n[selected] + shrinkage)
            delta = (rate - expected) * 100 * confidence
            noise = 100 * np.sqrt(expected * (1 - expected) / n[selected])
            corrected = np.sign(rate - expected) * np.sqrt(np.maximum((100 * (rate - expected)) ** 2 - noise ** 2, 0)) * confidence
            strength = float(np.median(np.abs(corrected)))
            strengths[kind][first, second] = strength
            evidence[kind][first, second] = len(selected)
            if kind == "synergy":
                strengths[kind][second, first] = strength
                evidence[kind][second, first] = len(selected)
            tables[kind][f"{first}:{second}"] = {f"{riots[a]}:{riots[b]}": [int(n[key]), int(w[key]), int(n_blue[key]),
                round(float(exp), 8), round(float(effect), 6)]
                for key, a, b, exp, effect in zip(selected, own, other, expected, delta)}
    matrices = {}
    for kind in tables:
        weights = _relative_weights(strengths[kind], evidence[kind], diagonal=kind == "synergy")
        matrices[f"{kind}_weights"] = weights
        matrices[f"{kind}_relative_percent"] = [[round(100 * (value - 1), 3) if kind != "synergy" or a != b else None
            for b, value in enumerate(row)] for a, row in enumerate(weights)]
        matrices[f"{kind}_median_adjusted_magnitude_pp"] = strengths[kind].tolist()
        matrices[f"{kind}_eligible_pairs"] = evidence[kind].tolist()
    return {"matches": len(rows), "blue_winrate": blue_rate,
            "baselines": baseline_doc, "population": population, "matrices": matrices, **tables}


def build_interaction_statistics(records, mapping, meta_data=None, *, patches=None, region=None,
                                 tiers=None, source="training_snapshot", min_samples=MIN_SAMPLES, shrinkage=SHRINKAGE):
    """Build compact pair evidence before fitting neural models, never in runtime."""
    if min_samples < 2 or shrinkage <= 0:
        raise ValueError("Interaction sample and shrinkage must be positive")
    by_internal = {int(index): str(riot) for riot, index in mapping.items()}
    if sorted(by_internal) != list(range(len(by_internal))):
        raise ValueError("Champion mapping must use dense internal indices")
    riots = [by_internal[index] for index in range(len(by_internal))]
    requested = list(dict.fromkeys(tier for tier in tiers if tier != "MIXED")) if tiers is not None else None
    patch_set = set(patches) if patches is not None else None
    grouped, all_rows, seen, observed_patches = {}, [], set(), set()
    excluded = 0
    for record in records:
        if not isinstance(record, dict):
            excluded += 1
            continue
        tier = record.get("tier")
        if (not isinstance(tier, str) or tier == "MIXED" or not tier
                or (requested is not None and tier not in requested)
                or (patch_set is not None and record.get("patch") not in patch_set)
                or (region is not None and record.get("region") != region)):
            excluded += 1
            continue
        blue, red = _team(record.get("blue_team")), _team(record.get("red_team"))
        if (not isinstance(record.get("blue_win"), bool) or blue is None or red is None
                or len(set(blue + red)) != 10 or any(champion not in mapping for champion in blue + red)):
            excluded += 1
            continue
        key = (record.get("region"), str(record["match_id"])) if record.get("match_id") else None
        if key is not None and key in seen:
            excluded += 1
            continue
        if key is not None:
            seen.add(key)
        row = [int(mapping[champion]) for champion in blue + red] + [int(record["blue_win"])]
        grouped.setdefault(tier, []).append(row)
        all_rows.append(row)
        if isinstance(record.get("patch"), str):
            observed_patches.add(record["patch"])
    buckets = {tier: _bucket(rows, riots, tier, meta_data, min_samples, shrinkage) for tier, rows in grouped.items()}
    if all_rows:
        buckets["MIXED"] = _bucket(all_rows, riots, "MIXED", meta_data, min_samples, shrinkage)
    scope = {"patches": list(patches) if patches is not None else sorted(observed_patches), "region": region,
             "tiers": requested if requested is not None else sorted(grouped), "source": source,
             "matches": len(all_rows), "excluded_records": excluded}
    for tier, bucket in buckets.items():
        bucket["scope"] = dict(scope, matches=bucket["matches"], tiers=scope["tiers"] if tier == "MIXED" else [tier])
    return {"version": 1, "roles": list(ROLE_NAMES), "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_samples": min_samples, "shrinkage": shrinkage,
            "method": "side-neutral champion odds; adjusted pair residual; sample shrinkage; noise-corrected median magnitude",
            "scope": scope, "tiers": buckets}


def _finite(value, lower, upper):
    return type(value) in (float, int) and math.isfinite(value) and lower <= value <= upper


def _count(value):
    return type(value) is int and value >= 0


def _riot_id(value):
    return (isinstance(value, str) and len(value) > 0 and value[0] != "0"
            and value.isascii() and value.isdigit())


def _valid_document(document):
    if not (isinstance(document, dict) and document.get("version") == 1 and document.get("roles") == list(ROLE_NAMES)
            and isinstance(document.get("tiers"), dict) and type(document.get("min_samples")) is int and document["min_samples"] >= 2
            and type(document.get("shrinkage")) in (float, int) and math.isfinite(document["shrinkage"]) and document["shrinkage"] > 0):
        return False
    for tier, bucket in document["tiers"].items():
        if (not isinstance(tier, str) or not tier or not isinstance(bucket, dict)
                or not _count(bucket.get("matches")) or not bucket["matches"]
                or not _finite(bucket.get("blue_winrate"), 0, 1)):
            return False
        if "scope" in bucket and not isinstance(bucket["scope"], dict):
            return False
        baselines, population, matrices = (bucket.get(key) for key in ("baselines", "population", "matrices"))
        if not all(isinstance(value, dict) for value in (baselines, population, matrices)):
            return False
        for role in map(str, range(5)):
            base, picks = baselines.get(role), population.get(role)
            if not isinstance(base, dict) or not isinstance(picks, dict):
                return False
            for riot, row in base.items():
                if (not _riot_id(riot) or not isinstance(row, list) or len(row) != 4
                        or not all(_count(value) for value in row[:3]) or row[0] < 1
                        or row[1] > row[0] or row[2] > row[0] or not _finite(row[3], 0, 1)):
                    return False
            if any(not _riot_id(riot) or riot not in base or not _count(count) or count != base[riot][0]
                   for riot, count in picks.items()):
                return False
        for kind in ("counter", "synergy"):
            matrix = matrices.get(f"{kind}_weights")
            if (not isinstance(matrix, list) or len(matrix) != 5
                    or any(not isinstance(row, list) or len(row) != 5 for row in matrix)
                    or any(not _finite(value, .25, 4) if kind == "counter" or first != second else value != 0
                           for first, row in enumerate(matrix) for second, value in enumerate(row))):
                return False
            relative = matrices.get(f"{kind}_relative_percent")
            if relative is not None and (not isinstance(relative, list) or len(relative) != 5
                    or any(not isinstance(row, list) or len(row) != 5 for row in relative)
                    or any(not _finite(value, -100, 400) if kind == "counter" or first != second else value is not None
                           for first, row in enumerate(relative) for second, value in enumerate(row))):
                return False
            tables = bucket.get(kind)
            if not isinstance(tables, dict):
                return False
            for pair, table in tables.items():
                if not isinstance(pair, str) or pair not in {f"{a}:{b}" for a in range(5) for b in range(5) if kind == "counter" or a < b} or not isinstance(table, dict):
                    return False
                first, second = pair.split(":")
                for champions, row in table.items():
                    ids = champions.split(":") if isinstance(champions, str) else []
                    if (len(ids) != 2 or not all(_riot_id(riot) for riot in ids)
                            or ids[0] not in population[first] or ids[1] not in population[second]
                            or not isinstance(row, list) or len(row) != 5
                            or not all(_count(value) for value in row[:3]) or row[0] < document["min_samples"]
                            or row[1] > row[0] or row[2] > row[0]
                            or not _finite(row[3], 0, 1) or not _finite(row[4], -100, 100)):
                        return False
    return True


def load_interaction_statistics(path=None):
    try:
        document = json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
        return document if _valid_document(document) else {}
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        return {}


def merge_interaction_statistics(previous, current, selected_tiers):
    if not _valid_document(current):
        raise ValueError("Invalid generated interaction statistics")
    if not _valid_document(previous):
        return current
    selected = set(selected_tiers) | {"MIXED"}
    return dict(current, tiers={**{tier: bucket for tier, bucket in previous["tiers"].items() if tier not in selected},
                                **current["tiers"]})


def save_interaction_statistics(path, document, *, selected_tiers=None):
    if selected_tiers is not None:
        document = merge_interaction_statistics(load_interaction_statistics(path), document, selected_tiers)
    if not _valid_document(document):
        raise ValueError("Invalid generated interaction statistics")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="interaction-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return document


class InteractionStatistics:
    def __init__(self, document, mapping):
        self.document = document if _valid_document(document) else {}
        self._riot = {int(index): str(riot) for riot, index in mapping.items()}
        self._internal = {riot: index for index, riot in self._riot.items()}

    def _bucket(self, tier):
        bucket = self.document.get("tiers", {}).get(tier)
        return bucket if isinstance(bucket, dict) else {}

    def available(self, tier):
        bucket = self._bucket(tier)
        return bool(bucket.get("matches") and any(any(bucket.get(kind, {}).values()) for kind in ("counter", "synergy")))

    def eligible(self, tier, champion, role):
        return self._riot.get(champion) in self._bucket(tier).get("population", {}).get(str(role), {})

    def baseline(self, tier, champion, role):
        row = self._bucket(tier).get("baselines", {}).get(str(role), {}).get(self._riot.get(champion))
        if not isinstance(row, list) or len(row) != 4 or row[0] <= 0:
            return None
        return {"matches": row[0], "wins": row[1], "observed_winrate": 100 * row[1] / row[0], "winrate": 100 * row[3]}

    def role_pick_rates(self, tier, role):
        counts = self._bucket(tier).get("population", {}).get(str(role), {})
        total = sum(counts.values())
        return {self._internal[riot]: count / total for riot, count in counts.items() if riot in self._internal} if total else {}

    def _edge(self, kind, tier, champion, role, other, other_role):
        if champion not in self._riot or other not in self._riot:
            return None
        if kind == "synergy" and role > other_role:
            champion, role, other, other_role = other, other_role, champion, role
        record = self._bucket(tier).get(kind, {}).get(f"{role}:{other_role}", {}).get(f"{self._riot[champion]}:{self._riot[other]}")
        if (not isinstance(record, list) or len(record) != 5 or record[0] < self.document.get("min_samples", MIN_SAMPLES)
                or any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in record)):
            return None
        return record

    def _summary(self, kind, tier, champion, role, other, other_role):
        row = self._edge(kind, tier, champion, role, other, other_role)
        if row is None:
            return None
        count, wins, _, expected, delta = row
        confidence = count / (count + self.document["shrinkage"])
        noise = 100 * math.sqrt(expected * (1 - expected) / count) * confidence
        corrected = math.copysign(math.sqrt(max(delta * delta - noise * noise, 0)), delta)
        base = self.baseline(tier, champion, role)
        return {"matches": count, "wins": wins, "winrate": 100 * wins / count, "expected_winrate": 100 * expected,
                "adjusted_delta_pp": delta, "noise_corrected_delta_pp": corrected, "uncertainty_pp": noise,
                "confidence": confidence, "adjusted_winrate": (base["winrate"] if base else 50) + delta}

    def counter(self, tier, champion, role, other, other_role):
        return self._summary("counter", tier, champion, role, other, other_role)

    def synergy(self, tier, champion, role, other, other_role):
        return self._summary("synergy", tier, champion, role, other, other_role)

    def prior(self, tier):
        bucket = self._bucket(tier)
        if not self.available(tier):
            return None
        eligible = {}
        for role in range(5):
            for riot in bucket.get("population", {}).get(str(role), {}):
                if riot in self._internal:
                    eligible.setdefault(self._internal[riot], []).append(role)
        return {"counter_weights": bucket["matrices"]["counter_weights"],
                "synergy_weights": bucket["matrices"]["synergy_weights"], "eligible": eligible}
