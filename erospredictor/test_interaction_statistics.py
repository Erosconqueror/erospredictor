"""Interaction caches remove champion/side baselines and retain sparse evidence."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from model.statistical import StatisticalModel
from services.interaction_statistics import (InteractionStatistics, build_interaction_statistics,
    load_interaction_statistics, merge_interaction_statistics, save_interaction_statistics)


MAPPING = {str(riot): riot - 1 for riot in range(1, 13)}


def counter_records(count=40):
    rows = []
    for own in (1, 2):
        for foe in (3, 4):
            for number in range(count):
                rows.append({"match_id": f"{own}:{foe}:{number}", "tier": "GOLD", "patch": "16.20", "region": "euw1",
                    "blue_team": [own, 5, 6, 7, 8], "red_team": [9, 10, foe, 11, 12],
                    "blue_win": (own == 1) == (foe == 3)})
    return rows


class InteractionBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = counter_records()
        cls.document = build_interaction_statistics(cls.records, MAPPING)

    def test_cross_role_counter_is_adjusted_against_both_marginal_baselines(self):
        cache = InteractionStatistics(self.document, MAPPING)
        self.assertAlmostEqual(cache.baseline("GOLD", 0, 0)["winrate"], 50, places=5)
        pair = cache.counter("GOLD", 0, 0, 3, 2)
        self.assertEqual(pair["matches"], 40)
        self.assertAlmostEqual(pair["expected_winrate"], 50, places=5)
        self.assertAlmostEqual(pair["adjusted_delta_pp"], -100 / 3, places=5)
        self.assertLess(pair["noise_corrected_delta_pp"], -30)
        self.assertGreater(pair["uncertainty_pp"], 0)
        reverse = cache.counter("GOLD", 3, 2, 0, 0)
        self.assertAlmostEqual(reverse["adjusted_delta_pp"], -pair["adjusted_delta_pp"], places=5)

    def test_blue_side_bias_alone_does_not_create_a_champion_counter(self):
        rows = [{"tier": "GOLD", "blue_team": [1, 5, 6, 7, 8], "red_team": [9, 10, 3, 11, 12],
                 "blue_win": index < 80} for index in range(100)]
        cache = InteractionStatistics(build_interaction_statistics(rows, MAPPING), MAPPING)
        self.assertAlmostEqual(cache.baseline("GOLD", 0, 0)["winrate"], 50, places=5)
        pair = cache.counter("GOLD", 0, 0, 2, 2)
        self.assertAlmostEqual(pair["expected_winrate"], 80, places=5)
        self.assertAlmostEqual(pair["adjusted_delta_pp"], 0, places=5)

    def test_side_exposure_correction_also_handles_uneven_sides_and_winning_champions(self):
        rows = []
        for index in range(200):
            blue = [1, 5, 6, 7, 8]
            red = [9, 10, 3, 11, 12]
            if index >= 100:
                blue, red = red, blue
            rows.append({"tier": "GOLD", "blue_team": blue, "red_team": red,
                         "blue_win": index % 100 < 80})
        cache = InteractionStatistics(build_interaction_statistics(rows, MAPPING), MAPPING)
        self.assertAlmostEqual(cache.baseline("GOLD", 0, 0)["winrate"], 50, places=5)
        self.assertAlmostEqual(cache.counter("GOLD", 0, 0, 2, 2)["adjusted_delta_pp"], 0, places=5)

    def test_relative_matrices_use_positive_weights_and_preserve_same_role_synergy_holes(self):
        matrices = self.document["tiers"]["GOLD"]["matrices"]
        for role, row in enumerate(matrices["counter_weights"]):
            self.assertAlmostEqual(sum(row), 5)
            self.assertTrue(all(.25 <= value <= 4 for value in row))
            self.assertAlmostEqual(matrices["counter_relative_percent"][role][2], 100 * (row[2] - 1), places=3)
        self.assertGreater(matrices["counter_weights"][0][2], matrices["counter_weights"][0][0])
        for role, row in enumerate(matrices["synergy_weights"]):
            self.assertEqual(row[role], 0)
            self.assertAlmostEqual(sum(row), 4)
            self.assertIsNone(matrices["synergy_relative_percent"][role][role])

    def test_off_meta_endpoints_are_excluded_without_deleting_their_baseline(self):
        meta = {tier: {"strict": {"0": [1, 8]}} for tier in ("GOLD", "MIXED")}
        doc = build_interaction_statistics(self.records, MAPPING, meta)
        cache = InteractionStatistics(doc, MAPPING)
        self.assertFalse(cache.eligible("GOLD", 0, 0))
        self.assertIsNotNone(cache.baseline("GOLD", 0, 0))
        self.assertIsNone(cache.counter("GOLD", 0, 0, 3, 2))
        self.assertNotIn(0, cache.role_pick_rates("GOLD", 0))
        prior = cache.prior("GOLD")
        self.assertNotIn(0, prior["eligible"])

    def test_population_rates_are_role_specific_and_internal_zero_is_valid(self):
        cache = InteractionStatistics(self.document, MAPPING)
        rates = cache.role_pick_rates("GOLD", 0)
        self.assertIn(0, rates)
        self.assertAlmostEqual(sum(rates.values()), 1)
        self.assertNotIn(0, cache.role_pick_rates("GOLD", 2))

    def test_supported_synergy_remains_available_when_counter_samples_are_missing(self):
        doc = deepcopy(self.document)
        for bucket in doc["tiers"].values():
            bucket["counter"] = {}
        cache = InteractionStatistics(doc, MAPPING)
        self.assertTrue(cache.available("GOLD"))
        self.assertIsNotNone(cache.prior("GOLD"))
        self.assertIsNotNone(cache.synergy("GOLD", 0, 0, 4, 1))
        self.assertIsNone(cache.counter("GOLD", 0, 0, 3, 2))

    def test_exact_window_filters_duplicates_invalid_ids_and_non_boolean_results(self):
        rows = self.records + [deepcopy(self.records[0]), dict(self.records[0], match_id="old", patch="16.1"),
                              dict(self.records[0], match_id="bad", blue_win="false")]
        doc = build_interaction_statistics(rows, MAPPING, patches=["16.20"], region="euw1", tiers=["GOLD"])
        self.assertEqual(doc["scope"]["matches"], len(self.records))
        self.assertEqual(doc["scope"]["excluded_records"], 3)
        self.assertEqual(doc["tiers"]["GOLD"]["scope"]["patches"], ["16.20"])
        self.assertEqual(doc["tiers"]["MIXED"]["scope"]["tiers"], ["GOLD"])

    def test_partial_merge_preserves_untouched_rank_and_its_old_scope(self):
        previous = deepcopy(self.document)
        previous["tiers"]["SILVER"] = deepcopy(previous["tiers"]["GOLD"])
        previous["tiers"]["SILVER"]["scope"]["patches"] = ["15.10"]
        merged = merge_interaction_statistics(previous, self.document, ["GOLD"])
        self.assertEqual(merged["tiers"]["SILVER"]["scope"]["patches"], ["15.10"])
        self.assertEqual(merged["tiers"]["GOLD"], self.document["tiers"]["GOLD"])

    def test_artifact_uses_stable_ids_and_survives_a_changed_internal_catalog(self):
        shuffled = {riot: 11 - internal for riot, internal in MAPPING.items()}
        cache = InteractionStatistics(self.document, shuffled)
        self.assertIsNotNone(cache.counter("GOLD", 11, 0, 8, 2))
        self.assertEqual(cache.counter("GOLD", 11, 0, 8, 2)["matches"], 40)

    def test_future_positive_riot_ids_have_no_artificial_digit_limit(self):
        mapping = {("10000001" if riot == "1" else riot): internal for riot, internal in MAPPING.items()}
        rows = deepcopy(self.records)
        for row in rows:
            row["blue_team"] = [10000001 if champion == 1 else champion for champion in row["blue_team"]]
        document = build_interaction_statistics(rows, mapping)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "interaction_stats.json"
            save_interaction_statistics(path, document)
            cache = InteractionStatistics(load_interaction_statistics(path), mapping)
        self.assertTrue(cache.available("GOLD"))
        self.assertEqual(cache.counter("GOLD", 0, 0, 3, 2)["matches"], 40)

    def test_save_load_and_corrupt_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "interaction_stats.json"
            save_interaction_statistics(path, self.document)
            self.assertEqual(load_interaction_statistics(path), self.document)
            path.write_text('{"version":1,"min_samples":"twenty"}')
            self.assertEqual(load_interaction_statistics(path), {})

    def test_nested_corrupt_artifacts_become_unknown_without_breaking_queries(self):
        for key, value in (("population", {"0": []}), ("matrices", {"counter_weights": [[float('nan')]]}),
                           ("baselines", {"0": []}), ("counter", {"0:2": {"1:4": [40, 50, 0, .5, -3]}})):
            with self.subTest(key=key):
                doc = deepcopy(self.document)
                doc["tiers"]["GOLD"][key] = value
                cache = InteractionStatistics(doc, MAPPING)
                self.assertFalse(cache.available("GOLD"))
                self.assertEqual(cache.role_pick_rates("GOLD", 0), {})
                self.assertIsNone(cache.counter("GOLD", 0, 0, 3, 2))
                self.assertIsNone(cache.prior("GOLD"))

    def test_compact_atomic_write_failure_preserves_the_previous_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "interaction_stats.json"
            save_interaction_statistics(path, self.document)
            before = path.read_bytes()
            self.assertNotIn(b'\n  ', before)
            with patch("services.interaction_statistics.os.replace", side_effect=PermissionError("busy")):
                with self.assertRaises(PermissionError):
                    save_interaction_statistics(path, self.document)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).glob("interaction-*.tmp")), [])


class ExpandedStatisticalTests(unittest.TestCase):
    def setUp(self):
        rows = counter_records()
        self.db = SimpleNamespace(get_champion_mapping=lambda: MAPPING, get_all_matches=lambda: rows)
        self.model = StatisticalModel(self.db)
        self.model.build_stats()
        self.legacy = deepcopy(self.model.matchups)
        self.model.interactions = InteractionStatistics(build_interaction_statistics(rows, MAPPING), MAPPING)

    def test_cross_role_edge_changes_prediction_and_is_reported_as_available(self):
        blue, red = [0, -1, -1, -1, -1], [-1, -1, 3, -1, -1]
        self.assertEqual(self.model._predict_legacy("GOLD", blue, red), .5)
        self.assertTrue(self.model.has_interactions("GOLD"))
        self.assertTrue(self.model.uses_interactions("GOLD", blue, red))
        self.assertLess(self.model.predict("GOLD", blue, red), .25)
        self.assertGreater(self.model.predict("GOLD", red, blue), .75)

    def test_empty_unknown_and_old_cache_use_the_exact_original_predictor(self):
        blue, red = [0, 4, 5, 6, 7], [8, 9, 2, 10, 11]
        self.model.interactions = None
        self.assertFalse(self.model.uses_interactions("GOLD", blue, red))
        self.assertEqual(self.model.predict("GOLD", blue, red), self.model._predict_legacy("GOLD", blue, red))
        self.assertEqual(self.model.predict("GOLD", [-1] * 5, [-1] * 5), .5)
        self.assertEqual(self.model.matchups, self.legacy)

    def test_only_off_meta_picks_keep_the_exact_original_predictor(self):
        meta = {tier: {"strict": {"0": []}} for tier in ("GOLD", "MIXED")}
        doc = build_interaction_statistics(counter_records(), MAPPING, meta)
        self.model.interactions = InteractionStatistics(doc, MAPPING)
        blue, red = [0, -1, -1, -1, -1], [8, -1, -1, -1, -1]
        self.assertTrue(self.model.has_interactions("GOLD"))
        self.assertFalse(self.model.uses_interactions("GOLD", blue, red))
        self.assertEqual(self.model.predict("GOLD", blue, red), self.model._predict_legacy("GOLD", blue, red))

    def test_full_draft_is_finite_and_old_lane_evidence_is_still_present(self):
        self.assertIsNotNone(self.model.get_lane_matchup("GOLD", 0, 8, 0))
        probability = self.model.predict("GOLD", [0, 4, 5, 6, 7], [8, 9, 3, 10, 11])
        self.assertTrue(.01 <= probability <= .99)

    def test_sibling_artifact_is_loaded_only_with_the_corresponding_statistics_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stats_cache.json"
            artifact = build_interaction_statistics(counter_records(), MAPPING)
            save_interaction_statistics(path.with_name("interaction_stats.json"), artifact)
            self.db.load_stats_cache = lambda _path: self.legacy
            fresh = StatisticalModel(self.db)
            self.assertTrue(fresh.load_cache(str(path)))
            self.assertTrue(fresh.has_interactions("GOLD"))
            self.assertFalse(fresh.has_interactions("IRON"))
            path.with_name("interaction_stats.json").write_text('{"broken": true}')
            fresh.load_cache(str(path))
            self.assertFalse(fresh.has_interactions("GOLD"))


if __name__ == "__main__":
    unittest.main()
