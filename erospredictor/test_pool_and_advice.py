"""Stable own lists, bounded flex evaluation and cached ban recommendations."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from services import champion_pools as pools
from services.draft_assignment import DraftScenario
from services.robust_recommendation import rerank
from services.ban_advisor import recommend_bans


MAPPING = {"266": 0, "103": 1, "55": 2}


class PoolTests(unittest.TestCase):
    def test_core_list_filter_still_respects_bans_picks_and_role_filter(self):
        from model.core_model import CoreModel
        core = CoreModel(SimpleNamespace())
        core.meta_data = {"GOLD": {"strict": {"2": [0, 1, 2]}}}
        core.predict_match = MagicMock(return_value={"blue_win_prob": 55})
        blue, red = [0, -1, -1, -1, -1], [-1] * 5
        result = core.recommend_champs("GOLD", blue, red, [2], "blue", 2, candidate_ids={0, 1, 2})
        self.assertEqual([item["id"] for item in result], [1])
        core.predict_match.assert_called_once()
        self.assertEqual(blue, [0, -1, -1, -1, -1])
        core.predict_match.reset_mock()
        self.assertEqual(core.recommend_champs("GOLD", blue, red, [], "blue", 2, candidate_ids=set()), [])
        core.predict_match.assert_not_called()

    def test_tags_overlap_named_lists_are_independent_and_stable_ids_are_persisted(self):
        document = {"version": 1, "champions": {"266": ["played", "comfort", "learn"]},
                    "lists": {"Duo": ["266", "103"]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pools.json"
            pools.save_library(document, path)
            loaded = pools.load_library(path)
        self.assertEqual(pools.member_ids(loaded, MAPPING), {0})
        self.assertEqual(pools.member_ids(loaded, MAPPING, "Duo"), {0, 1})
        remapped = {"266": 80, "103": 42}
        self.assertEqual(pools.member_ids(loaded, remapped, "Duo"), {80, 42})
        self.assertEqual(set(pools.internal_tags(loaded, MAPPING)[0]), {"played", "comfort", "learn"})

    def test_history_import_uses_already_loaded_games_without_mutating_input(self):
        original = pools.empty_library()
        history = SimpleNamespace(games=[{"champion": 266}, {"champion": 99999}])
        updated = pools.include_history(original, history, MAPPING)
        self.assertEqual(original, pools.empty_library())
        self.assertEqual(updated["champions"], {"266": ["played"]})

    def test_corrupt_files_and_invalid_tags_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pools.json"
            path.write_text("broken")
            self.assertEqual(pools.load_library(path), pools.empty_library())
            with self.assertRaises(ValueError):
                pools.save_library({"version": 1, "champions": {"266": ["secret"]}, "lists": {}}, path)


class RobustTests(unittest.TestCase):
    def setUp(self):
        self.blue, self.red = [-1] * 5, [-1, -1, 10, -1, 11]
        other = [-1, -1, 11, -1, 10]
        self.scenarios = (DraftScenario(tuple(self.blue), tuple(self.red), .7),
                          DraftScenario(tuple(self.blue), tuple(other), .3))
        self.recs = [{"id": i, "wr": 60 - i} for i in range(10)]

    def test_only_first_five_are_reranked_with_original_percentages_preserved(self):
        before = deepcopy(self.recs)
        predict = MagicMock(side_effect=lambda div, b, r: {"blue_win_prob": 40 if b[0] == 0 else 55 - b[0] / 10})
        result = rerank(self.recs, predict, "GOLD", self.blue, self.red, "blue", 0, self.scenarios)
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["wr"], 59)
        self.assertEqual(predict.call_count, 5)
        self.assertEqual([row["id"] for row in result[5:]], [5, 6, 7, 8, 9])
        self.assertEqual(self.recs, before)
        self.assertEqual(result[0]["robust"]["count"], 2)
        self.assertTrue(all(scenario["blue"][0] == 1 for scenario in result[0]["robust"]["scenarios"]))

    def test_invalid_scenarios_cannot_change_own_picks_or_introduce_enemy_champions(self):
        invalid = DraftScenario((9, -1, -1, -1, -1), tuple(self.red), .5)
        other = DraftScenario(tuple(self.blue), (-1, -1, 20, -1, 11), .5)
        predict = MagicMock(side_effect=AssertionError("Unexpected inference"))
        result = rerank(self.recs, predict, "GOLD", self.blue, self.red, "blue", 0,
                        (self.scenarios[0], invalid, other))
        predict.assert_not_called()
        self.assertEqual([item["id"] for item in result], list(range(10)))
        self.assertEqual(result[0]["robust"]["count"], 1)

    def test_red_side_uses_red_probability_and_forced_flag_survives(self):
        scenarios = [DraftScenario(item.red, item.blue, item.weight, True) for item in self.scenarios]
        predict = MagicMock(return_value={"red_win_prob": 45})
        result = rerank(self.recs, predict, "GOLD", self.red, self.blue, "red", 0, scenarios)
        self.assertTrue(result[0]["robust"]["forced"])
        self.assertEqual(result[0]["robust"]["min"], 45)


class BanAdviceTests(unittest.TestCase):
    def setUp(self):
        self.stats = MagicMock()
        self.stats.get_lane_matchup.return_value = {"matches": 100, "weighted_winrate": 40}
        self.roles = {"version": 1, "roles": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
                      "tiers": {"MIXED": {riot: {"counts": [0, 0, 100, 0, 0], "total": 100,
                                  "probabilities": [0, 0, 1, 0, 0]} for riot in MAPPING}}}

    def test_uses_cached_lane_samples_and_excludes_own_pool_and_existing_bans(self):
        result = recommend_bans(self.stats, self.roles, MAPPING, {0, 1}, "MIXED", 2)
        self.assertEqual([item["id"] for item in result], [2])
        self.assertEqual(result[0]["covered"], 2)
        self.assertEqual(result[0]["pool_size"], 2)
        self.assertEqual(len(result[0]["matchups"]), 2)
        self.assertGreater(result[0]["score"], 0)
        self.assertEqual(recommend_bans(self.stats, self.roles, MAPPING, {0, 1}, "MIXED", 2, {2}), [])

    def test_small_samples_and_no_disadvantage_do_not_produce_misleading_bans(self):
        for summary in ({"matches": 2, "weighted_winrate": 0}, {"matches": 100, "weighted_winrate": 60}, None):
            self.stats.get_lane_matchup.return_value = summary
            self.assertEqual(recommend_bans(self.stats, self.roles, MAPPING, {0, 1}, "MIXED", 2), [])


if __name__ == "__main__":
    unittest.main()
