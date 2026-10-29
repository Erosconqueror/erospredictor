"""Role-specific ban ranking blends only sufficient, relevant personal evidence."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from services.ban_advisor import recommend_bans
from services.personal_history import PersonalHistory


MAPPING = {"266": 0, "103": 1, "55": 2, "101": 3, "201": 4}


def role_document():
    return {"version": 1, "roles": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
        "tiers": {"MIXED": {riot: {"counts": [100] * 5, "probabilities": [.2] * 5, "total": 500}
                            for riot in MAPPING}}}


def games(count, wins=0, champion=266, enemy=55, role=2, prefix="test"):
    return [{"match_id": f"{prefix}:{index}", "champion": champion, "enemy": enemy,
             "role": role, "win": index < wins} for index in range(count)]


def history(rows):
    return PersonalHistory("test-puuid", "Test#EU", tuple(rows))


class PersonalBanAdviceTests(unittest.TestCase):
    def setUp(self):
        self.stats = MagicMock()
        self.stats.get_lane_matchup.return_value = {"matches": 100, "weighted_winrate": 40.0}
        self.roles = role_document()

    def recommend(self, rows=(), pool=(0,), role=2, unavailable=()):
        return recommend_bans(self.stats, self.roles, MAPPING, pool, "MIXED", role,
                              unavailable, history=history(rows))

    def test_ten_games_activate_exact_half_and_half_but_nine_keep_population_rate(self):
        nine = next(item for item in self.recommend(games(9, wins=2)) if item["id"] == 2)
        self.assertEqual(nine["personal"]["games"], 9)
        self.assertFalse(nine["personal"]["enough"])
        self.assertEqual(nine["personal_weight"], 0)
        self.assertAlmostEqual(nine["combined_winrate"], 40)
        ten = next(item for item in self.recommend(games(10, wins=2)) if item["id"] == 2)
        self.assertEqual(ten["personal_weight"], .5)
        self.assertAlmostEqual(ten["personal"]["winrate"], 20)
        self.assertAlmostEqual(ten["global_winrate"], 40)
        self.assertAlmostEqual(ten["combined_winrate"], 30)

    def test_good_personal_results_can_remove_a_population_counter_from_the_list(self):
        results = self.recommend(games(10, wins=8))
        self.assertNotIn(2, [item["id"] for item in results])  # (40 + 80) / 2 = 60.
        self.assertIn(3, [item["id"] for item in results])

    def test_bad_personal_results_can_promote_an_otherwise_favorable_matchup(self):
        self.stats.get_lane_matchup.side_effect = lambda div, own, enemy, role: {
            "matches": 100, "weighted_winrate": 60 if enemy == 2 else 40}
        self.assertNotIn(2, [item["id"] for item in self.recommend()])
        results = self.recommend(games(10))
        self.assertEqual(results[0]["id"], 2)
        self.assertAlmostEqual(results[0]["combined_winrate"], 30)

    def test_only_unique_games_on_the_selected_role_and_pool_against_known_lane_enemy_count(self):
        relevant = games(9)
        irrelevant = (relevant[:1] + games(15, role=0, prefix="top")
            + games(15, champion=103, prefix="outside-pool")
            + games(15, enemy=None, prefix="unknown-enemy")
            + games(15, role=None, prefix="unknown-role"))
        row = next(item for item in self.recommend(relevant + irrelevant) if item["id"] == 2)
        self.assertEqual(row["personal"]["games"], 9)
        self.assertEqual(row["personal_weight"], 0)

    def test_threshold_is_per_enemy_and_role_across_the_selected_pool(self):
        rows = games(5, champion=266) + games(5, champion=103, prefix="ahri")
        rows += games(9, enemy=101, prefix="xerath")
        by_id = {item["id"]: item for item in self.recommend(rows, pool=(0, 1))}
        self.assertEqual(by_id[2]["personal"]["games"], 10)
        self.assertEqual(by_id[2]["personal_weight"], .5)
        self.assertEqual(by_id[3]["personal_weight"], 0)

    def test_top_and_mid_have_independent_personal_results(self):
        rows = games(10, wins=10) + games(10, role=0, prefix="top")
        self.assertNotIn(2, [item["id"] for item in self.recommend(rows, role=2)])
        top = next(item for item in self.recommend(rows, role=0) if item["id"] == 2)
        self.assertEqual(top["personal"]["wins"], 0)
        self.assertAlmostEqual(top["combined_winrate"], 20)

    def test_population_average_retains_sample_reliability_and_signed_favorable_results(self):
        self.stats.get_lane_matchup.side_effect = lambda div, own, enemy, role: {
            "matches": 20 if own == 0 else 100, "weighted_winrate": 20 if own == 0 else 60}
        item = self.recommend(pool=(0, 1))[0]
        expected = (20 * .5 + 60 * (100 / 120)) / (.5 + 100 / 120)
        self.assertAlmostEqual(item["global_winrate"], expected)
        self.assertAlmostEqual(item["combined_winrate"], expected)
        self.assertEqual(item["global_matches"], 120)

    def test_personal_games_do_not_bypass_population_sample_or_availability_guards(self):
        self.stats.get_lane_matchup.return_value = {"matches": 19, "weighted_winrate": 0}
        self.assertEqual(self.recommend(games(10)), [])
        self.stats.get_lane_matchup.return_value = {"matches": 20, "weighted_winrate": 0}
        ids = {item["id"] for item in self.recommend(games(10), unavailable=(2,))}
        self.assertNotIn(0, ids)
        self.assertNotIn(2, ids)
        self.assertTrue(ids)

    def test_missing_personal_history_keeps_general_result_and_has_no_fetch_side_effect(self):
        result = recommend_bans(self.stats, self.roles, MAPPING, (0,), "MIXED", 2)
        self.assertTrue(result)
        self.assertTrue(all(item["personal"] is None and item["personal_weight"] == 0 for item in result))
        self.assertEqual({call[0] for call in self.stats.mock_calls}, {"get_lane_matchup"})

    def test_champion_zero_can_also_be_a_personal_opponent(self):
        row = next(item for item in self.recommend(games(10, champion=103, enemy=266), pool=(1,)) if item["id"] == 0)
        self.assertEqual(row["personal_weight"], .5)


class BanListControllerTests(unittest.TestCase):
    def test_all_roles_share_one_loaded_distribution_and_do_not_run_neural_models(self):
        from controller.controller import Controller
        controller = Controller.__new__(Controller)
        controller.db = SimpleNamespace(get_champion_mapping=MagicMock(return_value=MAPPING))
        controller.stat_model = MagicMock()
        controller.stat_model.get_lane_matchup.return_value = {"matches": 100, "weighted_winrate": 40}
        controller.core_model = MagicMock()
        personal = history(games(10))
        with patch("services.role_distribution.load_role_distribution", return_value=role_document()) as load:
            lists = controller.recommend_ban_lists("MIXED", {0}, (), personal)
        self.assertEqual(set(lists), set(range(5)))
        self.assertTrue(all(lists.values()))
        self.assertEqual(lists[2][0]["id"], 2)
        self.assertEqual(lists[2][0]["personal_weight"], .5)
        self.assertTrue(all(item["personal_weight"] == 0 for item in lists[0]))
        load.assert_called_once()
        controller.db.get_champion_mapping.assert_called_once()
        self.assertEqual(controller.core_model.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
