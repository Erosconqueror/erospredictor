"""The 30% floor requires support on every applicable edge, with exact boundaries."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from model.statistical import StatisticalModel
from services.ensemble_weights import draft_minimums
from services.interaction_statistics import InteractionStatistics, build_interaction_statistics
from test_interaction_statistics import MAPPING, counter_records


class StatisticalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = counter_records(count=50)
        cls.document = build_interaction_statistics(rows, MAPPING, min_samples=2)
        cls.blue = [MAPPING[str(cid)] for cid in rows[0]["blue_team"]]
        cls.red = [MAPPING[str(cid)] for cid in rows[0]["red_team"]]

    def setUp(self):
        self.doc = deepcopy(self.document)
        self.model = StatisticalModel(SimpleNamespace(get_champion_mapping=lambda: MAPPING))

    def floor(self, blue=None, red=None, tier="GOLD"):
        self.model.interactions = InteractionStatistics(self.doc, MAPPING)
        return draft_minimums(self.model, tier, self.blue if blue is None else blue,
                             self.red if red is None else red)["statistical"]

    def set_count(self, kind, roles, champions, count):
        self.doc["tiers"]["GOLD"][kind][roles][champions] = [count, count // 2, count // 2, .5, 0.0]

    def test_every_main_lane_pair_with_one_hundred_games_qualifies(self):
        self.assertEqual(self.floor(), .30)
        self.assertEqual(self.floor(self.red, self.blue), .30)

    def test_one_main_edge_with_ninety_nine_games_blocks_thirty_percent(self):
        self.set_count("counter", "0:0", "1:9", 99)
        self.assertEqual(self.floor(), .16)
        self.set_count("counter", "0:0", "1:9", 100)
        self.assertEqual(self.floor(), .30)

    def test_other_counter_edges_need_twenty_games_not_one_hundred(self):
        self.set_count("counter", "0:2", "1:3", 20)
        self.assertEqual(self.floor(), .30)
        self.set_count("counter", "0:2", "1:3", 19)
        self.assertEqual(self.floor(), .16)

    def test_ally_synergies_on_both_teams_need_twenty_games(self):
        for key in ("1:5", "9:10"):
            with self.subTest(pair=key):
                self.set_count("synergy", "0:1", key, 20)
                self.assertEqual(self.floor(), .30)
                self.set_count("synergy", "0:1", key, 19)
                self.assertEqual(self.floor(), .16)
                self.set_count("synergy", "0:1", key, 100)

    def test_missing_counter_or_synergy_cannot_pass_by_ignoring_that_edge(self):
        for kind, roles, champions in (("counter", "0:2", "1:3"), ("synergy", "0:1", "1:5")):
            with self.subTest(kind=kind):
                saved = self.doc["tiers"]["GOLD"][kind][roles].pop(champions)
                self.assertEqual(self.floor(), .16)
                self.doc["tiers"]["GOLD"][kind][roles][champions] = saved

    def test_empty_slots_are_ignored_but_at_least_one_lane_pair_is_required(self):
        self.assertEqual(self.floor([0, -1, -1, -1, -1], [8, -1, -1, -1, -1]), .30)
        self.assertEqual(self.floor([-1] * 5, [-1] * 5), .16)
        self.assertEqual(self.floor([0, -1, -1, -1, -1], [-1, 9, -1, -1, -1]), .16)

    def test_missing_tier_does_not_borrow_mixed_evidence(self):
        self.assertEqual(self.floor(tier="IRON"), .16)
        self.assertEqual(self.floor(tier="MIXED"), .30)

    def test_legacy_lane_and_bot_counts_also_obey_the_gate(self):
        self.model.interactions = None
        self.model.matchups = {"GOLD": {"TOP": {"0": {"8": {"m": 99, "w": 50}}},
                                        "BOT_SYNERGY": {"6": {"7": {"m": 19, "w": 10}}}}}
        blue, red = [0, -1, -1, 6, 7], [8, -1, -1, -1, -1]
        self.assertEqual(draft_minimums(self.model, "GOLD", blue, red)["statistical"], .16)
        self.model.matchups["GOLD"]["TOP"]["0"]["8"]["m"] = 100
        self.assertEqual(draft_minimums(self.model, "GOLD", blue, red)["statistical"], .16)
        self.model.matchups["GOLD"]["BOT_SYNERGY"]["6"]["7"]["m"] = 20
        self.assertEqual(draft_minimums(self.model, "GOLD", blue, red)["statistical"], .30)


if __name__ == "__main__":
    unittest.main()
