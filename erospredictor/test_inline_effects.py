"""Deterministic checks for the optional single-champion draft sensitivity."""

import copy
import unittest
from unittest.mock import patch

from model.core_model import CoreModel


class PairInteractionModel(CoreModel):
    """A nonlinear focus interaction fixture, independent of neural packages."""

    def __init__(self):
        super().__init__(None)
        self.prediction_calls = []

    def predict_match(self, div, blue, red):
        self.prediction_calls.append((tuple(blue), tuple(red)))

        def strength(own, enemy):
            if 0 not in own:
                return 0.0
            return (3.0 + (4.0 if 1 in own else 0.0)
                    + (2.0 if 2 in enemy else 0.0) - (5.0 if 3 in enemy else 0.0))

        blue_probability = 50.0 + strength(blue, red) - strength(red, blue)
        return {"blue_win_prob": blue_probability, "red_win_prob": 100.0 - blue_probability}


class InlineEffectTests(unittest.TestCase):
    def setUp(self):
        self.model = PairInteractionModel()
        self.blue = [0, 1, -1, -1, -1]
        self.red = [2, 3, -1, -1, -1]

    def test_occupied_focus_has_signed_candidate_specific_effects_and_absolute_teams(self):
        original = copy.deepcopy((self.blue, self.red))
        result = self.model.explain_champion_effects("GOLD", self.blue, self.red, "blue", 0)
        self.assertEqual((self.blue, self.red), original)
        self.assertEqual(result["id"], 0)
        self.assertEqual(result["wr"], 54.0)
        self.assertEqual(result["baseline_win_prob"], 50.0)
        self.assertEqual(result["candidate_delta_pp"], 4.0)
        effects = {item["champion_id"]: item for item in result["interactions"]}
        self.assertEqual(set(effects), {1, 2, 3})
        self.assertEqual(effects[1]["delta_pp"], 4.0)
        self.assertEqual(effects[2]["delta_pp"], 2.0)
        self.assertEqual(effects[3]["delta_pp"], -5.0)
        self.assertEqual(effects[1]["side"], "ally")
        self.assertEqual(effects[1]["team"], "blue")
        self.assertEqual(effects[2]["team"], "red")
        self.assertTrue(effects[2]["is_lane_opponent"])
        self.assertFalse(effects[3]["is_lane_opponent"])

    def test_candidate_for_empty_position_matches_already_selected_champion(self):
        empty = [-1, 1, -1, -1, -1]
        result = self.model.explain_champion_effects("GOLD", empty, self.red, "blue", 0, champion_id=0)
        selected = self.model.explain_champion_effects("GOLD", self.blue, self.red, "blue", 0)
        self.assertEqual(result, selected)
        self.assertEqual(empty[0], -1)

    def test_red_perspective_keeps_beneficial_and_unfavorable_signs(self):
        result = self.model.explain_champion_effects("GOLD", self.red, self.blue, "red", 0)
        self.assertEqual(result["wr"], 54.0)
        self.assertEqual(result["team"], "red")
        effects = {item["champion_id"]: item for item in result["interactions"]}
        self.assertEqual(effects[1]["delta_pp"], 4.0)
        self.assertEqual(effects[1]["team"], "red")
        self.assertEqual(effects[3]["delta_pp"], -5.0)
        self.assertEqual(effects[3]["team"], "blue")
        self.assertTrue(effects[2]["is_lane_opponent"])

    def test_full_draft_has_at_most_twenty_memoized_prediction_calls(self):
        blue, red = [0, 1, 4, 5, 6], [2, 3, 7, 8, 9]
        with patch.object(self.model, "explain_recommendations", side_effect=AssertionError("top five")), \
                patch.object(self.model, "_prediction_components", side_effect=AssertionError("components")):
            result = self.model.explain_champion_effects("GOLD", blue, red, "blue", 0)
        self.assertEqual(len(result["interactions"]), 9)
        self.assertEqual(len(self.model.prediction_calls), 20)
        self.assertEqual(len(set(self.model.prediction_calls)), 20)

    def test_lone_focus_needs_only_current_and_empty_baseline(self):
        result = self.model.explain_champion_effects("GOLD", [0, -1, -1, -1, -1], [-1] * 5, "blue", 0)
        self.assertEqual(result["interactions"], [])
        self.assertEqual(len(self.model.prediction_calls), 2)

    def test_invalid_or_stale_requests_stop_before_any_prediction(self):
        invalid = [
            (self.blue, self.red, "sideways", 0, None),
            (self.blue, self.red, "blue", True, None),
            (self.blue, self.red, "blue", 5, None),
            ([-1] * 5, self.red, "blue", 0, None),
            (self.blue, self.red, "blue", 0, 4),
            ([-1] * 5, self.red, "blue", 0, 2),
            (self.blue, self.red, "blue", 0, -1),
            (self.blue, self.red, "blue", 0, True),
            ([0, 0, -1, -1, -1], self.red, "blue", 0, None),
        ]
        for blue, red, team, role, champion in invalid:
            with self.subTest(team=team, role=role, champion=champion):
                with self.assertRaises(ValueError):
                    self.model.explain_champion_effects("GOLD", blue, red, team, role, champion)
        self.assertEqual(self.model.prediction_calls, [])

    def test_controller_forwards_the_explicit_optional_request(self):
        try:
            from controller.controller import Controller
        except ImportError:
            self.skipTest("PostgreSQL adapter is optional for model-only tests")
        controller = Controller.__new__(Controller)
        controller.core_model = self.model
        result = controller.explain_champion_effects("GOLD", self.blue, self.red, "blue", 0)
        self.assertEqual(result["id"], 0)
        self.assertEqual(result["wr"], 54.0)


if __name__ == "__main__":
    unittest.main()
