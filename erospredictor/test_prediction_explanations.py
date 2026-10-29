"""Single-focus prediction explanations reuse diagnostics without ranking picks."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from services.explanation_format import explanations_html
from test_recommend_explanations import InteractionModel


class PredictionExplanationTests(unittest.TestCase):
    def setUp(self):
        self.model = InteractionModel()
        self.blue = [0, 1, -1, -1, -1]
        self.red = [2, -1, -1, -1, -1]

    def test_only_focus_is_analyzed_without_recommending_or_mutating_the_draft(self):
        before = deepcopy((self.blue, self.red))
        prediction = self.model.predict_match("GOLD", self.blue, self.red)["blue_win_prob"]
        with patch.object(self.model, "recommend_champs", side_effect=AssertionError("Unexpected ranking")):
            records = self.model.explain_prediction("GOLD", self.blue, self.red, "blue", 0)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((self.blue, self.red), before)
        self.assertEqual(record["id"], 0)
        self.assertEqual(record["explanation_context"], "prediction")
        self.assertEqual((record["team"], record["role_index"]), ("blue", 0))
        self.assertIsNone(record["recommendation_rank"])
        self.assertAlmostEqual(record["wr"], prediction)
        self.assertAlmostEqual(record["candidate_delta_pp"], 9)
        effects = {item["champion_id"]: item for item in record["interactions"]}
        self.assertNotIn(0, effects)
        self.assertAlmostEqual(effects[1]["delta_pp"], 4)
        self.assertAlmostEqual(effects[2]["delta_pp"], 2)

    def test_red_focus_uses_red_win_probability_and_empty_focus_baseline(self):
        prediction = self.model.predict_match("GOLD", self.red, self.blue)["red_win_prob"]
        record = self.model.explain_prediction("GOLD", self.red, self.blue, "red", 0)[0]
        self.assertAlmostEqual(record["wr"], prediction)
        self.assertAlmostEqual(record["baseline_win_prob"], 50)
        self.assertEqual(record["team"], "red")
        self.assertTrue(any(item["is_lane_opponent"] for item in record["interactions"]))

    def test_missing_invalid_or_duplicate_focus_stops_before_inference(self):
        for blue, team, role in (([-1] * 5, "blue", 0), (self.blue, "unknown", 0),
                                 (self.blue, "blue", True), (self.blue, "blue", 5),
                                 ([0, 0, -1, -1, -1], "blue", 0)):
            with self.subTest(team=team, role=role), self.assertRaises(ValueError):
                self.model.explain_prediction("GOLD", blue, self.red, team, role)
        self.assertEqual(self.model.calls, [])

    def test_report_names_prediction_focus_and_never_mentions_top_five(self):
        records = self.model.explain_prediction("GOLD", self.blue, self.red, "blue", 0)
        html = explanations_html(records, {0: "<Fókusz>", 1: "Társ", 2: "Ellenfél"})
        self.assertIn("Predikció fókuszhőse", html)
        self.assertIn("&lt;Fókusz&gt;", html)
        self.assertIn("a fókuszhős nélkül", html)
        self.assertNotIn("első öt", html.lower())
        self.assertNotIn("ajánlási lista", html.lower())
        self.assertNotIn("ajánl", html.lower())

    def test_controller_preserves_optional_personal_evidence_for_the_single_focus(self):
        from controller.controller import Controller
        from services import feature_flags
        from test_personal_history import profile, MAPPING
        controller = Controller.__new__(Controller)
        controller.core_model = MagicMock()
        controller.core_model.explain_prediction.return_value = [{"id": 0, "wr": 59, "explanation_context": "prediction"}]
        controller.db = SimpleNamespace(get_champion_mapping=lambda: MAPPING)
        with patch.object(feature_flags, "PERSONAL_HISTORY_ENABLED", True):
            records = controller.explain_prediction("GOLD", self.blue, self.red, "blue", 0, history=profile())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["wr"], 59)
        self.assertIsNotNone(records[0]["personal"])
        controller.core_model.recommend_champs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
