"""Weight floors and graded recommendation QA without changing prediction QA."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from services.ensemble_weights import bounded_weights, MODEL_NAMES
from model.golden_dataset import GoldenDataset


class WeightFloorTests(unittest.TestCase):
    def test_minimum_survives_normalization_and_preserves_total(self):
        for raw in ({name: (100 if name == "gnn" else 0) for name in MODEL_NAMES},
                    {name: 0 for name in MODEL_NAMES}, {"gnn": 1, "statistical": .001}):
            weights = bounded_weights(raw)
            self.assertAlmostEqual(sum(weights.values()), 1)
            self.assertTrue(all(value >= .08 for value in weights.values()))
        self.assertEqual(bounded_weights({"gnn": 1, "statistical": 0}), {"gnn": .92, "statistical": .08})

    def test_unavailable_models_are_absent_and_unspecified_available_models_get_the_floor(self):
        result = bounded_weights({"gnn": 1}, ["gnn", "roleaware"])
        self.assertEqual(result, {"gnn": .92, "roleaware": .08})
        self.assertEqual(bounded_weights({"gnn": 1}, ["statistical"]), {"statistical": 1})
        self.assertEqual(bounded_weights({}, []), {})

    def test_runtime_uses_floor_for_previously_zero_weight_model(self):
        import torch
        from model.core_model import CoreModel
        core = CoreModel(SimpleNamespace(predict=lambda *args: .4))
        core.models = {"roleweighted": lambda inputs: torch.tensor(.8)}
        core.ensemble_weights = {"statistical": 1, "roleweighted": 0}
        with patch.object(core, "load_models"):
            components = core._component_predictions("GOLD", [-1] * 5, [-1] * 5)
        self.assertEqual({item["model"]: item["weight"] for item in components}, {"roleweighted": .08, "statistical": .92})

    def test_calibration_publishes_at_least_eight_percent_for_all_present_models(self):
        from model.meta_calibrator import MetaLearningCalibrator
        calibrator = MetaLearningCalibrator(SimpleNamespace(validate_predictions=lambda fn, div: {"passed": fn()}))
        with patch.object(calibrator, "_wrap_model_prediction", side_effect=lambda name, model: lambda: 100 if name == "gnn" else 0):
            weights = calibrator.calibrate({name: object() for name in MODEL_NAMES}, "GOLD")
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertTrue(all(value >= .08 for value in weights.values()))

    def test_floor_also_holds_after_side_balance_when_one_orientation_loses_a_model(self):
        from model.core_model import CoreModel
        core = CoreModel(SimpleNamespace())
        direct = [{"model": "gnn", "probability": .7, "weight": .08},
                  {"model": "statistical", "probability": .4, "weight": .92}]
        swapped = [{"model": "statistical", "probability": .2, "weight": 1.0}]
        blue, red = [0, -1, -1, -1, -1], [1, -1, -1, -1, -1]
        with patch.object(core, "_component_predictions", side_effect=lambda div, b, r: direct if b[0] == 0 else swapped):
            result = core.predict_match("GOLD", blue, red)
            details = core._prediction_components("GOLD", blue, red, "blue")
        self.assertEqual({item["model"]: item["weight"] for item in details}, {"gnn": .24, "statistical": .76})
        self.assertAlmostEqual(sum(item["weighted_win_prob"] for item in details), result["blue_win_prob"])
        self.assertAlmostEqual(result["blue_win_prob"] + result["red_win_prob"], 100)


class GradedRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = GoldenDataset.__new__(GoldenDataset)
        self.dataset.name_map = {f"Champion{i}": i for i in range(30)}
        self.dataset.prediction_cases = []
        self.dataset.recommendation_cases = [{"blue_team": [""] * 5, "red_team": [""] * 5,
            "position": 0, "must_pick_from": ["Champion7"], "avoid_picks": ["Champion25"]}]

    def validate(self):
        return self.dataset.validate_recommendations(lambda **kwargs: [{"id": i} for i in range(30)], "GOLD")

    def test_must_pick_at_rank_eight_gets_partial_credit_instead_of_zero(self):
        result = self.validate()
        self.assertEqual(result["score"], .875)
        self.assertEqual(result["passed"], 1)

    def test_avoid_pick_in_top_three_still_has_a_real_penalty(self):
        case = self.dataset.recommendation_cases[0]
        case["must_pick_from"] = ["Champion0"]
        case["avoid_picks"] = ["Champion1"]
        result = self.validate()
        self.assertEqual(result["score"], .5)
        self.assertEqual(result["passed"], 0)

    def test_prediction_acceptance_ranges_are_unchanged(self):
        self.dataset.prediction_cases = [{"blue_team": [""] * 5, "red_team": [""] * 5,
            "expected_blue_winrate_min": 55, "expected_blue_winrate_max": 60}]
        for value, expected in ((54.99, 0), (55, 1), (60, 1), (60.01, 0)):
            result = self.dataset.validate_predictions(lambda **kwargs: {"blue_win_prob": value}, "GOLD")
            self.assertEqual(result["passed"], expected)

    def test_training_qa_uses_fractional_recommendation_score(self):
        from trainer_script import quick_evaluate_model
        dataset = SimpleNamespace(validate_predictions=lambda *args: {"passed": 1, "total": 2},
            validate_recommendations=lambda *args: {"passed": 0, "score": .75, "total": 1})
        self.assertAlmostEqual(quick_evaluate_model("unused", None, "GOLD", dataset, "cpu"), 1.75 / 3)


if __name__ == "__main__":
    unittest.main()
