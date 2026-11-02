"""New floors apply only to the statistics actually usable in this draft."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from services.ensemble_weights import bounded_weights, ensemble_minimums, MODEL_NAMES


class InteractionWeightTests(unittest.TestCase):
    def test_both_policies_hold_after_normalization_with_extreme_raw_weights(self):
        for enhanced in (False, True):
            for strong in (False, True):
                floors = ensemble_minimums(enhanced, strong)
                self.assertEqual(floors["statistical"], .30 if strong else .16)
                for dominant in MODEL_NAMES:
                    weights = bounded_weights({name: int(name == dominant) for name in MODEL_NAMES}, minimums=floors)
                    self.assertAlmostEqual(sum(weights.values()), 1)
                    for name in MODEL_NAMES:
                        self.assertGreaterEqual(weights[name], floors[name])

    def test_absent_models_are_never_added_and_infeasible_floor_maps_fail(self):
        self.assertEqual(bounded_weights({}, ["gnn"], minimums=ensemble_minimums()), {"gnn": 1})
        self.assertNotIn("statistical", bounded_weights({}, ["gnn", "roleaware"], minimums=ensemble_minimums(True)))
        with self.assertRaises(ValueError):
            bounded_weights({"gnn": 1, "statistical": 1}, minimums={"gnn": .9, "statistical": .9})

    def test_runtime_switches_policy_when_new_statistics_are_actually_used(self):
        import torch
        from model.core_model import CoreModel
        blue, red = [0, -1, -1, -1, -1], [1, -1, -1, -1, -1]
        for enhanced in (False, True):
            stat = SimpleNamespace(predict=lambda *args: .4, uses_interactions=lambda *args: enhanced)
            core = CoreModel(stat)
            core.ensemble_weights = {"statistical": 1, "roleweighted": 0, "gnn": 0}
            core.models = {"gnn": object(), "roleweighted": lambda inputs: torch.tensor(.8)}
            with patch.object(core, "load_models"), patch("model.gnn_predictor.predict_gnn", return_value=.6):
                components = core._component_predictions("GOLD", blue, red)
                details = core._prediction_components("GOLD", blue, red, "blue")
            floors = ensemble_minimums(enhanced)
            for result in (components, details):
                self.assertGreaterEqual(next(item["weight"] for item in result if item["model"] == "gnn"), floors["gnn"])
                self.assertGreaterEqual(next(item["weight"] for item in result if item["model"] == "statistical"), floors["statistical"])
                self.assertAlmostEqual(sum(item["weight"] for item in result), 1)

    def test_calibration_detects_the_statistical_artifact(self):
        from model.meta_calibrator import MetaLearningCalibrator
        dataset = SimpleNamespace(validate_predictions=lambda *args: {"passed": 0})
        for enhanced in (False, True):
            calibrator = MetaLearningCalibrator(dataset)
            models = {name: object() for name in MODEL_NAMES}
            models["statistical"] = SimpleNamespace(has_interactions=lambda div: enhanced)
            with patch.object(calibrator, "_wrap_model_prediction"):
                weights = calibrator.calibrate(models, "GOLD")
            for name, floor in ensemble_minimums(enhanced).items():
                self.assertGreaterEqual(weights[name], floor)

    def test_calibration_roundtrip_preserves_the_selected_policy(self):
        import tempfile
        from pathlib import Path
        from model.meta_calibrator import MetaLearningCalibrator
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "calibration.pkl")
            before = MetaLearningCalibrator()
            before.enhanced_statistical = True
            before.weights = {"gnn": .16, "statistical": .30, "roleaware": .27, "roleweighted": .27}
            before.save(path)
            after = MetaLearningCalibrator()
            result = after.load(path)
        self.assertTrue(after.enhanced_statistical)
        self.assertAlmostEqual(result["gnn"], .16)
        self.assertAlmostEqual(result["statistical"], .30)

    def test_old_calibration_and_defaults_have_sixteen_percent_base_floor_without_training(self):
        import pickle
        import tempfile
        from pathlib import Path
        from model.meta_calibrator import MetaLearningCalibrator
        calibrator = MetaLearningCalibrator()
        self.assertGreaterEqual(calibrator.get_weights()["statistical"], .16)
        self.assertGreaterEqual(calibrator.calibrate({name: object() for name in MODEL_NAMES}, "GOLD")["statistical"], .16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.pkl"
            with path.open("wb") as stream:
                pickle.dump({"weights": {"statistical": .08, "gnn": .76, "roleaware": .08, "roleweighted": .08},
                             "enhanced_statistical": False}, stream)
            self.assertAlmostEqual(calibrator.load(path)["statistical"], .16)

    def test_draft_gate_applies_before_and_after_side_balance(self):
        import torch
        from model.core_model import CoreModel
        blue, red = [0, -1, -1, -1, -1], [1, -1, -1, -1, -1]
        for enhanced in (False, True):
            for strong in (False, True):
                stat = SimpleNamespace(predict=lambda *args: .4, uses_interactions=lambda *args: enhanced,
                                       has_strong_evidence=lambda *args: strong)
                core = CoreModel(stat)
                core.ensemble_weights = {"roleweighted": 1, "statistical": 0}
                core.models = {"roleweighted": lambda inputs: torch.tensor(.8)}
                with patch.object(core, "load_models"):
                    for details in (core._component_predictions("GOLD", blue, red),
                                    core._prediction_components("GOLD", blue, red, "blue"),
                                    core._prediction_components("GOLD", blue, red, "red")):
                        weight = next(row["weight"] for row in details if row["model"] == "statistical")
                        self.assertAlmostEqual(weight, .30 if strong else .16)


if __name__ == "__main__":
    unittest.main()
