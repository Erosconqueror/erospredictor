"""Offline tests for the optional diagnostic stage and real cached statistics."""
import copy
import importlib.util
import unittest
from unittest.mock import patch

from model.core_model import CoreModel, torch as torch_runtime
from model.statistical import StatisticalModel


class MemoryDB:
    def get_champion_mapping(self):
        return {}


class InteractionModel(CoreModel):
    """Small deterministic fixture with known ally/opponent interactions."""

    def __init__(self):
        super().__init__(StatisticalModel(MemoryDB()))
        self.calls = []

    def _component_predictions(self, div, blue, red):
        self.calls.append((tuple(blue), tuple(red)))

        def strength(own, other):
            if 0 not in own:
                return 0.0
            return 0.03 + (0.04 if 1 in own else 0) + (0.02 if 2 in other else 0)

        probability = 0.5 + strength(blue, red) - strength(red, blue)
        return [{"model": "fixture", "probability": probability, "weight": 1.0}]


class TestRecommendationExplanations(unittest.TestCase):
    def setUp(self):
        self.model = InteractionModel()
        self.blue = [-1, 1, -1, -1, -1]
        self.red = [2, -1, -1, -1, -1]

    def test_reports_candidate_specific_ally_and_opponent_sensitivity(self):
        draft = copy.deepcopy((self.blue, self.red))
        result = self.model.explain_recommendations(
            "GOLD", self.blue, self.red, "blue", 0, [{"id": 0, "wr": 59.0}])[0]
        self.assertEqual((self.blue, self.red), draft)
        self.assertAlmostEqual(result["wr"], 59.0)
        self.assertAlmostEqual(result["candidate_delta_pp"], 9.0)
        effects = {item["champion_id"]: item for item in result["interactions"]}
        self.assertAlmostEqual(effects[1]["delta_pp"], 4.0)
        self.assertAlmostEqual(effects[2]["delta_pp"], 2.0)
        self.assertEqual(effects[1]["side"], "ally")
        self.assertTrue(effects[2]["is_lane_opponent"])
        self.assertIsNone(result["lane_matchup"])
        self.assertEqual(result["explanation_kind"], "conditional_leave_one_out")
        self.assertAlmostEqual(sum(c["weighted_win_prob"] for c in result["components"]), result["wr"])

    def test_red_side_uses_own_teams_probability(self):
        result = self.model.explain_recommendations(
            "GOLD", self.red, self.blue, "red", 0, [{"id": 0}])[0]
        self.assertAlmostEqual(result["wr"], 59.0)
        self.assertAlmostEqual(result["candidate_delta_pp"], 9.0)
        self.assertTrue(any(item["champion_id"] == 2 and item["side"] == "opponent"
                            for item in result["interactions"]))

    def test_never_explains_more_than_five_and_reuses_baseline_evaluations(self):
        candidates = [{"id": cid} for cid in (0, 3, 4, 5, 6, 7, 8)]
        result = self.model.explain_recommendations(
            "GOLD", self.blue, self.red, "blue", 0, candidates, max_items=99)
        self.assertEqual([item["id"] for item in result], [0, 3, 4, 5, 6])
        base = (tuple(self.blue), tuple(self.red))
        self.assertEqual(self.model.calls.count(base), 1)

    def test_normal_recommendations_never_start_diagnostics(self):
        self.model.meta_data = {"GOLD": {"strict": {"0": [0, 3]}}}
        with patch.object(self.model, "explain_recommendations", side_effect=AssertionError("extra inference")):
            recommendations = self.model.recommend_champs(
                "GOLD", self.blue, self.red, [], "blue", 0, top_k=2)
        self.assertEqual([item["id"] for item in recommendations], [0, 3])
        self.assertEqual(len(self.model.calls), 4)

    def test_selected_outside_top_five_is_added_first_with_ranked_five_intact(self):
        candidates = [{"id": cid} for cid in (0, 3, 4, 5, 6, 7, 8)]
        result = self.model.explain_recommendations(
            "GOLD", self.blue, self.red, "blue", 0, candidates, selected_id=8)
        self.assertEqual([item["id"] for item in result], [8, 0, 3, 4, 5, 6])
        self.assertTrue(result[0]["is_selected"])
        self.assertIsNone(result[0]["recommendation_rank"])
        self.assertEqual([item["recommendation_rank"] for item in result[1:]], [1, 2, 3, 4, 5])
        self.assertEqual(sum(item["is_selected"] for item in result), 1)
        self.assertEqual(self.model.calls.count((tuple(self.blue), tuple(self.red))), 1)

    def test_selected_inside_top_five_is_not_analyzed_twice(self):
        candidates = [{"id": cid} for cid in (0, 3, 4, 5, 6)]
        result = self.model.explain_recommendations(
            "GOLD", self.blue, self.red, "blue", 0, candidates, selected_id=4)
        self.assertEqual([item["id"] for item in result], [4, 0, 3, 5, 6])
        self.assertEqual(result[0]["recommendation_rank"], 3)
        self.assertEqual(sum(item["is_selected"] for item in result), 1)

    def test_invalid_selection_stops_before_expensive_explanation(self):
        with self.assertRaises(ValueError):
            self.model.explain_recommendations(
                "GOLD", self.blue, self.red, "blue", 0, [{"id": 0}], selected_id=2)
        self.assertEqual(self.model.calls, [])

    def test_empty_meta_list_does_not_disable_filter(self):
        self.model.meta_data = {"GOLD": {"strict": {"0": []}}}
        self.assertEqual(self.model.recommend_champs("GOLD", self.blue, self.red, [], "blue", 0), [])
        self.assertEqual(self.model.calls, [])

    def test_rejects_stale_or_invalid_request_before_inference(self):
        with self.assertRaises(ValueError):
            self.model.explain_recommendations("GOLD", [0, 1, -1, -1, -1], self.red, "blue", 0, [{"id": 3}])
        with self.assertRaises(ValueError):
            self.model.recommend_champs("GOLD", self.blue, self.red, [], "sideways", 0)
        with self.assertRaises(ValueError):
            self.model.recommend_champs("GOLD", self.blue, self.red, [], "blue", 8)
        with self.assertRaises(ValueError):
            self.model.explain_recommendations("GOLD", self.blue, self.red, "blue", 0, [{"id": 2}])

    def test_zero_limit_or_no_candidates_performs_no_inference(self):
        self.assertEqual(self.model.explain_recommendations("GOLD", self.blue, self.red, "blue", 0, []), [])
        self.assertEqual(self.model.explain_recommendations("GOLD", self.blue, self.red, "blue", 0,
                                                          [{"id": 0}], max_items=0), [])
        self.assertEqual(self.model.calls, [])

    def test_component_breakdown_matches_prediction_when_one_side_loses_a_model(self):
        direct = [{"model": "gnn", "probability": 0.7, "weight": 0.75},
                  {"model": "statistical", "probability": 0.4, "weight": 0.25}]
        swapped = [{"model": "statistical", "probability": 0.2, "weight": 1.0}]
        with patch.object(self.model, "_component_predictions",
                          side_effect=lambda div, blue, red: direct if blue[0] == 0 else swapped):
            blue = [0, -1, -1, -1, -1]
            red = [2, -1, -1, -1, -1]
            prediction = self.model.predict_match("GOLD", blue, red)
            for side in ("blue", "red"):
                parts = self.model._prediction_components("GOLD", blue, red, side)
                self.assertAlmostEqual(sum(c["weight"] for c in parts), 1.0)
                self.assertAlmostEqual(sum(c["weighted_win_prob"] for c in parts), prediction[f"{side}_win_prob"])


class TestHistoricalEvidence(unittest.TestCase):
    def setUp(self):
        self.stats = StatisticalModel(MemoryDB())
        self.stats.matchups = {
            "GOLD": {
                "TOP": {"0": {"2": {"w": 6, "m": 10}}, "2": {"0": {"w": 2, "m": 10}}},
                "BOT_SYNERGY": {"3": {"4": {"w": 7, "m": 10}}},
            },
            "IRON": {"TOP": {"0": {"2": {"w": 0, "m": 10}}}},
        }

    def test_lane_observations_include_both_sides_and_real_sample_count(self):
        evidence = self.stats.get_lane_matchup("GOLD", 0, 2, 0)
        self.assertEqual(evidence["matches"], 20)
        self.assertEqual(evidence["wins"], 14)
        self.assertAlmostEqual(evidence["winrate"], 70.0)
        self.assertEqual(evidence["kind"], "historical_lane_matchup")
        self.assertIsNone(self.stats.get_lane_matchup("GOLD", -1, 2, 0))
        self.assertIsNone(self.stats.get_lane_matchup("GOLD", 0, 2, 2))

    def test_mixed_keeps_raw_sample_count_separate_from_rank_weighting(self):
        evidence = self.stats.get_lane_matchup("MIXED", 0, 2, 0)
        self.assertEqual(evidence["matches"], 30)
        self.assertEqual(evidence["wins"], 14)
        self.assertAlmostEqual(evidence["winrate"], 14 / 30 * 100)
        self.assertAlmostEqual(evidence["weighted_winrate"], 14 * 0.75 / (20 * 0.75 + 10 * 0.3) * 100)

    def test_partial_update_scope_keeps_mixed_inference_and_evidence_consistent(self):
        self.stats.matchups["__scope__"] = {"mixed_tiers": ["GOLD"], "patches": ["16.7"]}
        blue, red = [0, -1, -1, -1, -1], [2, -1, -1, -1, -1]
        self.assertAlmostEqual(self.stats.predict("MIXED", blue, red), self.stats.predict("GOLD", blue, red))
        evidence = self.stats.get_lane_matchup("MIXED", 0, 2, 0)
        self.assertEqual(evidence["matches"], 20)
        self.assertEqual(evidence["tiers"], ["GOLD"])
        self.assertAlmostEqual(evidence["weighted_winrate"], 70.0)
        self.stats.matchups["IRON"]["BOT_SYNERGY"] = {"3": {"4": {"w": 0, "m": 100}}}
        synergy = self.stats.get_bot_synergy("MIXED", 3, 4)
        self.assertEqual(synergy["matches"], 10)
        self.assertAlmostEqual(synergy["winrate"], 70.0)

    def test_bot_synergy_returns_counts_or_no_evidence(self):
        evidence = self.stats.get_bot_synergy("GOLD", 3, 4)
        self.assertEqual(evidence["matches"], 10)
        self.assertAlmostEqual(evidence["winrate"], 70.0)
        self.assertIsNone(self.stats.get_bot_synergy("GOLD", -1, 4))
        self.assertIsNone(self.stats.get_bot_synergy("GOLD", 3, 9))

    def test_stats_only_inference_keeps_aatrox_zero_and_renormalizes(self):
        core = CoreModel(self.stats)
        core.active_div = "GOLD"
        core.ensemble_weights = {"gnn": 0.7, "statistical": 0.3}
        self.assertAlmostEqual(core.calc_win_prob_fast("GOLD", [0, -1, -1, -1, -1], [2, -1, -1, -1, -1]), 0.6)
        self.assertAlmostEqual(core.calc_win_prob_fast("GOLD", [-1] * 5, [-1] * 5), 0.5)
        parts = core._component_predictions("GOLD", [0, -1, -1, -1, -1], [2, -1, -1, -1, -1])
        self.assertEqual(parts[0]["model"], "statistical")
        self.assertEqual(parts[0]["weight"], 1.0)


@unittest.skipUnless(importlib.util.find_spec("psycopg2") is not None,
                     "PostgreSQL adapter is optional for the pure model tests")
class TestControllerExplanationIntegration(unittest.TestCase):
    def test_controller_uses_cached_statistics_and_validates_internal_zero(self):
        from controller.controller import Controller

        class CachedDB(MemoryDB):
            def load_stats_cache(self, path):
                return {"GOLD": {"TOP": {"0": {"2": {"w": 6, "m": 10}},
                                           "2": {"0": {"w": 2, "m": 10}}}}}

            def load_meta_champs(self):
                return {}

        with patch("controller.controller.DataManager", return_value=CachedDB()):
            controller = Controller()
        controller.core_model.active_div = "GOLD"
        blue, red = [0, -1, -1, -1, -1], [2, -1, -1, -1, -1]
        self.assertAlmostEqual(controller.predict_match("GOLD", blue, red)["blue_win_prob"], 68.4)
        self.assertTrue(controller.validate_draft(blue, red, [-1] * 5, [-1] * 5)[0])
        self.assertFalse(controller.validate_draft(blue, red, [0, -1, -1, -1, -1], [-1] * 5)[0])
        self.assertFalse(controller.validate_draft(blue, [0, -1, -1, -1, -1], [], [])[0])


@unittest.skipUnless(torch_runtime is not None, "PyTorch is optional for offline tests")
class TestNeuralEmptyChampionEncoding(unittest.TestCase):
    def test_core_encoders_keep_aatrox_zero_distinct_from_empty(self):
        torch = torch_runtime

        class Capture:
            def __call__(self, inputs):
                self.inputs = inputs.detach().clone()
                return torch.tensor([[0.5]], device=inputs.device)

        class NoStats:
            def predict(self, *args):
                return None

        core = CoreModel(NoStats())
        core.device = torch.device("cpu")
        core.active_div = "GOLD"
        blue, red = [-1, 0, -1, -1, -1], [-1] * 5
        for model_name, feature_offset in (("roleweighted", 0), ("roleaware", 172)):
            capture = Capture()
            core.models = {model_name: capture}
            core._component_predictions("GOLD", blue, red)
            self.assertNotEqual(capture.inputs[0, feature_offset].item(), 0)
            self.assertEqual(torch.count_nonzero(capture.inputs).item(), 1)

    def test_role_aware_unknown_position_has_no_aatrox_embedding(self):
        torch = torch_runtime
        from model.predictor import RoleAwareEmbeddingPredictor
        model = RoleAwareEmbeddingPredictor().eval()
        with torch.no_grad():
            model.c_emb.weight[0].fill_(2.0)
            model.r_emb.weight.zero_()
        captured = []
        hook = model.fc1.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            features = torch.zeros((1, 1720))
            with torch.no_grad():
                model(features)
                features[0, 0] = 1.0
                model(features)
        finally:
            hook.remove()
        self.assertEqual(torch.count_nonzero(captured[0]).item(), 0)
        self.assertTrue(torch.all(captured[1][0, :64] == 2.0))
        self.assertEqual(torch.count_nonzero(captured[1][0, 64:]).item(), 0)

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric") is not None,
                         "torch-geometric is optional for offline tests")
    def test_gnn_unknown_position_has_no_aatrox_embedding(self):
        torch = torch_runtime
        from model.gnn_predictor import LeagueGNN, create_graph
        model = LeagueGNN().eval()
        with torch.no_grad():
            model.champ_emb.weight[0].fill_(2.0)
        captured = []
        hook = model.conv1.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            for blue in ([-1] * 5, [0, -1, -1, -1, -1]):
                graph = create_graph(blue, [-1] * 5)
                with torch.no_grad():
                    model(graph.x, graph.edge_index, torch.zeros(10, dtype=torch.long))
        finally:
            hook.remove()
        self.assertEqual(torch.count_nonzero(captured[0][:, :32]).item(), 0)
        self.assertTrue(torch.all(captured[1][0, :32] == 2.0))
        self.assertEqual(torch.count_nonzero(captured[1][1:, :32]).item(), 0)


if __name__ == "__main__":
    unittest.main()
