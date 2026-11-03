"""Counter exposure removes unavailable picks and keeps unknown evidence visible."""

import unittest

from services.blind_pick import blind_risk, annotate_blind_risk, risk_badge, risk_badges


class Index:
    def __init__(self):
        self.rates = {0: {1: .2, 2: .3, 3: .5}, 1: {1: .5, 3: .5}}
        self.edges = {(0, 1): -8, (0, 2): 0, (0, 3): 2, (1, 1): -8, (1, 3): 0}

    def eligible(self, div, champion, role):
        return champion == 0 and role == 2

    def baseline(self, *args):
        return {"winrate": 50, "matches": 1000}

    def role_pick_rates(self, div, role):
        return self.rates.get(role, {})

    def prior(self, div):
        return {"counter_weights": [[1.0] * 5 for _ in range(5)]}

    def counter(self, div, champion, role, foe, foe_role):
        delta = self.edges.get((foe_role, foe))
        return None if delta is None else {"adjusted_delta_pp": delta, "noise_corrected_delta_pp": delta,
                                           "matches": 100}


class BlindRiskTests(unittest.TestCase):
    def setUp(self):
        self.index = Index()

    def test_pick_rate_mass_of_supported_counters_is_used(self):
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 11, 12, 13])
        self.assertAlmostEqual(result["risk_percent"], 20)
        self.assertEqual(result["coverage_percent"], 100)
        self.assertEqual(result["status"], "estimated")

    def test_removing_bans_and_ally_picks_renormalizes_remaining_pool(self):
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 11, 12, 13], [2])
        self.assertAlmostEqual(result["risk_percent"], 100 * .2 / .7)
        none = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 11, 12, 13], [1])
        self.assertEqual(none["risk_percent"], 0)

    def test_unknown_pair_mass_is_not_renormalized_away_or_presented_as_safe(self):
        self.index.edges = {(0, 1): -8}
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 11, 12, 13])
        self.assertAlmostEqual(result["risk_percent"], 20)
        self.assertAlmostEqual(result["coverage_percent"], 20)
        self.assertAlmostEqual(result["upper_percent"], 100)
        self.assertEqual(result["status"], "insufficient")
        caption, color = risk_badge(result)
        self.assertIn("kevés adat", caption)
        self.assertEqual(color, "#9E9EAE")

    def test_multiple_roles_use_explicit_independence_approximation(self):
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, -1, 11, 12, 13])
        self.assertAlmostEqual(result["risk_percent"], 60)  # 1 - (1-.2)*(1-.5).

    def test_sparse_cross_role_data_falls_back_to_supported_same_tier_lane(self):
        self.index.rates[2] = {1: .2, 2: .8}
        self.index.edges.update({(2, 1): -8, (2, 2): 0})
        result = blind_risk(self.index, "GOLD", 0, 2, [-1] * 5)
        self.assertEqual(result["scope"], "lane")
        self.assertEqual(result["status"], "estimated")
        self.assertAlmostEqual(result["risk_percent"], 20)
        self.assertEqual(result["lane"]["coverage_percent"], 100)
        self.assertEqual(result["others"]["status"], "insufficient")
        badges = risk_badges(result)
        self.assertEqual(badges[0][0], "Lane ~20%")
        self.assertIn("csak lane", badges[1][0])

    def test_lane_and_other_roles_have_separate_risks_and_colors(self):
        self.index.rates[2] = {1: .2, 2: .8}
        self.index.edges.update({(2, 1): -8, (2, 2): 0})
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, -1, -1, 12, 13])
        self.assertEqual(result["scope"], "all")
        self.assertAlmostEqual(result["lane"]["risk_percent"], 20)
        self.assertAlmostEqual(result["others"]["risk_percent"], 60)
        self.assertAlmostEqual(result["risk_percent"], 68)
        badges = risk_badges(result)
        self.assertEqual([item[0] for item in badges], ["Lane ~20%", "Más role ~60%"])
        self.assertNotEqual(badges[0][1], badges[1][1])

    def test_lane_fallback_does_not_hide_its_own_missing_data_or_use_other_ranks(self):
        self.index.rates[2] = {1: .2, 2: .8}
        self.index.edges[(2, 1)] = -8
        original = self.index.counter
        tiers = []
        def counter(tier, *args):
            tiers.append(tier)
            return original(tier, *args)
        self.index.counter = counter
        result = blind_risk(self.index, "IRON", 0, 2, [-1] * 5)
        self.assertEqual(result["lane"]["status"], "insufficient")
        self.assertNotEqual(result["scope"], "lane")
        self.assertEqual(set(tiers), {"IRON"})
        self.assertIn("Lane: kevés adat", [caption for caption, _ in risk_badges(result)])

    def test_already_picked_laner_does_not_warn_about_visible_counter(self):
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 1, 12, 13])
        self.assertEqual(result["lane"]["status"], "resolved")
        self.assertTrue(all(not caption.startswith("Lane") for caption, _ in risk_badges(result)))

    def test_partial_coverage_never_turns_zero_known_counter_mass_into_safe_green(self):
        self.index.edges = {(0, 2): 0, (0, 3): 0}
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 11, 12, 13])
        self.assertEqual(result["status"], "estimated")
        self.assertAlmostEqual(result["risk_percent"], 0)
        self.assertAlmostEqual(result["upper_percent"], 20)
        caption, color = risk_badge(result)
        self.assertEqual(caption, "! Blind ~0–20%")
        self.assertNotEqual(color, "#66BB6A")

    def test_known_counter_does_not_warn_when_no_future_pick_remains(self):
        result = blind_risk(self.index, "GOLD", 0, 2, [1, 10, 11, 12, 13])
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["risk_percent"], 0)
        self.assertIsNone(result["coverage_percent"])
        self.assertEqual(len(result["known_counters"]), 1)
        self.assertEqual(risk_badge(result), ("", None))

    def test_safe_blind_has_no_caption_and_no_color(self):
        for risk in (0, 2.5, 9.9):
            with self.subTest(risk=risk):
                self.assertEqual(risk_badge({"status": "estimated", "risk_percent": risk, "upper_percent": risk}), ("", None))

    def test_known_counters_never_change_the_remaining_blind_warning(self):
        future = {"status": "estimated", "risk_percent": 20, "upper_percent": 20}
        self.assertEqual(risk_badge(dict(future, known_counters=[{"id": 1}])), risk_badge(future))

    def test_warning_color_changes_gradually_across_risk_levels(self):
        risks = (10, 15, 20, 25, 35, 45, 55, 65, 75, 85, 95, 100)
        colors = [risk_badge({"status": "estimated", "risk_percent": risk, "upper_percent": risk})[1] for risk in risks]
        self.assertEqual(len(set(colors)), len(risks))
        greens = [int(color[3:5], 16) for color in colors]
        self.assertEqual(greens, sorted(greens, reverse=True))

    def test_noise_corrected_effect_and_role_relevance_determine_counter_threshold(self):
        original = self.index.counter
        self.index.counter = lambda *args: (dict(original(*args), noise_corrected_delta_pp=-1) if original(*args) else None)
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 11, 12, 13])
        self.assertEqual(result["risk_percent"], 0)
        weights = [[1.0] * 5 for _ in range(5)]
        weights[2][0] = 4
        self.index.prior = lambda div: {"counter_weights": weights}
        result = blind_risk(self.index, "GOLD", 0, 2, [-1, 10, 11, 12, 13])
        self.assertAlmostEqual(result["risk_percent"], 100)

    def test_off_meta_missing_artifact_and_missing_prior_never_look_safe(self):
        self.assertEqual(blind_risk(None, "GOLD", 0, 2, [-1] * 5)["status"], "unavailable")
        self.assertEqual(blind_risk(self.index, "GOLD", 0, 0, [-1] * 5)["status"], "off_meta")
        self.index.prior = lambda div: None
        self.index.edges = {}
        result = blind_risk(self.index, "GOLD", 0, 2, [-1] * 5)
        self.assertEqual(result["coverage_percent"], 0)
        self.assertIn("kevés adat", risk_badge(result)[0])

    def test_annotation_preserves_order_winrate_and_inputs_and_removes_both_teams_picks(self):
        recs = [{"id": 0, "wr": 55.5}]
        blue, red = [2, -1, -1, -1, -1], [-1, 10, 11, 12, 13]
        annotated = annotate_blind_risk(recs, self.index, "GOLD", blue, red, [], "blue", 2)
        self.assertEqual(recs, [{"id": 0, "wr": 55.5}])
        self.assertEqual(annotated[0]["wr"], 55.5)
        self.assertAlmostEqual(annotated[0]["blind"]["risk_percent"], 100 * .2 / .7)
        self.assertEqual(blue, [2, -1, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
