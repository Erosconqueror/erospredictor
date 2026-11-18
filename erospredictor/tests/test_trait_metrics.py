import unittest
from services.trait_metrics import damage_rate, phase_profiles, PHASE_PREFIX
from services.champion_traits import TraitLibrary, extract_observations, team_summary, recommendation_badges, composition_summary, _balanced_tags


class TraitMetricTests(unittest.TestCase):
    def test_balanced_damage_requires_volume_and_complements_across_teammates(self):
        from services.trait_metrics import balanced_damage_score
        def details(p, m, total):
            return {kind: dict(dpm=value, baseline=base, samples=100, estimated=False)
                    for kind, value, base in (("physical", p, 200), ("magic", m, 200), ("overall", total, 500))}
        self.assertIsNone(balanced_damage_score(details(10, 10, 20)))
        self.assertIsNone(balanced_damage_score(details(900, 10, 910)))
        self.assertIsNone(balanced_damage_score({}))
        self.assertGreaterEqual(balanced_damage_score(details(300, 300, 600)), 3.5)
        # Individually pure damage, collectively balanced and sufficiently high.
        profiles = [dict(values={}, damage_details=details(600, 0, 600)),
                    dict(values={}, damage_details=details(0, 600, 600))]
        summary = team_summary(profiles)
        self.assertGreaterEqual(summary["balanced_damage"]["score"], 3.5)
        self.assertEqual(summary["balanced_damage"]["known"], 2)
        partial = [dict(values={}, damage_details={"physical": details(600, 0, 600)["physical"]}),
                   dict(values={}, damage_details={"magic": details(0, 600, 600)["magic"]})]
        self.assertIsNone(team_summary(partial)["balanced_damage"]["score"])

    def test_high_overall_and_balanced_damage_are_positive_only_tags(self):
        rows = []
        for riot, p, m, total in ((1, 400, 400, 900), (2, 50, 50, 100)):
            rows.append(dict(patch="16.7", tier="GOLD", champion=riot, role=0, metrics={
                "damage_per_min": [total*100, 100], "physical_damage_per_min": [p*100, 100],
                "magic_damage_per_min": [m*100, 100]}))
        library = TraitLibrary({}, {"patches": ["16.7"], "rows": rows}, {}, {})
        badges = recommendation_badges(library, "1", 0, "GOLD")
        keys = {tag["key"] for tag in badges}
        self.assertIn("overall_damage", keys)
        self.assertIn("balanced_damage", keys)
        weak = recommendation_badges(library, "2", 0, "GOLD")
        self.assertFalse(any(tag["key"] in ("overall_damage", "balanced_damage") for tag in weak))
        team = composition_summary(library, [("1", 0)], "GOLD")
        self.assertTrue(any(tag["key"] == "overall_damage" for tag in team["tags"]))

    def test_exact_damage_average_is_preferred_to_product_of_averages(self):
        metrics = {"magic_damage_per_min": [3000, 30], "damage_per_min": [30000, 30], "magic_share": [15, 30]}
        self.assertEqual(damage_rate(metrics, "magic"), (100, 30, False))
        del metrics["magic_damage_per_min"]
        self.assertEqual(damage_rate(metrics, "magic"), (500, 30, True))
        self.assertIsNone(damage_rate({}, "magic"))

    def test_same_damage_share_does_not_give_tank_mage_damage_weight(self):
        rows = [dict(patch="16.7", tier="GOLD", champion=riot, role=role,
                     metrics={"magic_damage_per_min": [dpm*100, 100], "physical_damage_per_min": [100, 100]})
                for riot, role, dpm in ((1, 4, 100), (2, 2, 500))]
        library = TraitLibrary({}, {"patches": ["16.7"], "rows": rows}, {}, {})
        tank, mage = library.profile(1, 4, "GOLD"), library.profile(2, 2, "GOLD")
        self.assertLess(tank["values"]["magic_damage"], mage["values"]["magic_damage"])
        summary = team_summary([tank, mage])
        self.assertEqual(summary["damage_rates"]["magic"]["dpm"], 600)
        self.assertEqual(summary["damage_rates"]["magic"]["baseline"], 600)
        self.assertFalse(summary["damage_rates"]["magic"]["estimated"])
        self.assertEqual(composition_summary(library, [("1", 4)], "GOLD")["tags"][0]["key"], "magic_damage")

    def test_duration_boundaries_and_type_rates_are_collected_without_new_calls(self):
        for seconds, phase in ((1379, "early"), (1380, "mid"), (2039, "mid"), (2040, "late")):
            obs = extract_observations(dict(gameDuration=seconds, participants=[dict(championId=1,
                teamPosition="TOP", win=True, magicDamageDealtToChampions=600)]))[0]["metrics"]
            self.assertEqual(obs[PHASE_PREFIX+phase], [1, 1])
            self.assertAlmostEqual(obs["magic_damage_per_min"][0], 600/(seconds/60))
            self.assertEqual(len([key for key in obs if key.startswith(PHASE_PREFIX)]), 1)

    def test_phase_winrate_uses_own_baseline_and_rejects_noise_or_old_bins(self):
        metrics = {PHASE_PREFIX+key: [w, n] for key, w, n in (("early", 700, 1000), ("mid", 500, 1000), ("late", 300, 1000))}
        phases = phase_profiles(metrics)
        self.assertAlmostEqual(phases["early"]["delta"], .2)
        self.assertAlmostEqual(phases["late"]["delta"], -.2)
        self.assertNotIn("mid", phases)
        self.assertEqual(phase_profiles({"observed_win_early": [700, 1000]}), {})
        self.assertEqual(phase_profiles({PHASE_PREFIX+"early": [1, 1]}), {})
        metrics[PHASE_PREFIX+"early"] = [510, 1000]
        metrics[PHASE_PREFIX+"late"] = [500, 1000]
        self.assertEqual(phase_profiles(metrics), {})

    def test_phase_tags_keep_the_sign_of_the_champions_own_winrate_difference(self):
        metrics = {PHASE_PREFIX+key: [w, 1000] for key, w in (("early", 700), ("mid", 500), ("late", 300))}
        library = TraitLibrary({}, {"patches": ["16.7"], "rows": [dict(patch="16.7", champion=1, role=0, tier="GOLD", metrics=metrics)]}, {}, {})
        tags = recommendation_badges(library, "1", 0, "GOLD")
        self.assertEqual(next(tag for tag in tags if tag["key"] == "early")["tone"], "positive")
        self.assertIn("+20.0%", next(tag for tag in tags if tag["key"] == "early")["label"])
        team = composition_summary(library, [("1", 0)], "GOLD")
        self.assertEqual(next(tag for tag in team["tags"] if tag["key"] == "late")["direction"], "low")

    def test_damage_deficits_take_priority_over_large_map_presence_outlier(self):
        tags = _balanced_tags([dict(key="map_presence", priority=100, tone="positive", extreme=True),
                              dict(key="magic_damage", priority=1, tone="negative", extreme=False)])
        self.assertEqual(tags[0]["key"], "magic_damage")
