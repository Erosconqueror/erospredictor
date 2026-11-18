"""Descriptive observations, patch pooling, filters and winrate isolation."""

import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
from services.champion_traits import (extract_observations, TraitLibrary, team_summary,
                                      export_observations, merge_metrics, fit_notes)


def sample():
    return {"gameDuration": 1200, "participants": [{"championId": 266, "teamPosition": "TOP",
        "timePlayed": 1200, "timeCCingOthers": 40, "totalDamageDealtToChampions": 10000,
        "physicalDamageDealtToChampions": 8000, "magicDamageDealtToChampions": 0,
        "trueDamageDealtToChampions": 2000, "spell1Casts": 60,
        "challenges": {"enemyChampionImmobilizations": 20}, "win": True}]}


class TraitTests(unittest.TestCase):
    def test_map_presence_is_separate_from_combat_mobility_and_editable(self):
        basics = {"champions": {"29": {"name": "Twitch", "tags": ["Assassin"]},
                                 "55": {"name": "Katarina", "tags": ["Assassin"]},
                                 "30": {"name": "Karthus", "tags": ["Mage"]}}}
        library = TraitLibrary(basics, {}, {})
        twitch = library.profile("29", 3)["values"]
        self.assertEqual((twitch["mobility"], twitch["map_presence"]), (2, 4))
        self.assertEqual(library.profile("55", 2)["values"]["mobility"], 5)
        self.assertNotIn("map_presence", library.profile("55", 2)["values"])
        self.assertEqual(library.profile("30", 2)["values"]["map_presence"], 4)
        library.overrides = {"29": {"3": {"mobility": 3, "map_presence": None}}}
        self.assertEqual(library.profile("29", 3)["values"]["mobility"], 3)
        self.assertNotIn("map_presence", library.profile("29", 3)["values"])
        self.assertAlmostEqual(team_summary([{"values": {"map_presence": 5}},
            {"values": {"map_presence": 3}}, {"values": {}}])["map_presence"]["score"], 4.3)

    def sustain_library(self, heal=(9000, 30), shield=(4500, 30)):
        metrics = {key: list(value) for key, value in (
            ("ally_heal_per_min", heal), ("ally_shield_per_min", shield)) if value is not None}
        metrics["healing_per_min"] = [999999, 30]  # Self-healing must not contribute.
        return TraitLibrary({}, {"patches": ["16.7"], "rows": [dict(patch="16.7", tier="GOLD",
            champion=266, role=4, metrics=metrics)]}, {})

    def test_sustain_uses_only_ally_healing_and_shielding_and_preserves_scope(self):
        library = self.sustain_library()
        profile = library.profile(266, 4, "GOLD")
        self.assertEqual(profile["values"]["team_sustain"], 4)
        self.assertIn("n=30", profile["sources"]["team_sustain"])
        self.assertNotIn("team_sustain", library.profile(266, 0, "GOLD")["values"])
        self.assertNotIn("team_sustain", library.profile(266, 4, "SILVER")["values"])
        self.assertEqual(library.profile(266, 4, "MIXED")["values"]["team_sustain"], 4)
        self.assertTrue(library.accepts(266, 4, "GOLD", ["team_sustain"], minimum=4))

    def test_sustain_needs_both_sample_counts_and_distinguishes_zero_from_missing(self):
        for heal, shield in (((0, 29), (0, 30)), ((0, 30), None), (None, (0, 30))):
            self.assertNotIn("team_sustain", self.sustain_library(heal, shield).profile(266, 4, "GOLD")["values"])
        self.assertEqual(self.sustain_library((0, 30), (0, 30)).profile(266, 4, "GOLD")["values"]["team_sustain"], 1)
        self.assertEqual(self.sustain_library((90000, 30), (0, 30)).profile(266, 4, "GOLD")["values"]["team_sustain"], 5)

    def test_sustain_team_summary_and_recommendation_badge(self):
        from services.champion_traits import recommendation_badges, recommendation_team_digest
        library = self.sustain_library((18000, 30), (0, 30))
        profiles = [library.profile(266, 4, "GOLD"), {"values": {"team_sustain": 3}}, {"values": {}}]
        self.assertAlmostEqual(team_summary(profiles)["team_sustain"]["score"], 4.3)
        self.assertEqual(team_summary(profiles)["team_sustain"]["known"], 2)
        self.assertIn("Csapatfenntartás", [b["label"] for b in recommendation_badges(library, "266", 4, "GOLD")])
        self.assertIn("Csapatfenntartás", recommendation_team_digest(library, [("266", 4)], "GOLD")["character"])

    def test_missing_fields_are_not_zero_samples_and_no_identities_are_stored(self):
        info = sample()
        info["participants"][0]["puuid"] = "private"
        observation = extract_observations(info)[0]
        self.assertEqual(observation["metrics"]["cc_seconds_per_min"], [2, 1])
        self.assertEqual(observation["metrics"]["magic_share"], [0, 1])
        self.assertNotIn("taken_per_min", observation["metrics"])
        self.assertNotIn("private", str(observation))
        self.assertEqual(observation["metrics"]["spell1_casts_per_min"], [3, 1])

    def test_invalid_and_short_games_do_not_generate_observations(self):
        value = sample()
        value["participants"][0]["timePlayed"] = 60
        self.assertEqual(extract_observations(value), [])

    def test_aggregates_pool_by_sample_without_age_decay_and_drop_sixth_patch(self):
        patches = [f"16.{i}" for i in range(1, 7)]
        rows = [dict(patch=p, champion=266, role=0, tier="GOLD", metrics={"cc": [i*10, i]})
                for i, p in enumerate(patches, 1)]
        library = TraitLibrary({}, {"patches": patches, "rows": rows}, {})
        self.assertEqual(library.measured(266, 0, "MIXED")["cc"], [200, 20])
        self.assertEqual(library.measured(266, 1, "MIXED"), {})
        self.assertEqual(library.measured(266, 0, "SILVER"), {})

    def test_unknown_is_not_an_invented_low_score_and_manual_values_override_presets(self):
        library = TraitLibrary({"champions": {"266": {"tags": ["Tank"]}}}, {},
                               {"266": {"0": {"frontline": 2, "cc": None}}})
        profile = library.profile(266, 0)
        self.assertEqual(profile["values"]["frontline"], 2)
        self.assertNotIn("cc", profile["values"])
        self.assertIsNone(team_summary([profile])["percent_health"]["score"])
        self.assertFalse(library.accepts(266, 0, "MIXED", ["percent_health"]))

    def test_effective_range_is_not_the_melee_ranged_flag(self):
        library = TraitLibrary({"champions": {"6": {"name": "Urgot", "tags": ["Marksman"],
                                                              "stats": {"attackrange": 350}}}}, {}, {})
        self.assertEqual(library.profile(6, 0)["values"]["range"], 2)
        self.assertEqual(library.profile(6, 0)["attack_range"], 350)

    def test_cc_utilization_uses_observations_only_after_thirty_samples(self):
        row = dict(patch="16.7", tier="GOLD", champion=266, role=0, metrics={"cc_seconds_per_min": [90, 30]})
        library = TraitLibrary({}, {"patches": ["16.7"], "rows": [row]}, {})
        self.assertEqual(library.profile(266, 0)["values"]["cc_use"], 3)
        self.assertNotIn("cc_access", library.profile(266, 0)["values"])
        self.assertNotIn("cc_use", library.profile(266, 1)["values"])

    def test_team_frontline_uses_two_best_and_unknown_coverage_is_explicit(self):
        profiles = [{"values": {"frontline": value, "range": value}} for value in (5, 3, 1)]
        result = team_summary(profiles)
        self.assertAlmostEqual(result["frontline"]["score"], 4.3)
        self.assertEqual(result["frontline"]["known"], 3)
        self.assertEqual(result["range_mix"], {"közel": 1, "közép": 1, "távol": 1, "ismeretlen": 0})

    def test_export_keeps_patch_rows_and_limits_to_last_five(self):
        db = MagicMock()
        db.get_trait_aggregates.return_value = []
        with tempfile.TemporaryDirectory() as folder:
            export_observations(db, [f"16.{i}" for i in range(1, 7)], Path(folder)/"observations.json")
        self.assertEqual(db.get_trait_aggregates.call_args.args[0], ["16.2", "16.3", "16.4", "16.5", "16.6"])


class TraitDatabaseTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TRAITS_POSTGRES_TEST") == "1", "Explicit local PostgreSQL integration test")
    def test_transaction_deduplication_upsert_and_rollback_using_temporary_tables(self):
        import psycopg2
        import configs
        from model.data_manager import DataManager
        from services.trait_store import SCHEMA
        assert configs.DB_HOST in ("localhost", "127.0.0.1", "::1")
        db = DataManager(False)
        db.conn = psycopg2.connect(dbname=configs.DB_NAME, user=configs.DB_USER, password=configs.DB_PASS,
                                   host=configs.DB_HOST, port=configs.DB_PORT)
        self.addCleanup(db.close)
        with db.conn.cursor() as cursor:
            cursor.execute("CREATE TEMP TABLE matches (match_id TEXT PRIMARY KEY, region TEXT, tier TEXT, patch TEXT, blue_win BOOLEAN, blue_team INTEGER[], red_team INTEGER[])")
            cursor.execute(SCHEMA.replace("CREATE TABLE IF NOT EXISTS", "CREATE TEMP TABLE"))
        db.conn.commit()
        value = dict(tier="GOLD", patch="16.7", blue_win=True, blue_team=[1,2,3,4,5], red_team=[6,7,8,9,10],
                     trait_observations=extract_observations(sample()))
        self.assertTrue(db.save_match("one", "euw1", value))
        self.assertFalse(db.save_match("one", "euw1", value))
        self.assertTrue(db.save_match("two", "euw1", value))
        rows = db.get_trait_aggregates(["16.7"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metrics"]["cc_seconds_per_min"], [4, 2])
        with patch("services.trait_store.save_observations", side_effect=RuntimeError("interruption")):
            with self.assertRaises(RuntimeError):
                db.save_match("three", "euw1", value)
        with db.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM matches")
            self.assertEqual(cursor.fetchone()[0], 2)
        self.assertTrue(db.save_match("three", "euw1", value))
        self.assertEqual(db.get_trait_aggregates(["16.7"])[0]["metrics"]["cc_seconds_per_min"], [6, 3])


class TraitUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_test_support import isolate_preferences
        isolate_preferences(self)
        from view.main_window import MainWindow
        self.window = MainWindow(["Aatrox", "Ahri"], {"Aatrox": 0, "Ahri": 1})
        self.addCleanup(self.window.close)

    def test_experiment_is_disabled_by_default_and_filters_do_not_run(self):
        self.assertFalse(self.window.chk_traits.isChecked())
        self.assertTrue(self.window.btn_traits.isHidden())

    def test_dialog_never_invokes_model_inference(self):
        from view.trait_dialog import TraitDialog
        self.window.controller = MagicMock()
        self.window.chk_traits.setChecked(True)
        with patch.object(self.window, "_get_trait_library", return_value=TraitLibrary({}, {}, {})):
            dialog = TraitDialog(self.window)
            self.assertEqual(dialog.controls["frontline"].value(), 0)
            dialog.close()
        self.window.controller.predict_match.assert_not_called()
        self.window.controller.recommend_champs.assert_not_called()

    def test_editing_one_trait_does_not_freeze_measured_values_as_manual(self):
        import json
        from view.trait_dialog import TraitDialog
        library = TraitLibrary({"champions": {"266": {"tags": ["Tank"]}}},
            {"patches": ["16.7"], "rows": [dict(patch="16.7", tier="GOLD", champion=266, role=0,
                                                  metrics={"cc_seconds_per_min": [90, 30]})]}, {})
        with tempfile.TemporaryDirectory() as folder, \
                patch("services.client_settings.settings_path", return_value=Path(folder)/"settings.json"), \
                patch.object(self.window, "_get_trait_library", return_value=library):
            dialog = TraitDialog(self.window)
            dialog.controls["frontline"].setValue(3)
            dialog.save_profile()
            result = json.loads((Path(folder)/"champion_trait_profiles.json").read_text())
            self.assertEqual(result["266"]["0"], {"frontline": 3})
            dialog.close()

    def test_experimental_filter_only_selects_candidates_and_preserves_winrate(self):
        from PyQt6.QtTest import QTest
        library = TraitLibrary({"champions": {"266": {"tags": ["Tank"]}, "103": {"tags": ["Mage"]}}}, {}, {})
        controller = MagicMock()
        controller.validate_draft.return_value = (True, "")
        controller.recommend_champs.return_value = [dict(id=0, wr=53.25, display_color="#FFFFFF")]
        self.window.controller = controller
        self.window.chk_traits.setChecked(True)
        self.window._trait_filters = ["frontline"]
        with patch.object(self.window, "_get_trait_library", return_value=library):
            self.window._on_recommend()
            for _ in range(300):
                if not self.window._busy:
                    break
                QTest.qWait(10)
        self.assertFalse(self.window._busy)
        self.assertEqual(controller.recommend_champs.call_args.kwargs["candidate_ids"], [0])
        self.assertEqual(self.window._last_recommendations[0]["wr"], 53.25)
        self.assertIn("Frontline", self.window._last_recommendations[0]["experimental_tags"])
        controller.predict_match.assert_not_called()
