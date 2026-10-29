"""Headless tests for the optional, focus-relative effects on draft rows."""

import importlib.util
import inspect
import os
import unittest
from unittest.mock import MagicMock, patch


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 is not installed")
class InlineEffectsWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_test_support import isolate_preferences
        isolate_preferences(self)
        from view.main_window import MainWindow
        self.window = MainWindow(["Aatrox", "Ahri", "Akali", "Alistar"],
                                 {"Aatrox": 0, "Ahri": 1, "Akali": 2, "Alistar": 4})
        self.controller = MagicMock()
        self.controller.validate_draft.return_value = (True, "")
        self.controller.predict_match.return_value = {
            "blue_text": "Kék: 55.0%", "red_text": "Piros: 45.0%", "bar_val": 55,
            "blue_win_prob": 55.0, "red_win_prob": 45.0,
        }
        self.controller.recommend_champs.return_value = [
            {"id": 0, "wr": 55.0, "display_color": "#66BB6A"},
            {"id": 1, "wr": 54.0, "display_color": "#66BB6A"},
        ]
        self.controller.explain_champion_effects.side_effect = self.effects
        self.window.controller = self.controller
        self.addCleanup(self.close_window)

    def close_window(self):
        self.wait_for_worker()
        self.window.close()

    def wait_for_worker(self):
        from PyQt6.QtCore import QElapsedTimer
        from PyQt6.QtTest import QTest
        timer = QElapsedTimer()
        timer.start()
        QTest.qWait(10)
        while self.window._busy and timer.elapsed() < 3000:
            QTest.qWait(10)
        self.assertFalse(self.window._busy, "Inline effects worker did not finish")

    def option(self, checked):
        """Set the option without launching its independent toggle action."""
        self.window.chk_effects.blockSignals(True)
        self.window.chk_effects.setChecked(checked)
        self.window.chk_effects.blockSignals(False)

    def effects(self, div, blue, red, team, r_idx, champion_id=None):
        focus_id = champion_id if champion_id is not None else (blue if team == "blue" else red)[r_idx]
        interactions = []
        for absolute_team, champions in (("blue", blue), ("red", red)):
            for role, champion in enumerate(champions):
                if champion < 0 or (absolute_team == team and role == r_idx):
                    continue
                lane = absolute_team != team and role == r_idx
                interactions.append({
                    "champion_id": champion, "team": absolute_team,
                    "side": "ally" if absolute_team == team else "opponent",
                    "role_index": role, "delta_pp": -2.5 if lane else 1.25,
                    "score_delta_pp": -2.5 if lane else 1.25,
                    "is_lane_opponent": lane,
                })
        return {"id": focus_id, "team": team, "role_index": r_idx, "wr": 55.0,
                "baseline_win_prob": 52.0, "candidate_delta_pp": 3.0,
                "interactions": interactions}

    def effect_arguments(self):
        args, kwargs = self.controller.explain_champion_effects.call_args
        return inspect.signature(self.effects).bind(*args, **kwargs).arguments

    def all_badges(self):
        return self.window.blue_effects + self.window.red_effects

    def test_default_prediction_runs_no_optional_effect_inference(self):
        self.assertFalse(self.window.chk_effects.isChecked())
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window._on_predict()
        self.wait_for_worker()
        self.controller.predict_match.assert_called_once()
        self.controller.explain_champion_effects.assert_not_called()
        self.controller.explain_recommendations.assert_not_called()

    def test_default_recommendation_runs_no_optional_effect_inference(self):
        self.window._on_recommend()
        self.wait_for_worker()
        self.controller.recommend_champs.assert_called_once()
        self.controller.explain_champion_effects.assert_not_called()
        self.controller.explain_recommendations.assert_not_called()

    def test_checked_prediction_displays_aatrox_focus_and_lane_effect(self):
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window.blue_picks[2].setCurrentText("Ahri")
        self.window.red_picks[0].setCurrentText("Akali")
        self.option(True)
        self.window._on_predict()
        self.wait_for_worker()
        self.controller.explain_champion_effects.assert_called_once()
        self.assertEqual(self.effect_arguments()["blue"][0], 0)
        self.assertEqual(self.effect_arguments()["team"], "blue")
        self.assertIn("Fókusz", self.window.blue_effects[0].text())
        self.assertIn("▲", self.window.blue_effects[2].text())
        self.assertIn("▼", self.window.red_effects[0].text())
        self.assertIn("2.50", self.window.red_effects[0].text())
        self.assertTrue(self.window.red_effects[0].property("laneOpponent"))
        self.assertFalse(self.window.blue_effects[2].property("laneOpponent"))
        self.assertTrue(self.window.blue_effects[0].property("focusChampion"))
        self.controller.explain_recommendations.assert_not_called()

    def test_red_focus_places_badges_on_the_correct_absolute_team(self):
        self.window.btn_red.click()
        self.window.red_picks[0].setCurrentText("Aatrox")
        self.window.red_picks[2].setCurrentText("Ahri")
        self.window.blue_picks[0].setCurrentText("Akali")
        self.option(True)
        self.window._on_predict()
        self.wait_for_worker()
        self.assertEqual(self.effect_arguments()["team"], "red")
        self.assertIn("Fókusz", self.window.red_effects[0].text())
        self.assertIn("▲", self.window.red_effects[2].text())
        self.assertIn("▼", self.window.blue_effects[0].text())
        self.assertNotEqual(self.window.blue_effects[0].styleSheet(),
                            self.window.red_effects[2].styleSheet())

    def test_recommendation_candidate_click_changes_the_effect_focus(self):
        self.window.red_picks[0].setCurrentText("Akali")
        self.option(True)
        self.window._on_recommend()
        self.wait_for_worker()
        self.assertEqual(self.effect_arguments()["champion_id"], 0)
        self.assertIn("champion:1", self.window.lbl_recommend.text())
        self.controller.explain_champion_effects.reset_mock()
        self.window._on_recommendation_selected("champion:1")
        self.wait_for_worker()
        self.controller.explain_champion_effects.assert_called_once()
        self.assertEqual(self.effect_arguments()["champion_id"], 1)
        self.assertIn("Ahri", self.window.lbl_effects_hint.text())

    def test_unchecked_candidate_click_does_not_run_effect_inference(self):
        self.window._on_recommend()
        self.wait_for_worker()
        self.window._on_recommendation_selected("champion:1")
        self.wait_for_worker()
        self.controller.explain_champion_effects.assert_not_called()

    def test_detailed_analysis_includes_selected_candidate_outside_top_five(self):
        ids = [0, 1, 2, 4, 5, 6]
        self.controller.recommend_champs.return_value = [
            {"id": cid, "wr": 55.0 - index, "display_color": "#FFFFFF"}
            for index, cid in enumerate(ids)]
        self.controller.explain_recommendations.return_value = []
        self.window._on_recommend()
        self.wait_for_worker()
        self.window._on_recommendation_selected("champion:6")
        with patch.object(self.window, "_display_explanations") as display:
            self.window._on_explain()
            self.wait_for_worker()
        request = self.controller.explain_recommendations.call_args.kwargs
        self.assertEqual(request["selected_id"], 6)
        self.assertEqual([item["id"] for item in request["recommendations"]], ids[:5])
        self.controller.explain_champion_effects.assert_not_called()
        display.assert_called_once_with([])

    def test_draft_edit_removes_stale_arrows_and_focus(self):
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window.red_picks[0].setCurrentText("Akali")
        self.option(True)
        self.window._on_predict()
        self.wait_for_worker()
        self.assertTrue(any(badge.text() for badge in self.all_badges()))
        self.window.red_picks[0].setCurrentText("Ahri")
        self.assertTrue(all(not badge.text() for badge in self.all_badges()))

    def test_turning_effects_off_clears_badges_without_more_inference(self):
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window.red_picks[0].setCurrentText("Akali")
        self.option(True)
        self.window._on_predict()
        self.wait_for_worker()
        calls = self.controller.explain_champion_effects.call_count
        self.window.chk_effects.setChecked(False)
        self.wait_for_worker()
        self.assertTrue(all(not badge.text() for badge in self.all_badges()))
        self.assertEqual(self.controller.explain_champion_effects.call_count, calls)

    def test_focus_role_change_removes_stale_badges_without_more_inference(self):
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window.red_picks[0].setCurrentText("Akali")
        self.option(True)
        self.window._on_predict()
        self.wait_for_worker()
        calls = self.controller.explain_champion_effects.call_count
        self.window.grp_role.button(2).click()
        self.assertTrue(all(not badge.text() for badge in self.all_badges()))
        self.assertIsNone(self.window._effects_context)
        self.assertEqual(self.controller.explain_champion_effects.call_count, calls)

    def test_empty_focus_does_not_block_normal_prediction(self):
        self.window.blue_picks[2].setCurrentText("Ahri")
        self.option(True)
        self.window._on_predict()
        self.wait_for_worker()
        self.controller.predict_match.assert_called_once()
        self.controller.explain_champion_effects.assert_not_called()
        self.assertEqual(self.window.bar_wr.value(), 55)
        self.assertIn("Kék: 55.0%", self.window.lbl_blue_wr.text())


if __name__ == "__main__":
    unittest.main()
