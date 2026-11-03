"""History refresh runs in a worker; markers leave model output unchanged."""

import importlib.util
import os
from pathlib import Path
import time
from threading import get_ident
import unittest
from unittest.mock import MagicMock, patch

from test_personal_history import profile, MAPPING
from services.personal_history import annotate_recommendations
from services import feature_flags


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 required")
class WindowHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_test_support import isolate_preferences
        isolate_preferences(self)
        enabled = patch.object(feature_flags, "PERSONAL_HISTORY_ENABLED", True)
        enabled.start()
        self.addCleanup(enabled.stop)
        from view.main_window import MainWindow
        self.window = MainWindow(["Aatrox", "Ahri", "Garen"], {"Aatrox": 0, "Ahri": 1, "Garen": 2})
        self.addCleanup(self.window.close)
        self.ctrl = MagicMock()
        self.ctrl.validate_draft.return_value = (True, "")
        self.window.controller = self.ctrl

    def wait(self):
        from PyQt6.QtCore import QElapsedTimer
        from PyQt6.QtTest import QTest
        timer = QElapsedTimer()
        timer.start()
        while self.window._busy and timer.elapsed() < 4000:
            QTest.qWait(10)
        self.assertFalse(self.window._busy)

    def test_refresh_is_off_thread_and_exposes_account_and_count(self):
        thread_ids = []
        def read(*args, **kwargs):
            thread_ids.append(get_ident())
            return profile()
        with patch("services.personal_history.LcuHistoryClient") as client:
            client.return_value.read_history.side_effect = read
            self.window.cb_history_limit.setCurrentIndex(1)
            self.window._on_refresh_history()
            self.wait()
            self.assertEqual(client.return_value.read_history.call_args.args[0], 500)
            self.assertNotIn("allow_client_service", client.return_value.read_history.call_args.kwargs)
        self.assertFalse(self.window.chk_history_service.isChecked())
        self.assertNotEqual(thread_ids, [get_ident()])
        self.assertIn("Tester#EUW", self.window.lbl_history.toolTip())
        self.assertIn("20", self.window.lbl_history.toolTip())
        self.assertTrue(self.window.chk_personal.isEnabled())

    def test_external_client_service_requires_checkbox_and_explicit_refresh(self):
        with patch("services.personal_history.LcuHistoryClient") as client:
            client.return_value.read_history.return_value = profile()
            self.window.chk_history_service.setChecked(True)
            client.assert_not_called()
            self.window._on_refresh_history()
            self.wait()
        self.assertTrue(client.return_value.read_history.call_args.kwargs["allow_client_service"])

    def test_repeated_page_warning_shows_actual_received_count(self):
        from dataclasses import replace
        self.window._history_loaded(replace(profile(0), requested_games=1000, examined_games=20, limited=True))
        self.assertIn("20/1000", self.window.lbl_dash_title.text())
        self.assertIn("ismételte", self.window.lbl_dash_title.text())
        self.assertIn("0 rangsorolt", self.window.lbl_history.toolTip())
        self.assertIn("helyi LCU", self.window.lbl_history.toolTip())

    def test_missing_client_is_a_soft_status_and_releases_controls(self):
        self.window.lbl_dash_title.setText("Manual draft")
        with patch("services.personal_history.LcuHistoryClient") as client:
            client.return_value.read_history.side_effect = RuntimeError("Start the LoL client")
            self.window._on_refresh_history()
            self.wait()
        self.assertEqual(self.window.lbl_dash_title.text(), "Manual draft")
        self.assertIn("Start the LoL client", self.window.lbl_history.toolTip())
        self.assertFalse(self.window.chk_personal.isEnabled())
        self.assertTrue(self.window.btn_rec.isEnabled())

    def test_large_refresh_can_be_cancelled(self):
        def read(*args, cancelled, **kwargs):
            while not cancelled():
                time.sleep(0.005)
            raise RuntimeError("Cancelled")
        with patch("services.personal_history.LcuHistoryClient") as client:
            client.return_value.read_history.side_effect = read
            self.window._on_refresh_history()
            self.assertTrue(self.window.btn_history.isEnabled())
            self.window._on_refresh_history()
            self.wait()
        self.assertIn("Cancelled", self.window.lbl_history.toolTip())

    def test_markers_toggle_without_extra_inference_or_changed_percentages(self):
        history = profile(20, 14)
        self.window._history_loaded(history)
        blue, red = [-1] * 5, [2, -1, -1, -1, -1]
        recs = [{"id": 0, "wr": 55.25, "display_color": "#66BB6A"}]
        self.ctrl.recommend_champs.return_value = annotate_recommendations(recs, history, MAPPING, blue, red, "blue", 0)
        self.window.red_picks[0].setCurrentText("Garen")
        with patch("services.personal_history.LcuHistoryClient") as client:
            self.window._on_recommend()
            self.wait()
            client.return_value.verify_account.assert_called_once_with(history)
        self.assertIn("#FFD166", self.window.lbl_recommend.text())
        self.assertIn("55.25%", self.window.lbl_recommend.text())
        self.assertIn("Bevált saját hős", self.window.lbl_recommend.text())
        self.window.chk_personal.setChecked(False)
        self.assertNotIn("#FFD166", self.window.lbl_recommend.text())
        self.assertIn("55.25%", self.window.lbl_recommend.text())
        self.window.chk_personal.setChecked(True)
        self.assertIn("#FFD166", self.window.lbl_recommend.text())
        self.ctrl.recommend_champs.assert_called_once()
        self.ctrl.explain_champion_effects.assert_not_called()

    def test_account_change_blocks_using_the_previous_accounts_annotations(self):
        self.window._history_loaded(profile())
        with patch("services.personal_history.LcuHistoryClient") as client:
            client.return_value.verify_account.side_effect = RuntimeError("Másik fiók; frissítsd")
            self.window._on_recommend()
            self.wait()
        self.ctrl.recommend_champs.assert_not_called()
        self.assertIn("Másik fiók", self.window.lbl_dash_title.text())

    def test_detailed_analysis_receives_history_and_keeps_the_selected_champion(self):
        history = profile()
        self.window._history_loaded(history)
        self.window._display_recommendations([{"id": 0, "wr": 55, "display_color": "#fff"}],
            {"div": "GOLD", "blue": [-1] * 5, "red": [-1] * 5, "team": "blue", "r_idx": 0})
        self.ctrl.explain_recommendations.return_value = []
        with patch("services.personal_history.LcuHistoryClient"), patch.object(self.window, "_display_explanations"):
            self.window._on_explain()
            self.wait()
        self.assertIs(self.ctrl.explain_recommendations.call_args.kwargs["history"], history)
        self.assertEqual(self.ctrl.explain_recommendations.call_args.kwargs["selected_id"], 0)

    def test_startup_schedules_history_read_after_showing_window(self):
        spec = importlib.util.spec_from_file_location("_history_test_app", Path(__file__).resolve().parents[1] / "erospredictor.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, "QApplication"), patch.object(module, "DataManager") as db, \
                patch.object(module, "Controller"), patch.object(module, "MainWindow") as view, \
                patch.object(module, "QTimer") as timer, patch.object(module.sys, "exit"):
            db.return_value.get_champion_names.return_value = {0: "Aatrox"}
            module.run_predictor()
            view.return_value.show.assert_called_once()
            timer.singleShot.assert_called_once_with(0, view.return_value._on_startup_history)


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 required")
class DisabledWindowHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_test_support import isolate_preferences
        isolate_preferences(self)
        disabled = patch.object(feature_flags, "PERSONAL_HISTORY_ENABLED", False)
        disabled.start()
        self.addCleanup(disabled.stop)
        from view.main_window import MainWindow
        self.window = MainWindow(["Aatrox", "Ahri"], {"Aatrox": 0, "Ahri": 1})
        self.addCleanup(self.window.close)

    def test_history_refresh_is_disabled_before_creating_client_or_worker(self):
        with patch("services.personal_history.LcuHistoryClient", side_effect=AssertionError("History client")) as client, \
                patch.object(self.window, "_start_task", side_effect=AssertionError("History worker")) as task:
            self.window._on_refresh_history()
        client.assert_not_called()
        task.assert_not_called()
        self.assertFalse(self.window._history_loading)
        self.assertIsNone(self.window._task)

    def test_finishing_other_tasks_cannot_reenable_history_controls(self):
        self.window._personal_history = profile()
        self.window._set_busy(True)
        self.window._set_busy(False)
        for control in (self.window.chk_personal, self.window.btn_history, self.window.cb_history_limit,
                        self.window.chk_history_service):
            self.assertFalse(control.isEnabled())
        self.assertFalse(self.window.chk_personal.isChecked())
        self.assertTrue(self.window.btn_rec.isEnabled())

    def test_disabled_history_has_no_startup_timer(self):
        spec = importlib.util.spec_from_file_location("_disabled_history_test_app", Path(__file__).resolve().parents[1] / "erospredictor.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module, "QApplication"), patch.object(module, "DataManager") as db, \
                patch.object(module, "Controller"), patch.object(module, "MainWindow") as view, \
                patch.object(module, "QTimer") as timer, patch.object(module.sys, "exit"):
            db.return_value.get_champion_names.return_value = {0: "Aatrox"}
            module.run_predictor()
            view.return_value.show.assert_called_once()
            timer.singleShot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
