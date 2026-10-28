"""Headless integration checks for import, optional explanations and GUI workers."""

import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 is not installed")
class MainWindowFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from view.main_window import MainWindow
        self.window = MainWindow(["Aatrox", "Ahri", "Akali", "Alistar"],
                                 {"Aatrox": 0, "Ahri": 1, "Akali": 2, "Alistar": 4})
        self.addCleanup(self.window.close)

    def wait_for_worker(self):
        from PyQt6.QtCore import QElapsedTimer
        from PyQt6.QtTest import QTest
        timer = QElapsedTimer()
        timer.start()
        while self.window._busy and timer.elapsed() < 3000:
            QTest.qWait(10)
        self.assertFalse(self.window._busy, "GUI worker did not finish")

    def test_empty_slots_and_aatrox_have_distinct_ids(self):
        self.assertEqual(self.window._get_ids(self.window.blue_picks), [-1] * 5)
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.assertEqual(self.window._get_ids(self.window.blue_picks), [0, -1, -1, -1, -1])

    def test_reviewed_import_replaces_previous_draft_and_keeps_internal_zero(self):
        self.window.blue_picks[1].setCurrentText("Ahri")
        self.window.red_bans[4].setCurrentText("Ahri")
        self.window._apply_import({"blue": [0, None, None, None, None],
                                   "red": [None, None, 1, None, None],
                                   "blue_bans": [2], "red_bans": [4]})
        self.assertEqual(self.window._get_ids(self.window.blue_picks), [0, -1, -1, -1, -1])
        self.assertEqual(self.window._get_ids(self.window.red_picks), [-1, -1, 1, -1, -1])
        self.assertEqual(self.window._get_ids(self.window.blue_bans), [2, -1, -1, -1, -1])
        self.assertEqual(self.window._get_ids(self.window.red_bans), [4, -1, -1, -1, -1])
        self.assertIsNone(self.window._recommendation_context)

    def test_recommendation_does_not_run_optional_explanations(self):
        controller = MagicMock()
        controller.validate_draft.return_value = (True, "")
        controller.recommend_champs.return_value = [
            {"id": index % 3, "wr": 55.0 - index, "display_color": "#FFFFFF"}
            for index in range(6)
        ]
        self.window.controller = controller
        self.window._on_recommend()
        self.wait_for_worker()
        controller.recommend_champs.assert_called_once()
        controller.explain_recommendations.assert_not_called()
        self.assertEqual(len(self.window._recommendation_context["recommendations"]), 5)
        self.assertTrue(self.window.btn_explain.isEnabled())
        self.window.blue_picks[1].setCurrentText("Ahri")
        self.assertIsNone(self.window._recommendation_context)
        self.assertFalse(self.window.btn_explain.isEnabled())

    def test_background_error_returns_gui_to_editable_state(self):
        def fail():
            raise RuntimeError("<network unavailable>")
        self.window._start_task(fail, lambda _: self.fail("Failure should not invoke success"))
        self.wait_for_worker()
        self.assertTrue(self.window.btn_import.isEnabled())
        self.assertIn("&lt;network unavailable&gt;", self.window.lbl_dash_title.text())

    def test_client_snapshot_is_read_off_thread_before_review(self):
        from threading import get_ident
        main_thread = get_ident()
        worker_threads = []
        snapshot = object()
        def read():
            worker_threads.append(get_ident())
            return snapshot
        with patch("services.draft_import.LcuDraftClient") as factory, patch.object(self.window, "_review_import") as review:
            factory.return_value.read_draft.side_effect = read
            self.window._on_import_draft()
            self.wait_for_worker()
            review.assert_called_once_with(snapshot)
        self.assertNotEqual(worker_threads, [main_thread])

    def test_window_closing_is_deferred_while_operation_is_running(self):
        from PyQt6.QtGui import QCloseEvent
        self.window._set_busy(True)
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.window._set_busy(False)


if __name__ == "__main__":
    unittest.main()
