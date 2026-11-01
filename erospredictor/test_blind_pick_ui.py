"""Blind warnings stay inline, hide low risk, and never open a separate window."""

import os
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication
from services.blind_pick import blind_risk
from test_blind_pick import Index
from ui_test_support import isolate_preferences
from view.main_window import MainWindow


class BlindPickWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        isolate_preferences(self)
        self.window = MainWindow(["Aatrox", "Ahri", "Braum", "Xerath"], {"Aatrox": 0, "Ahri": 1, "Braum": 2, "Xerath": 3})
        self.window.controller = MagicMock()
        self.addCleanup(self.window.close)

    def test_colored_inline_text_has_no_details_link_or_dialog(self):
        info = blind_risk(Index(), "GOLD", 0, 2, [-1, 10, 11, 12, 13])
        rec = {"id": 0, "wr": 55.5, "display_color": "#66BB6A", "blind": info}
        context = dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2)
        self.window._display_recommendations([rec], context)
        html = self.window.lbl_recommend.text()
        self.assertNotIn("blind:", html)
        self.assertIn("Blind ~20%", html)
        self.assertIn("55.50%", html)
        with patch("view.main_window.QDialog.exec", side_effect=AssertionError("No blind window should open")):
            self.window.lbl_recommend.linkActivated.emit("blind:0")
        self.assertEqual(self.window.controller.mock_calls, [])

    def test_safe_blind_and_full_enemy_roster_leave_no_label_or_empty_extra_line(self):
        context = dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2)
        for info in ({"status": "estimated", "risk_percent": 4, "upper_percent": 5},
                     blind_risk(Index(), "GOLD", 0, 2, [1, 10, 11, 12, 13])):
            with self.subTest(status=info["status"]):
                self.window._display_recommendations([{"id": 0, "wr": 55.5, "display_color": "#66BB6A", "blind": info}], context)
                html = self.window.lbl_recommend.text()
                self.assertNotIn("Blind", html)
                self.assertNotIn("counter", html)
                self.assertNotIn("<br>", html)
                self.assertIn("55.50%", html)

    def test_missing_artifact_stays_gray_and_does_not_hide_recommendation(self):
        rec = {"id": 0, "wr": 55.5, "display_color": "#66BB6A",
               "blind": blind_risk(None, "GOLD", 0, 2, [-1] * 5)}
        self.window._display_recommendations([rec], dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2))
        html = self.window.lbl_recommend.text()
        self.assertIn("55.50%", html)
        self.assertIn("Blind: nincs adat", html)
        self.assertIn("#9E9EAE", html)


if __name__ == "__main__":
    unittest.main()
