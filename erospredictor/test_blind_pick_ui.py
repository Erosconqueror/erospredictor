"""Blind warnings are descriptive, clickable and do not launch inference."""

import os
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication
from services.blind_pick import blind_risk
from test_blind_pick import Index
from ui_test_support import isolate_preferences
from view.main_window import MainWindow
from view.blind_pick_dialog import BlindPickDialog


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

    def test_colored_link_opens_evidence_without_extra_neural_calls(self):
        info = blind_risk(Index(), "GOLD", 0, 2, [-1, 10, 11, 12, 13])
        rec = {"id": 0, "wr": 55.5, "display_color": "#66BB6A", "blind": info}
        context = dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2)
        self.window._display_recommendations([rec], context)
        html = self.window.lbl_recommend.text()
        self.assertIn("blind:0", html)
        self.assertIn("Blind ~20%", html)
        self.assertIn("55.50%", html)
        seen = []
        def inspect(dialog):
            text = dialog.details.toPlainText()
            self.assertIn("Ahri", text)
            self.assertIn("nem az ellenfél döntésének", text)
            self.assertIn("lefedettség", text)
            seen.append(True)
            return 0
        with patch.object(BlindPickDialog, "exec", new=inspect):
            self.window.lbl_recommend.linkActivated.emit("blind:0")
        self.assertEqual(seen, [True])
        self.assertEqual(self.window.controller.mock_calls, [])

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
