"""Blind warnings stay inline, hide low risk, and never open a separate window."""

import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QTextDocument, QFontDatabase
from services.blind_pick import blind_risk
from test_blind_pick import Index
from ui_test_support import isolate_preferences
from view.main_window import MainWindow


class BlindPickWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])
        # Windows offscreen Qt does not enumerate system fonts like the desktop backend.
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
        cls.font_id = QFontDatabase.addApplicationFont(str(font)) if font.is_file() else -1

    @classmethod
    def tearDownClass(cls):
        if cls.font_id >= 0:
            QFontDatabase.removeApplicationFont(cls.font_id)

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
        self.assertIn("Más role ~20%", html)
        self.assertGreater(html.index("Más role ~20%"), html.index("55.50%"))
        self.assertIn("55.50%", html)
        with patch("view.main_window.QDialog.exec", side_effect=AssertionError("No blind window should open")):
            self.window.lbl_recommend.linkActivated.emit("blind:0")
        self.assertEqual(self.window.controller.mock_calls, [])

    def test_safe_blind_and_full_enemy_roster_leave_no_warning_and_reserve_same_row_height(self):
        context = dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2)
        for info in ({"status": "estimated", "risk_percent": 4, "upper_percent": 5},
                     blind_risk(Index(), "GOLD", 0, 2, [1, 10, 11, 12, 13])):
            with self.subTest(status=info["status"]):
                self.window._display_recommendations([{"id": 0, "wr": 55.5, "display_color": "#66BB6A", "blind": info}], context)
                html = self.window.lbl_recommend.text()
                self.assertNotIn("Blind", html)
                self.assertNotIn("counter", html)
                self.assertEqual(html.count("<br>"), 2)
                self.assertIn("55.50%", html)

    def test_missing_artifact_stays_gray_and_does_not_hide_recommendation(self):
        rec = {"id": 0, "wr": 55.5, "display_color": "#66BB6A",
               "blind": blind_risk(None, "GOLD", 0, 2, [-1] * 5)}
        self.window._display_recommendations([rec], dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2))
        html = self.window.lbl_recommend.text()
        self.assertIn("55.50%", html)
        self.assertIn("Blind: nincs adat", html)
        self.assertIn("#9E9EAE", html)

    def test_list_frames_and_independent_blind_colors_share_two_line_rows(self):
        info = {"lane": {"status": "estimated", "risk_percent": 20, "upper_percent": 20},
                "others": {"status": "estimated", "risk_percent": 80, "upper_percent": 80}}
        for tag, color in (("played", "#64B5F6"), ("comfort", "#81C784"), ("learn", "#B39DDB")):
            with self.subTest(tag=tag):
                self.window._display_recommendations([{"id": 0, "wr": 55.5, "display_color": "#66BB6A",
                    "pool_tags": (tag,), "blind": info}],
                    dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2))
                html = self.window.lbl_recommend.text()
                self.assertIn(f"border-top: 2px solid {color}", html)
                self.assertEqual(html.count("<br>"), 2)
                self.assertGreater(html.index("Lane ~20%"), html.index("55.50%"))
                self.assertIn("Más role ~80%", html)

    def test_combining_pool_tags_and_blind_warnings_does_not_increase_row_height(self):
        context = dict(div="GOLD", blue=[-1] * 5, red=[-1] * 5, team="blue", r_idx=2)
        base = {"id": 0, "wr": 55.5, "display_color": "#66BB6A"}
        self.window.chk_personal.blockSignals(True)
        self.window.chk_personal.setChecked(True)
        self.window.chk_personal.blockSignals(False)
        warning = {"lane": {"status": "estimated", "risk_percent": 65, "upper_percent": 65},
                   "others": {"status": "estimated", "risk_percent": 20, "upper_percent": 20}}
        heights = []
        for row in (base, dict(base, pool_tags=("played",)), dict(base, blind=warning),
                    dict(base, blind=warning, pool_tags=("played", "comfort", "learn")),
                    dict(base, blind=warning, pool_tags=("played", "comfort", "learn"),
                         personal={"marker": "good"}, robust={"count": 3, "min": 45, "max": 65, "forced": True})):
            self.window._display_recommendations([row], context)
            document = QTextDocument()
            document.setDefaultFont(self.window.lbl_recommend.font())
            document.setHtml(self.window.lbl_recommend.text())
            document.setTextWidth(460)
            heights.append(document.size().height())
            self.assertLessEqual(document.idealWidth(), 462, str(row))
        self.assertEqual(len(set(heights)), 1, heights)


if __name__ == "__main__":
    unittest.main()
