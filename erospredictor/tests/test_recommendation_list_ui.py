"""Every ranked candidate remains reachable, including the last losing pick."""

import os
from pathlib import Path
import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtTest import QTest
from ui_test_support import isolate_preferences
from view.main_window import MainWindow


class CompleteRecommendationListTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])
        cls.font = QFontDatabase.addApplicationFont(str(Path("C:/Windows/Fonts/segoeui.ttf")))

    @classmethod
    def tearDownClass(cls):
        if cls.font >= 0:
            QFontDatabase.removeApplicationFont(cls.font)

    def setUp(self):
        isolate_preferences(self)
        mapping = {f"Champion {i:03d}": i for i in range(173)}
        self.window = MainWindow(list(mapping), mapping)
        self.addCleanup(self.window.close)
        self.recs = [dict(id=i, wr=70-i*.2, display_color="#FFFFFF") for i in range(173)]
        self.window.resize(1180, 820)
        self.window._display_recommendations(self.recs,
            dict(div="MIXED", blue=[-1]*5, red=[-1]*5, team="blue", r_idx=0))
        self.window.show()
        QTest.qWait(30)

    def assert_last_pick_reachable(self):
        browser = self.window.scroll_recommend
        bar = browser.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(bar.maximum())
        QTest.qWait(10)
        last = browser.document().find("Champion 172")
        self.assertFalse(last.isNull())
        rect = browser.cursorRect(last)
        self.assertGreaterEqual(rect.top(), 0)
        self.assertLessEqual(rect.bottom(), browser.viewport().height())
        self.assertIn("35.60%", browser.toPlainText())
        self.assertEqual(self.window.lbl_recommend.text().count("<tr"), 173)
        self.assertIn("173 ajánlás", self.window.lbl_counterpick.text())

    def test_full_list_includes_last_below_fifty_percent_pick(self):
        self.assert_last_pick_reachable()

    def test_last_pick_remains_reachable_after_resize_and_focus_render(self):
        self.assert_last_pick_reachable()
        self.window.resize(1280, 940)
        self.window._render_recommendation_rows()
        QTest.qWait(20)
        self.assert_last_pick_reachable()
