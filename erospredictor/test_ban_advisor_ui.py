"""Independent role selection and account-scoped personal ban evidence."""

import os
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from services.ban_advisor import recommend_bans
from services.draft_import import DraftImportError
from test_ban_personalization import MAPPING, games, history, role_document
from ui_test_support import isolate_preferences
from view.ban_advisor_dialog import BanAdvisorDialog
from view.main_window import MainWindow


NAMES = {0: "Aatrox", 1: "Ahri", 2: "Katarina", 3: "Xerath", 4: "Braum"}
ROLES = ["Top", "Jungle", "Mid", "ADC", "Support"]


class BanAdvisorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        isolate_preferences(self)
        self.window = MainWindow(list(NAMES.values()), {name: cid for cid, name in NAMES.items()})
        self.window._pool_mapping = MAPPING
        self.window._pool_library = {"version": 1, "champions": {"266": ["played"]}, "lists": {}}
        self.stats = MagicMock()
        self.stats.get_lane_matchup.return_value = {"matches": 100, "weighted_winrate": 40}
        self.controller = MagicMock()
        self.controller.validate_draft.return_value = (True, "")
        self.controller.recommend_ban_lists.side_effect = self.lists
        self.window.controller = self.controller
        self.addCleanup(self.window.close)

    def lists(self, div, pool, unavailable, history=None):
        return {role: recommend_bans(self.stats, role_document(), MAPPING, pool, div, role,
                    unavailable, history=history) for role in range(5)}

    def wait(self):
        from PyQt6.QtTest import QTest
        for _ in range(400):
            if not self.window._busy:
                break
            QTest.qWait(10)
        self.assertFalse(self.window._busy)

    def test_dialog_switches_all_roles_without_changing_main_focus_or_recalculating(self):
        self.window._personal_history = history(games(10, wins=2))
        self.window.chk_personal.setChecked(False)  # Recommendation markers are independent.
        inspected = []
        def inspect(dialog):
            self.assertEqual(dialog.cb_role.currentData(), 0)
            self.assertEqual(dialog.cb_role.count(), 5)
            dialog.cb_role.setCurrentIndex(2)
            text = dialog.details.toPlainText()
            self.assertIn("50% általános + 50% saját", text)
            self.assertIn("30.0%", text)
            self.assertIn("2/10 győzelem", text)
            self.assertIn("Test#EU", dialog.history_notice.text())
            dialog.cb_role.setCurrentIndex(4)
            self.assertIn("100% általános", dialog.details.toPlainText())
            self.assertEqual(self.window.grp_role.checkedId(), 0)
            self.assertEqual(self.controller.recommend_ban_lists.call_count, 1)
            inspected.append(True)
            return 0
        with patch("services.personal_history.LcuHistoryClient") as client, \
             patch.object(BanAdvisorDialog, "exec", new=inspect):
            self.window._on_ban_advice()
            self.wait()
            client.return_value.verify_account.assert_called_once_with(self.window._personal_history)
            client.return_value.read_history.assert_not_called()
        self.assertEqual(inspected, [True])
        self.assertEqual(self.window._ban_role, 4)
        self.controller.predict_match.assert_not_called()
        self.controller.recommend_champs.assert_not_called()

    def test_different_or_unavailable_account_falls_back_to_general_statistics(self):
        self.window._personal_history = history(games(10))
        inspected = []
        def inspect(dialog):
            dialog.cb_role.setCurrentIndex(2)
            self.assertIn("nem ellenőrizhető", dialog.history_notice.text())
            self.assertIn("100% általános", dialog.details.toPlainText())
            self.assertNotIn("50% általános + 50% saját", dialog.details.toPlainText())
            inspected.append(True)
            return 0
        with patch("services.personal_history.LcuHistoryClient") as client, \
             patch.object(BanAdvisorDialog, "exec", new=inspect):
            client.return_value.verify_account.side_effect = DraftImportError("Másik fiók")
            self.window._on_ban_advice()
            self.wait()
        self.assertEqual(inspected, [True])
        self.assertNotIn("history", self.controller.recommend_ban_lists.call_args.kwargs)

    def test_no_history_does_not_contact_client_and_empty_role_stays_selectable(self):
        self.window._ban_role = 2
        self.controller.recommend_ban_lists.side_effect = None
        self.controller.recommend_ban_lists.return_value = self.lists("MIXED", {0}, ()) | {2: []}
        inspected = []
        def inspect(dialog):
            self.assertEqual(dialog.cb_role.currentData(), 2)
            self.assertIn("legalább 20 meccs", dialog.details.toPlainText())
            self.assertIn("Nincs betöltött saját előzmény", dialog.history_notice.text())
            dialog.cb_role.setCurrentIndex(0)
            self.assertIn("1.", dialog.details.toPlainText())
            inspected.append(True)
            return 0
        with patch("services.personal_history.LcuHistoryClient") as client, \
             patch.object(BanAdvisorDialog, "exec", new=inspect):
            self.window._on_ban_advice()
            self.wait()
            client.assert_not_called()
        self.assertEqual(inspected, [True])

    def test_insufficient_personal_sample_is_visible_but_never_labeled_half_weight(self):
        lists = self.lists("MIXED", {0}, (), history(games(9)))
        dialog = BanAdvisorDialog(lists, NAMES, ROLES, 2, "Bansegéd", "Test", self.window)
        self.addCleanup(dialog.close)
        text = dialog.details.toPlainText()
        self.assertIn("saját minta: 9/10", text)
        self.assertIn("100% általános", text)
        self.assertNotIn("50% általános + 50% saját", text)


if __name__ == "__main__":
    unittest.main()
