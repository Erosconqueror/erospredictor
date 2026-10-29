"""Save/load draft actions are local and protect the active draft on failure."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from services import saved_drafts as drafts
from services.draft_import import DraftPick, DraftSnapshot
from test_saved_drafts import MAPPING, VALUES
from ui_test_support import isolate_preferences
from view.main_window import MainWindow

NAMES = {0: "Aatrox", 1: "Ahri", 2: "Garen", 3: "Braum", 4: "Katarina", 5: "Nunu"}


class SavedDraftWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        isolate_preferences(self)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.path = self.root / "saved.json"
        directory = patch.object(drafts, "drafts_directory", return_value=self.root)
        directory.start()
        self.addCleanup(directory.stop)
        self.window = MainWindow(list(NAMES.values()), {name: index for index, name in NAMES.items()})
        self.window._pool_mapping = MAPPING
        self.ctrl = MagicMock()
        self.ctrl.validate_draft.return_value = (True, "")
        self.ctrl.predict_match.return_value = {"blue_text": "Kék: 55%", "red_text": "Piros: 45%",
            "bar_val": 55, "blue_win_prob": 55, "red_win_prob": 45}
        self.window.controller = self.ctrl
        self.addCleanup(self.window.close)

    def values(self):
        return {key: [None if cid == -1 else cid for cid in self.window._get_ids(combos)]
                for key, combos in (("blue", self.window.blue_picks), ("red", self.window.red_picks),
                    ("blue_bans", self.window.blue_bans), ("red_bans", self.window.red_bans))}

    def wait(self):
        from PyQt6.QtTest import QTest
        for _ in range(400):
            if not self.window._busy:
                break
            QTest.qWait(10)
        self.assertFalse(self.window._busy)

    def prepare_file(self):
        drafts.save_draft(self.path, drafts.make_document(VALUES, MAPPING, "GOLD", "red", 4))

    def mark_following_without_io(self):
        self.window.chk_follow.blockSignals(True)
        self.window.chk_follow.setChecked(True)
        self.window.chk_follow.blockSignals(False)

    def test_save_preserves_all_slots_and_context_without_model_or_client_calls(self):
        self.window._apply_import(deepcopy(VALUES))
        self.window.cb_rank.setCurrentText("GOLD")
        self.window.btn_red.click()
        self.window.grp_role.button(4).click()
        with patch("view.main_window.QFileDialog.getSaveFileName", return_value=(str(self.path), "")):
            self.window.action_save_draft.trigger()
        restored = drafts.load_draft(self.path, MAPPING)
        self.assertEqual(restored.values, VALUES)
        self.assertEqual((restored.tier, restored.team, restored.role), ("GOLD", "red", 4))
        self.ctrl.predict_match.assert_not_called()
        self.ctrl.recommend_champs.assert_not_called()

    def test_load_replaces_inputs_stops_following_and_discards_all_old_analysis(self):
        self.prepare_file()
        self.window.blue_picks[1].setCurrentText("Nunu")
        self.mark_following_without_io()
        stale = DraftSnapshot((DraftPick("blue", 20, 5, 1, 0),), own_team="blue", session_key="old")
        self.window._sync_pending = self.window._sync_last_snapshot = stale
        self.window._sync_manual_roles = {("blue", 5): 1}
        self.window._prediction_context = {"old": True}
        self.window._recommendation_context = {"old": True}
        with patch("view.main_window.QFileDialog.getOpenFileName", return_value=(str(self.path), "")):
            self.window.action_load_draft.trigger()
        self.assertEqual(self.values(), VALUES)
        self.assertEqual(self.window.cb_rank.currentText(), "GOLD")
        self.assertTrue(self.window.btn_red.isChecked())
        self.assertEqual(self.window.grp_role.checkedId(), 4)
        self.assertFalse(self.window.chk_follow.isChecked())
        self.assertFalse(self.window._draft_follower.enabled)
        self.assertIsNone(self.window._sync_pending)
        self.assertEqual(self.window._sync_manual_roles, {})
        self.assertIsNone(self.window._prediction_context)
        self.assertIsNone(self.window._recommendation_context)
        self.assertFalse(self.window.btn_explain.isEnabled())
        self.window._receive_followed_draft(stale)
        self.assertEqual(self.values(), VALUES)
        self.ctrl.predict_match.assert_not_called()
        self.ctrl.recommend_champs.assert_not_called()
        self.window._on_predict()
        self.wait()
        self.assertTrue(self.window.btn_explain.isEnabled())
        self.assertEqual(self.window._prediction_context["team"], "red")

    def test_invalid_file_preserves_current_draft_focus_prediction_and_follow_choice(self):
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window._on_predict()
        self.wait()
        previous = self.values()
        context = deepcopy(self.window._prediction_context)
        self.mark_following_without_io()
        for content in ("invalid json", json.dumps({"version": 99})):
            self.path.write_text(content)
            with patch("view.main_window.QFileDialog.getOpenFileName", return_value=(str(self.path), "")):
                self.window._on_load_draft()
            self.assertEqual(self.values(), previous)
            self.assertEqual(self.window._prediction_context, context)
            self.assertTrue(self.window.btn_explain.isEnabled())
            self.assertTrue(self.window.chk_follow.isChecked())

    def test_cancelled_file_dialogs_do_not_change_the_draft(self):
        self.window.blue_picks[0].setCurrentText("Aatrox")
        previous = self.values()
        self.mark_following_without_io()
        with patch("view.main_window.QFileDialog.getSaveFileName", return_value=("", "")):
            self.window._on_save_draft()
        with patch("view.main_window.QFileDialog.getOpenFileName", return_value=("", "")):
            self.window._on_load_draft()
        self.assertEqual(self.values(), previous)
        self.assertTrue(self.window.chk_follow.isChecked())
        self.assertFalse(self.path.exists())

    def test_unfinished_hero_text_cannot_be_saved_as_an_empty_slot(self):
        self.window.blue_picks[0].setEditText("Aat")
        with patch("view.main_window.QFileDialog.getSaveFileName") as dialog:
            self.window._on_save_draft()
        dialog.assert_not_called()
        self.assertIn("fejezd be", self.window.lbl_draft_sync.text())

    def test_busy_actions_and_keyboard_shortcuts_cannot_replace_worker_inputs(self):
        self.window._set_busy(True)
        self.assertFalse(self.window.action_save_draft.isEnabled())
        self.assertFalse(self.window.action_load_draft.isEnabled())
        with patch("view.main_window.QFileDialog.getOpenFileName") as read, \
                patch("view.main_window.QFileDialog.getSaveFileName") as write:
            self.window._on_save_draft()
            self.window._on_load_draft()
        read.assert_not_called()
        write.assert_not_called()
        self.window._set_busy(False)
        self.assertTrue(self.window.btn_drafts.isEnabled())

    def test_save_captures_one_consistent_snapshot_before_file_dialog(self):
        self.window._apply_import(deepcopy(VALUES))
        def choose(*args):
            self.window.red_picks[0].setCurrentText("Nunu")
            return str(self.path), ""
        with patch("view.main_window.QFileDialog.getSaveFileName", side_effect=choose):
            self.window._on_save_draft()
        self.assertEqual(drafts.load_draft(self.path, MAPPING).values, VALUES)
        self.assertEqual(self.values()["red"][0], 5)


if __name__ == "__main__":
    unittest.main()
