"""Live draft updates, global role assignment and paused personal history."""

from dataclasses import replace
import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

from services.draft_import import DraftPick, DraftSnapshot
from services.role_distribution import ROLE_NAMES


def distributions():
    rows = {"266": [90, 0, 10, 0, 0], "101": [0, 0, 40, 0, 60], "201": [0, 0, 0, 0, 100],
            "103": [0, 0, 100, 0, 0], "55": [0, 0, 100, 0, 0]}
    return {"version": 1, "roles": list(ROLE_NAMES), "tiers": {"MIXED": {
        key: {"counts": counts, "probabilities": [value / sum(counts) for value in counts], "total": sum(counts)}
        for key, counts in rows.items()}}}


def snapshot(braum=True, **kwargs):
    picks = [DraftPick("blue", 266, 0, 0, 0), DraftPick("red", 101, 1, None, 5)]
    if braum:
        picks.append(DraftPick("red", 201, 2, None, 6))
    return DraftSnapshot(tuple(picks), own_team="blue", session_key="game:100", own_role=0, **kwargs)


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 required")
class DraftSyncWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from view.main_window import MainWindow
        self.window = MainWindow(["Aatrox", "Xerath", "Braum", "Ahri", "Katarina", "Nautilus"],
                                 {"Aatrox": 0, "Xerath": 1, "Braum": 2, "Ahri": 3, "Katarina": 4, "Nautilus": 5})
        self.ctrl = MagicMock()
        self.window.controller = self.ctrl
        self.roles = patch.object(self.window, "_load_roles", return_value=(distributions(), False))
        self.roles.start()
        self.addCleanup(self.roles.stop)
        self.addCleanup(self.close)

    def close(self):
        self.window.chk_follow.setChecked(False)
        self.wait()
        self.window.close()

    def wait(self):
        from PyQt6.QtCore import QElapsedTimer
        from PyQt6.QtTest import QTest
        timer = QElapsedTimer()
        timer.start()
        while (self.window._busy or self.window._draft_follower.running) and timer.elapsed() < 4000:
            QTest.qWait(10)
        QTest.qWait(10)
        self.assertFalse(self.window._busy or self.window._draft_follower.running)

    def enable_without_io(self):
        self.window.chk_follow.blockSignals(True)
        self.window.chk_follow.setChecked(True)
        self.window.chk_follow.blockSignals(False)

    def test_following_is_opt_in_and_poll_runs_without_model_inference(self):
        self.assertFalse(self.window.chk_follow.isChecked())
        with patch("services.draft_import.LcuDraftClient") as client:
            client.return_value.read_draft.return_value = snapshot()
            self.window.chk_follow.setChecked(True)
            self.wait()
            client.return_value.read_draft.assert_called_once()
            self.assertEqual(self.window._get_ids(self.window.red_picks), [-1, -1, 1, -1, 2])
        self.ctrl.predict_match.assert_not_called()
        self.ctrl.recommend_champs.assert_not_called()

    def test_new_braum_repositions_xerath_globally_and_updates_bans(self):
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot(False))
        self.assertEqual(self.window._get_ids(self.window.red_picks), [-1, -1, -1, -1, 1])
        self.window._receive_followed_draft(snapshot(own_bans=(3,), enemy_bans=(5,)))
        self.assertEqual(self.window._get_ids(self.window.blue_picks), [0, -1, -1, -1, -1])
        self.assertEqual(self.window._get_ids(self.window.red_picks), [-1, -1, 1, -1, 2])
        self.assertEqual(self.window._get_ids(self.window.blue_bans), [3, -1, -1, -1, -1])
        self.assertEqual(self.window._get_ids(self.window.red_bans), [5, -1, -1, -1, -1])
        self.assertIn("40.0%", self.window.red_role_labels[2].text())
        self.assertIn("100 megfigyelt", self.window.red_picks[2].toolTip())
        self.assertIn("kliens", self.window.blue_role_labels[0].text())

    def test_unchanged_snapshot_keeps_existing_predictions_and_recommendations(self):
        self.enable_without_io()
        current = snapshot()
        self.window._receive_followed_draft(current)
        context = {"already": "calculated"}
        self.window._recommendation_context = context
        self.window.lbl_dash_title.setText("Meglévő eredmény")
        self.window._receive_followed_draft(current)
        self.assertIs(self.window._recommendation_context, context)
        self.assertEqual(self.window.lbl_dash_title.text(), "Meglévő eredmény")

    def test_inflight_snapshot_is_deferred_during_model_work_then_invalidates_old_results(self):
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot(False))
        self.window._set_busy(True)
        self.window._receive_followed_draft(snapshot())
        self.assertEqual(self.window._get_ids(self.window.red_picks)[4], 1)
        self.window._recommendation_context = {"old": True}
        self.window._task_finished()
        self.assertEqual(self.window._get_ids(self.window.red_picks)[4], 2)
        self.assertIsNone(self.window._recommendation_context)

    def test_disable_discards_pending_and_late_updates(self):
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot(False))
        self.window._set_busy(True)
        self.window._receive_followed_draft(snapshot())
        self.window.chk_follow.setChecked(False)
        self.window._set_busy(False)
        self.window._receive_followed_draft(snapshot())
        self.assertEqual(self.window._get_ids(self.window.red_picks)[4], 1)
        self.assertIsNone(self.window._sync_pending)

    def test_known_ally_role_is_authoritative_and_enemy_role_is_statistical(self):
        self.enable_without_io()
        current = snapshot()
        picks = list(current.picks)
        picks[2] = replace(picks[2], role=0)
        self.window._receive_followed_draft(replace(current, picks=tuple(picks)))
        self.assertEqual(self.window._get_ids(self.window.blue_picks)[0], 0)
        self.assertEqual(self.window._get_ids(self.window.red_picks)[4], 2)
        self.assertEqual(self.window._get_ids(self.window.red_picks)[0], -1)

    def test_unknown_side_does_not_overwrite_manual_draft(self):
        self.enable_without_io()
        self.window.blue_picks[2].setCurrentText("Katarina")
        current = replace(snapshot(), own_team=None,
                          picks=(DraftPick("own", 266, 0, 0, 0), DraftPick("enemy", 101, 1, None, 5)))
        self.window._receive_followed_draft(current)
        self.assertEqual(self.window._get_ids(self.window.blue_picks)[2], 4)
        self.assertIn("oldala", self.window.lbl_draft_sync.toolTip())

    def test_missing_distribution_leaves_enemy_unresolved_and_imports_known_picks_and_bans(self):
        self.enable_without_io()
        self.window._load_roles.return_value = ({}, False)
        self.window._receive_followed_draft(snapshot(own_bans=(3,)))
        self.assertEqual(self.window._get_ids(self.window.blue_picks)[0], 0)
        self.assertEqual(self.window._get_ids(self.window.red_picks), [-1] * 5)
        self.assertEqual(self.window._get_ids(self.window.blue_bans)[0], 3)
        self.assertIn("Kézi pozíció szükséges", self.window.lbl_draft_sync.toolTip())

    def test_manual_role_confirmation_is_preserved_until_a_new_session(self):
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot(False))
        self.window.red_picks[2].setCurrentText("Xerath")
        self.window._confirm_manual_role("red", 2)
        self.assertEqual(self.window._get_ids(self.window.red_picks), [-1, -1, 1, -1, -1])
        self.window._receive_followed_draft(snapshot())
        self.assertIn("kézi", self.window.red_role_labels[2].text())
        self.window.btn_red.setChecked(True)
        self.window.grp_role.button(4).setChecked(True)
        self.window._receive_followed_draft(replace(snapshot(False), session_key="game:101"))
        self.assertEqual(self.window._get_ids(self.window.red_picks)[4], 1)
        self.assertEqual(self.window._sync_manual_roles, {})
        self.assertTrue(self.window.btn_blue.isChecked())
        self.assertEqual(self.window.grp_role.checkedId(), 0)

    def test_unavailable_session_resets_manual_choices_without_erasing_the_final_draft(self):
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot())
        self.window._sync_manual_roles[("red", 1)] = 2
        self.window._draft_unavailable()
        self.assertEqual(self.window._sync_manual_roles, {})
        self.assertEqual(self.window._sync_session_key, "")
        self.assertEqual(self.window._get_ids(self.window.red_picks)[4], 2)

    def test_clear_turns_off_following(self):
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot())
        self.window._on_clear_all()
        self.assertFalse(self.window.chk_follow.isChecked())
        self.assertEqual(self.window._get_ids(self.window.red_picks), [-1] * 5)

    def test_manual_review_prefills_global_role_suggestion(self):
        from PyQt6.QtWidgets import QDialog
        with patch("view.draft_import_dialog.DraftImportDialog") as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Rejected
            self.window._review_import(snapshot())
            prepared = dialog.call_args.args[0]
        self.assertEqual({pick.internal_id: pick.role for pick in prepared.picks}, {0: 0, 1: 2, 2: 4})

    def test_manual_review_keeps_confirmed_role_and_marks_accepted_manual_role(self):
        from PyQt6.QtWidgets import QDialog
        self._prepare_manual_follow()
        self.window._sync_manual_roles[("red", 1)] = 2
        values = {"blue": [0, None, None, None, None], "red": [None, None, 1, None, None],
                  "blue_bans": [], "red_bans": []}
        with patch("view.draft_import_dialog.DraftImportDialog") as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            dialog.return_value.own_side.currentData.return_value = "blue"
            dialog.return_value.draft_values.return_value = values
            self.window._review_import(snapshot(False))
            prepared = dialog.call_args.args[0]
        self.assertEqual(prepared.picks[1].role, 2)
        self.assertEqual(self.window._sync_manual_roles[("red", 1)], 2)
        self.assertIn("kézi", self.window.red_role_labels[2].text())

    def _prepare_manual_follow(self):
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot(False))

    def test_manual_review_rejects_contradicting_known_client_side(self):
        from PyQt6.QtWidgets import QDialog
        self._prepare_manual_follow()
        before = self.window._get_ids(self.window.blue_picks)
        with patch("view.draft_import_dialog.DraftImportDialog") as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            dialog.return_value.own_side.currentData.return_value = "red"
            dialog.return_value.draft_values.return_value = {"blue": [None] * 5, "red": [0, None, None, None, None]}
            with self.assertRaisesRegex(ValueError, "ellentmond"):
                self.window._review_import(snapshot(False))
        self.assertEqual(self.window._get_ids(self.window.blue_picks), before)

    def test_manual_import_suspends_following_and_nested_finish_does_not_release_busy(self):
        from PyQt6.QtWidgets import QDialog
        self.enable_without_io()
        self.window._receive_followed_draft(snapshot(False))
        with patch("services.draft_import.LcuDraftClient") as client, \
                patch("view.draft_import_dialog.DraftImportDialog") as dialog:
            client.return_value.read_draft.return_value = snapshot()
            def review():
                self.assertFalse(self.window._draft_follower.enabled)
                self.assertTrue(self.window._busy)
                self.window._task_finished()  # Simulate finished queued by the nested modal event loop.
                self.assertTrue(self.window._busy)
                self.assertFalse(self.window._draft_follower.enabled)
                self.window.chk_follow.setChecked(False)
                return QDialog.DialogCode.Rejected
            dialog.return_value.exec.side_effect = review
            self.window._on_import_draft()
            self.wait()
        self.assertFalse(self.window._manual_import_active)
        self.assertFalse(self.window._draft_follower.enabled)
        self.assertIsNone(self.window._sync_pending)


if __name__ == "__main__":
    unittest.main()
