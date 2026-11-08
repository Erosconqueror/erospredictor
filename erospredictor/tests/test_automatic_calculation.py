"""Debounced model work follows the latest meaningful imported draft."""

from dataclasses import replace
import unittest
from unittest.mock import patch
from PyQt6.QtTest import QTest
import tests.test_main_window_draft_sync as fixtures


class AutomaticCalculationTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.DraftSyncWindowTests.setUpClass.__func__)
    setUp = fixtures.DraftSyncWindowTests.setUp
    close = fixtures.DraftSyncWindowTests.close
    wait = fixtures.DraftSyncWindowTests.wait
    enable_without_io = fixtures.DraftSyncWindowTests.enable_without_io

    def test_changed_snapshots_coalesce_and_repeat_last_recommendation(self):
        self.enable_without_io()
        self.window._last_draft_action = "recommend"
        self.window._receive_followed_draft(fixtures.snapshot(False))
        self.window.grp_role.button(2).setChecked(True)
        self.window._receive_followed_draft(fixtures.snapshot())
        with patch.object(self.window, "_on_recommend") as recommend, patch.object(self.window, "_on_predict") as predict:
            QTest.qWait(460)
            recommend.assert_called_once()
            predict.assert_not_called()
            self.window._receive_followed_draft(fixtures.snapshot())
            QTest.qWait(460)
            recommend.assert_called_once()

    def test_filled_focus_uses_prediction_without_forgetting_recommendation_mode(self):
        self.enable_without_io()
        self.window._last_draft_action = "recommend"
        with patch.object(self.window, "_on_predict") as predict:
            self.window._receive_followed_draft(fixtures.snapshot())
            self.window._run_auto_calculation()
            predict.assert_called_once()
            self.assertEqual(self.window._last_draft_action, "recommend")

    def test_no_manual_action_yet_does_not_start_inference(self):
        self.enable_without_io()
        with patch.object(self.window, "_on_predict") as predict, patch.object(self.window, "_on_recommend") as recommend:
            self.window._receive_followed_draft(fixtures.snapshot())
            QTest.qWait(460)
            predict.assert_not_called()
            recommend.assert_not_called()

    def test_full_recommendation_worker_repeats_with_latest_draft_and_keeps_focus(self):
        self.enable_without_io()
        self.ctrl.validate_draft.return_value = (True, "")
        self.ctrl.recommend_champs.return_value = [dict(id=3, wr=54.2, display_color="#66BB6A")]
        self.ctrl.draft_scenarios.return_value = ()
        self.window._receive_followed_draft(fixtures.snapshot(False))
        self.window.grp_role.button(2).setChecked(True)
        self.window._on_recommend()
        self.wait()
        self.assertEqual(self.window._last_draft_action, "recommend")
        self.assertEqual(self.ctrl.recommend_champs.call_count, 1)
        self.window._receive_followed_draft(fixtures.snapshot())
        QTest.qWait(460)
        self.wait()
        self.assertEqual(self.ctrl.recommend_champs.call_count, 2)
        args = self.ctrl.recommend_champs.call_args.args
        self.assertEqual(args[2][4], 2)
        self.assertEqual(args[4:6], ("blue", 2))
        self.assertEqual(self.window.grp_role.checkedId(), 2)

    def test_latest_snapshot_applies_after_busy_work_before_recalculation(self):
        self.enable_without_io()
        self.window._last_draft_action = "predict"
        self.window._receive_followed_draft(fixtures.snapshot(False))
        self.window._set_busy(True)
        self.window._receive_followed_draft(fixtures.snapshot())
        with patch.object(self.window, "_on_predict") as predict:
            self.window._run_auto_calculation()
            predict.assert_not_called()
            self.window._task_finished()
            self.window._run_auto_calculation()
            predict.assert_called_once()
            self.assertEqual(self.window._get_ids(self.window.red_picks)[4], 2)

    def test_disabling_either_automation_switch_cancels_queued_work(self):
        for control in (self.window.chk_follow, self.window.chk_auto_calculate):
            self.enable_without_io()
            self.window.chk_auto_calculate.setChecked(True)
            self.window._last_draft_action = "predict"
            self.window._schedule_auto_calculation()
            control.setChecked(False)
            with patch.object(self.window, "_on_predict") as predict:
                self.window._run_auto_calculation()
                predict.assert_not_called()
            self.assertFalse(self.window._auto_calculation_pending)

    def test_loading_corrects_enemy_roles_and_recalculates_once(self):
        self.enable_without_io()
        self.window._receive_followed_draft(fixtures.snapshot())
        self.window._last_draft_action = "predict"
        current = fixtures.snapshot()
        final = replace(current, phase="GameStart", picks=tuple(
            replace(pick, role=0 if pick.internal_id == 1 else pick.role,
                    role_confirmed=pick.internal_id == 1) for pick in current.picks))
        with patch.object(self.window, "_on_predict") as predict:
            self.window._receive_followed_draft(final)
            self.window._run_auto_calculation()
            predict.assert_called_once()
            self.assertEqual(self.window._get_ids(self.window.red_picks)[0], 1)
            self.assertIn("kliens", self.window.red_role_labels[0].text())
            self.window._receive_followed_draft(final)
            self.assertFalse(self.window._auto_calculation_pending)

    def test_layout_toggle_preserves_draft_results_and_automation(self):
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window._last_draft_action = "recommend"
        self.window.lbl_dash_title.setText("Meglévő eredmény")
        for compact in (False, True, False, True):
            self.window.chk_compact_hud.setChecked(compact)
            self.assertEqual(self.window._get_ids(self.window.blue_picks)[0], 0)
            self.assertEqual(self.window.lbl_dash_title.text(), "Meglévő eredmény")
            self.assertEqual(self.window._last_draft_action, "recommend")
            self.assertFalse(self.window.chk_follow.isChecked())
            self.assertEqual(self.window.action_row.indexOf(self.window.btn_predict) >= 0, compact)


if __name__ == "__main__":
    unittest.main()
