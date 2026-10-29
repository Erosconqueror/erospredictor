"""Manual role swaps retain both picks and survive subsequent client updates."""

from dataclasses import replace
import os
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from services.draft_assignment import assign_draft
from services.draft_import import DraftPick
from test_main_window_draft_sync import distributions, snapshot
from ui_test_support import isolate_preferences
from view.main_window import MainWindow


class RoleSwapWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        isolate_preferences(self)
        names = ["Aatrox", "Xerath", "Braum", "Ahri", "Katarina", "Nautilus"]
        self.window = MainWindow(names, {name: index for index, name in enumerate(names)})
        self.window.controller = MagicMock()
        roles = patch.object(self.window, "_load_roles", return_value=(distributions(), False))
        roles.start()
        self.addCleanup(roles.stop)
        self.addCleanup(self.window.close)

    def follow(self, current=None):
        self.window.chk_follow.blockSignals(True)
        self.window.chk_follow.setChecked(True)
        self.window.chk_follow.blockSignals(False)
        current = current or snapshot()
        self.window._receive_followed_draft(current)
        return current

    def picks(self, team="red"):
        return self.window._get_ids(self.window.red_picks if team == "red" else self.window.blue_picks)

    def test_menu_swaps_two_picks_and_invalidates_results_without_inference(self):
        self.window._apply_import({"blue": [0], "red": [None, None, 1, None, 2], "blue_bans": [3]})
        self.window._prediction_context = {"old": True}
        self.window._recommendation_context = {"old": True}
        self.window._effects_context = {"old": True}
        self.window.red_effects[2].setText("↑ 2 pp")
        focus = (self.window.grp_team.checkedId(), self.window.grp_role.checkedId())
        menu = self.window._pick_role_menu("red", 2)
        action = next(action for action in menu.actions() if action.data() == 4)
        self.assertIn("Braum", action.text())
        action.trigger()
        menu.deleteLater()
        self.assertEqual(self.picks(), [-1, -1, 2, -1, 1])
        self.assertEqual(self.picks("blue"), [0, -1, -1, -1, -1])
        self.assertEqual(self.window._get_ids(self.window.blue_bans), [3, -1, -1, -1, -1])
        for role, name in ((2, "Braum"), (4, "Xerath")):
            combo = self.window.red_picks[role]
            self.assertEqual(combo.currentIndex(), combo.findText(name))
            self.assertIn("kézi", self.window.red_role_labels[role].text())
        self.assertEqual(self.window._sync_manual_roles, {("red", 1): 4, ("red", 2): 2})
        self.assertEqual((self.window.grp_team.checkedId(), self.window.grp_role.checkedId()), focus)
        self.assertIsNone(self.window._prediction_context)
        self.assertIsNone(self.window._recommendation_context)
        self.assertIsNone(self.window._effects_context)
        self.assertEqual(self.window.red_effects[2].text(), "")
        self.assertFalse(self.window.btn_explain.isEnabled())
        self.assertEqual(self.window.controller.mock_calls, [])

    def test_context_menu_is_reachable_from_the_editable_name_and_role_label(self):
        import gc
        from PyQt6.QtCore import QPoint, QTimer
        from PyQt6.QtGui import QContextMenuEvent
        self.window.show()
        self.app.processEvents()
        gc.collect()  # C++-created line edit wrappers must keep their signal connections.
        for widget in (self.window.red_picks[2].lineEdit(), self.window.red_role_labels[2]):
            with self.subTest(widget=type(widget).__name__):
                self.window._apply_import({"red": [None, None, 1, None, 2]})
                selected = []
                timer = QTimer(self.window)
                def choose():
                    menu = self.app.activePopupWidget()
                    if menu is not None:
                        timer.stop()
                        for action in menu.actions():
                            if action.data() == 4:
                                action.trigger()
                                selected.append(True)
                                break
                        menu.close()
                timer.timeout.connect(choose)
                timer.start(10)
                point = QPoint(5, 5)
                event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, point, widget.mapToGlobal(point))
                self.app.sendEvent(widget, event)
                self.app.processEvents()
                timer.stop()
                self.assertEqual(selected, [True])
                self.assertEqual(self.picks(), [-1, -1, 2, -1, 1])

    def test_move_to_empty_role_handles_internal_id_zero_on_both_sides(self):
        for team in ("blue", "red"):
            with self.subTest(team=team):
                self.window._on_clear_all()
                self.window._apply_import({team: [0]})
                self.assertTrue(self.window._swap_pick_roles(team, 0, 3))
                self.assertEqual(self.picks(team), [-1, -1, -1, 0, -1])
                combos = self.window.blue_picks if team == "blue" else self.window.red_picks
                self.assertEqual(combos[0].currentIndex(), 0)
                self.assertEqual(self.window._sync_manual_roles, {(team, 0): 3})

    def test_live_swap_replaces_both_old_manual_roles_and_survives_new_picks_and_bans(self):
        current = self.follow()
        self.window._sync_manual_roles = {("red", 1): 2, ("red", 2): 4}
        self.assertTrue(self.window._swap_pick_roles("red", 2, 4))
        self.assertEqual(self.picks(), [-1, -1, 2, -1, 1])
        self.window._receive_followed_draft(current)
        updated = replace(current, picks=current.picks + (DraftPick("red", 103, 3, None, 7),), own_bans=(4,))
        self.window._receive_followed_draft(updated)
        self.assertEqual(self.picks()[2], 2)
        self.assertEqual(self.picks()[4], 1)
        self.assertIn("kézi", self.window.red_role_labels[2].text())
        self.assertIn("kézi", self.window.red_role_labels[4].text())
        self.assertIn("Mid: 0.0%", self.window.red_picks[2].toolTip())
        self.assertEqual(self.window._get_ids(self.window.blue_bans)[0], 4)
        self.assertEqual(self.window.controller.mock_calls, [])
        self.window._receive_followed_draft(replace(current, session_key="game:101"))
        self.assertEqual(self.picks(), [-1, -1, 1, -1, 2])
        self.assertEqual(self.window._sync_manual_roles, {})

    def test_live_move_preserves_other_manual_roles_and_can_be_swapped_back(self):
        current = self.follow()
        self.window._sync_manual_roles[("red", 2)] = 4
        self.assertTrue(self.window._swap_pick_roles("red", 2, 0))
        self.assertEqual(self.picks(), [1, -1, -1, -1, 2])
        self.assertTrue(self.window._swap_pick_roles("red", 0, 4))
        self.assertEqual(self.picks(), [2, -1, -1, -1, 1])
        self.window._receive_followed_draft(replace(current, own_bans=(4,)))
        self.assertEqual(self.picks(), [2, -1, -1, -1, 1])

    def test_blue_enemy_and_relative_client_teams_support_the_same_swap(self):
        current = snapshot()
        picks = tuple(replace(pick, team="own" if pick.team == "blue" else "enemy") for pick in current.picks)
        current = self.follow(replace(current, own_team="red", picks=picks))
        self.assertEqual(self.picks("blue"), [-1, -1, 1, -1, 2])
        self.assertTrue(self.window._swap_pick_roles("blue", 2, 4))
        self.window._receive_followed_draft(replace(current, own_bans=(4,)))
        self.assertEqual(self.picks("blue"), [-1, -1, 2, -1, 1])
        self.assertEqual(self.picks("red")[0], 0)

    def test_manual_swaps_also_constrain_optional_alternative_enemy_lineups(self):
        current = self.follow()
        self.assertTrue(self.window._swap_pick_roles("red", 2, 4))
        result = assign_draft(current, distributions(), previous_roles=self.window._sync_manual_roles)
        self.assertTrue(result.scenarios)
        self.assertTrue(all(plan.red[2] == 2 and plan.red[4] == 1 for plan in result.scenarios))

    def test_live_own_client_roles_remain_authoritative_without_losing_picks(self):
        self.follow()
        before = self.picks("blue")
        self.assertFalse(self.window._swap_pick_roles("blue", 0, 2))
        self.assertEqual(self.picks("blue"), before)
        self.assertEqual(self.window._sync_manual_roles, {})
        self.assertIn("kliens", self.window.lbl_draft_sync.toolTip())

    def test_busy_incomplete_names_and_empty_sources_leave_the_draft_untouched(self):
        self.window.red_picks[2].setCurrentText("Xerath")
        self.window.red_picks[4].setEditText("Bra")
        self.assertFalse(self.window._swap_pick_roles("red", 2, 4))
        self.assertEqual(self.window.red_picks[4].currentText(), "Bra")
        self.window.red_picks[4].setCurrentIndex(0)
        self.window._set_busy(True)
        self.assertFalse(self.window._swap_pick_roles("red", 2, 4))
        self.window._set_busy(False)
        self.assertFalse(self.window._swap_pick_roles("red", 0, 2))
        self.assertFalse(self.window._swap_pick_roles("red", 2, 2))
        self.assertEqual(self.picks(), [-1, -1, 1, -1, -1])
        self.assertEqual(self.window._sync_manual_roles, {})

    def test_an_open_menu_cannot_swap_different_heroes_after_a_client_update(self):
        self.follow(snapshot(False))
        menu = self.window._pick_role_menu("red", 4)
        action = next(action for action in menu.actions() if action.data() == 2)
        self.window._receive_followed_draft(snapshot())
        action.trigger()
        menu.deleteLater()
        self.assertEqual(self.picks(), [-1, -1, 1, -1, 2])
        self.assertEqual(self.window._sync_manual_roles, {})
        self.assertIn("megváltozott", self.window.lbl_draft_sync.toolTip())

    def test_failed_live_assignment_does_not_commit_half_a_swap(self):
        self.follow()
        self.window.red_picks[2].setCurrentText("Ahri")  # Local edit not yet present in the client.
        before = self.picks()
        context = {"valid": True}
        self.window._prediction_context = context
        self.assertFalse(self.window._swap_pick_roles("red", 2, 4))
        self.assertEqual(self.picks(), before)
        self.assertEqual(self.window._sync_manual_roles, {})
        self.assertIs(self.window._prediction_context, context)
        self.assertIn("eltér", self.window.lbl_draft_sync.toolTip())


if __name__ == "__main__":
    unittest.main()
