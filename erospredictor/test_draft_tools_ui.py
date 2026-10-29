"""New draft tools stay optional, local and separate from the core prediction."""

from copy import deepcopy
from dataclasses import replace
import os
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from services.champion_pools import ALL_MINE
from services.draft_import import DraftSnapshot, DraftPick, parse_draft_session
from services.draft_assignment import PickAssignment
from view.main_window import MainWindow
from view.champion_pool_dialog import ChampionPoolDialog


MAPPING = {"266": 0, "103": 1, "55": 2, "201": 3, "20": 4}
NAMES = {0: "Aatrox", 1: "Ahri", 2: "Katarina", 3: "Braum", 4: "Nunu"}
LIBRARY = {"version": 1, "champions": {"266": ["played"], "103": ["comfort", "learn"]}, "lists": {"Mid": ["103"]}}


class DraftToolsWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from ui_test_support import isolate_preferences
        isolate_preferences(self)
        with patch("view.main_window.load_library", return_value=deepcopy(LIBRARY)):
            self.window = MainWindow(list(NAMES.values()), {name: cid for cid, name in NAMES.items()})
        self.window._pool_mapping = MAPPING
        self.controller = MagicMock()
        self.controller.validate_draft.return_value = (True, "")
        self.controller.recommend_champs.return_value = [{"id": 1, "wr": 55.25, "display_color": "#fff", "pool_tags": ("comfort", "learn")}]
        self.window.controller = self.controller
        self.addCleanup(self.close)

    def wait(self):
        from PyQt6.QtTest import QTest
        for _ in range(400):
            if not self.window._busy and not self.window._draft_follower.running:
                break
            QTest.qWait(10)
        QTest.qWait(5)
        self.assertFalse(self.window._busy or self.window._draft_follower.running)

    def close(self):
        self.window.chk_follow.setChecked(False)
        self.wait()
        self.window.close()

    def test_filters_are_opt_in_and_history_controls_are_in_settings(self):
        self.assertFalse(self.window.chk_pool_only.isChecked())
        self.assertFalse(self.window.chk_robust.isChecked())
        self.assertFalse(self.window.options_dialog.isVisible())
        self.assertTrue(self.window.options_dialog.isAncestorOf(self.window.chk_personal))
        self.assertTrue(self.window.options_dialog.isAncestorOf(self.window.chk_meta))
        self.assertTrue(self.window.options_dialog.isAncestorOf(self.window.btn_lockfile))
        self.window._on_recommend()
        self.wait()
        self.controller.draft_scenarios.assert_not_called()
        self.assertNotIn("candidate_ids", self.controller.recommend_champs.call_args.kwargs)
        self.assertNotIn("scenarios", self.controller.recommend_champs.call_args.kwargs)

    def test_active_list_limits_candidates_and_annotations_leave_percentage_alone(self):
        self.window.cb_pool.setCurrentText("Mid")
        self.window.chk_pool_only.setChecked(True)
        self.window._on_recommend()
        self.wait()
        self.assertEqual(self.controller.recommend_champs.call_args.kwargs["candidate_ids"], {1})
        self.assertIn("55.25%", self.window.lbl_recommend.text())
        self.assertIn("Komfort", self.window.lbl_recommend.text())
        self.assertIn("Tanulnám", self.window.lbl_recommend.text())

    def test_empty_filtered_pool_never_falls_back_to_all_champions(self):
        self.window._pool_library = {"version": 1, "champions": {}, "lists": {}}
        self.window.chk_pool_only.setChecked(True)
        self.window._on_recommend()
        self.controller.recommend_champs.assert_not_called()
        self.assertIn("üres", self.window.lbl_dash_title.text())

    def test_robust_mode_explicitly_requests_scenarios_in_worker(self):
        self.window.chk_robust.setChecked(True)
        self.controller.draft_scenarios.return_value = ()
        self.window._on_recommend()
        self.wait()
        self.controller.draft_scenarios.assert_called_once()
        self.assertEqual(self.controller.recommend_champs.call_args.kwargs["scenarios"], ())

    def test_ban_advice_uses_selected_pool_without_prediction_or_history_requests(self):
        self.window.cb_pool.setCurrentText("Mid")
        self.controller.recommend_bans.return_value = []
        self.window._on_ban_advice()
        self.wait()
        self.assertEqual(self.controller.recommend_bans.call_args.args[2], {1})
        self.controller.predict_match.assert_not_called()
        self.controller.recommend_champs.assert_not_called()
        self.assertIn("20 meccset", self.window.lbl_dash_title.text())

    def test_unknown_side_can_be_resolved_without_stopping_the_follower(self):
        self.window.chk_follow.blockSignals(True)
        self.window.chk_follow.setChecked(True)
        self.window.chk_follow.blockSignals(False)
        snapshot = DraftSnapshot((DraftPick("own", 266, 0, 0, 7),), session_key="test:one")
        self.window._receive_followed_draft(snapshot)
        self.assertIn("oldala", self.window.lbl_draft_sync.text())
        self.window.cb_sync_side.setCurrentIndex(self.window.cb_sync_side.findData("red"))
        self.assertEqual(self.window._get_ids(self.window.red_picks)[0], 0)

    def test_manual_review_keeps_fallback_side_editable_and_maps_relative_picks(self):
        from view.draft_import_dialog import DraftImportDialog
        self.window.cb_sync_side.setCurrentIndex(self.window.cb_sync_side.findData("red"))
        snapshot = DraftSnapshot((DraftPick("own", 266, 0, None, 7),), session_key="test:manual")
        captured = []
        def inspect(dialog):
            captured.append((dialog.own_side.currentData(), dialog.own_side.isEnabled(), dialog.team_selectors[0].currentData()))
            return 0
        with patch.object(DraftImportDialog, "exec", new=inspect):
            self.window._review_import(snapshot)
        self.assertEqual(captured, [("red", True, "red")])

    def test_live_normal_draft_shape_is_applied_by_real_background_follower(self):
        # Sanitized shape captured from the user's actual queue-400 session:
        # teamId absent, team=2, gameflow rosters empty, local pick not locked yet.
        mapping = {"34": 1, "136": 2, "20": 4}
        own = [{"cellId": i + 5, "team": 2, "championId": champion, "assignedPosition": role}
               for i, (champion, role) in enumerate(((34, "utility"), (136, "bottom"), (0, "jungle"), (0, "middle"), (0, "top")))]
        enemy = [{"cellId": i, "team": 1, "championId": 20 if i == 0 else 0} for i in range(5)]
        actions = [{"type": "pick", "actorCellId": cell, "championId": champion, "completed": True, "isAllyAction": ally}
                   for cell, champion, ally in ((5, 34, True), (6, 136, True), (0, 20, False))]
        snapshot = parse_draft_session({"myTeam": own, "theirTeam": enemy, "actions": [actions], "localPlayerCellId": 7},
                                      mapping, {"gameData": {"queue": {"id": 400}, "teamOne": [], "teamTwo": []}})
        entry = {"counts": [0, 100, 0, 0, 0], "probabilities": [0, 1, 0, 0, 0], "total": 100}
        roles = {"version": 1, "roles": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], "tiers": {"MIXED": {"20": entry}}}
        with patch("services.draft_import.LcuDraftClient") as client, patch.object(self.window, "_load_roles", return_value=(roles, False)):
            client.return_value.read_draft.return_value = snapshot
            self.window.chk_follow.setChecked(True)
            self.wait()
        self.assertEqual(self.window._get_ids(self.window.red_picks), [-1, -1, -1, 2, 1])
        self.assertEqual(self.window._get_ids(self.window.blue_picks), [-1, 4, -1, -1, -1])
        self.assertTrue(self.window.btn_red.isChecked())
        self.assertEqual(self.window.grp_role.checkedId(), 1)
        self.assertIn("Normal draft", self.window.lbl_draft_sync.text())

    def test_forced_position_is_not_labeled_as_manual_or_confident_statistics(self):
        self.window.red_picks[0].setCurrentText("Ahri")
        self.window._show_role_metadata((PickAssignment("red", 1, 103, 0, "forced", None, 0, ()),))
        self.assertIn("kényszer", self.window.red_role_labels[0].text())
        self.assertIn("5%", self.window.red_picks[0].toolTip())


class PoolEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def test_overlapping_tags_and_custom_membership_save_together(self):
        document = deepcopy(LIBRARY)
        dialog = ChampionPoolDialog(document, MAPPING, NAMES)
        self.addCleanup(dialog.close)
        row = next(index for index, (riot, _) in enumerate(dialog.rows) if riot == "103")
        dialog.table.item(row, 1).setCheckState(Qt.CheckState.Checked)
        dialog.lists.setCurrentText("Mid")
        dialog.table.item(row, 4).setCheckState(Qt.CheckState.Checked)
        with patch("view.champion_pool_dialog.save_library") as save:
            dialog._save()
        saved = save.call_args.args[0]
        self.assertEqual(set(saved["champions"]["103"]), {"played", "comfort", "learn"})
        self.assertEqual(saved["lists"]["Mid"], ["103"])
        self.assertEqual(document, LIBRARY)

    def test_cancelling_editor_does_not_modify_or_save_library(self):
        document = deepcopy(LIBRARY)
        dialog = ChampionPoolDialog(document, MAPPING, NAMES)
        with patch("view.champion_pool_dialog.save_library") as save:
            dialog.reject()
        save.assert_not_called()
        self.assertEqual(document, LIBRARY)

    def test_failed_save_keeps_changes_available_for_retry(self):
        dialog = ChampionPoolDialog(deepcopy(LIBRARY), MAPPING, NAMES)
        self.addCleanup(dialog.close)
        with patch("view.champion_pool_dialog.save_library", side_effect=OSError("Access denied")):
            dialog._save()
        self.assertIn("nem menthető", dialog.status.text())
        self.assertEqual(dialog.result(), 0)


if __name__ == "__main__":
    unittest.main()
