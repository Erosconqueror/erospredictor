"""Real temporary preferences across windows; no real client or user files."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from services import client_settings as settings
from services import ui_preferences as preferences
from services import champion_pools as pools


class PreferenceFixture(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "settings.json"
        location = patch.object(settings, "settings_path", return_value=self.path)
        location.start()
        self.addCleanup(location.stop)


class PreferenceStorageTests(PreferenceFixture):
    def test_ui_changes_merge_with_lockfile_and_unrelated_preferences(self):
        self.path.write_text('{"theme":"dark"}')
        settings.save_lockfile_path(Path(self.folder.name) / "League/lockfile")
        self.assertTrue(preferences.save_ui_preferences({"auto_draft": True, "off_meta": False}))
        preferences.save_ui_preferences({"simple_effects": True})
        settings.save_lockfile_path(Path(self.folder.name) / "Moved/lockfile")
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["theme"], "dark")
        self.assertIn("Moved", saved["lcu_lockfile"])
        self.assertEqual(preferences.load_ui_preferences(), {"auto_draft": True, "off_meta": False, "simple_effects": True})

    def test_invalid_values_and_credentials_do_not_become_ui_preferences(self):
        preferences.save_ui_preferences({"auto_draft": "false", "off_meta": 1, "history_limit": True,
            "preferred_role": 99, "sync_side": "enemy", "token": "private-token", "active_pool": "Mid"})
        self.assertEqual(preferences.load_ui_preferences(), {"active_pool": "Mid"})
        self.assertNotIn("private-token", self.path.read_text())

    def test_corruption_and_failed_atomic_write_leave_safe_state(self):
        self.path.write_text("broken")
        self.assertEqual(preferences.load_ui_preferences(), {})
        preferences.save_ui_preferences({"auto_draft": True})
        with patch.object(settings.os, "replace", side_effect=PermissionError("locked")):
            self.assertFalse(preferences.save_ui_preferences({"auto_draft": False}))
        self.assertEqual(preferences.load_ui_preferences()["auto_draft"], True)
        self.assertEqual(list(self.path.parent.glob("settings-*.tmp")), [])

    def test_parallel_client_and_gui_updates_preserve_each_others_sections(self):
        from concurrent.futures import ThreadPoolExecutor
        def location():
            for i in range(10):
                settings.save_lockfile_path(Path(self.folder.name) / f"League{i}/lockfile")
        def ui():
            for i in range(10):
                preferences.save_ui_preferences({"history_limit": 1000, "auto_draft": True})
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(location), executor.submit(ui)]
            for future in futures:
                future.result()
        self.assertTrue(settings.saved_lockfile().parent.name == "League9")
        self.assertEqual(preferences.load_ui_preferences(), {"history_limit": 1000, "auto_draft": True})


class PersistentWindowTests(PreferenceFixture):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        pool_path = patch.object(pools, "library_path", return_value=self.path.parent / "champion_pools.json")
        pool_path.start()
        self.addCleanup(pool_path.stop)
        def fake_follower(*args, **kwargs):
            follower = MagicMock()
            follower.running = False
            return follower
        follow = patch("view.main_window.DraftFollower", side_effect=fake_follower)
        follow.start()
        self.addCleanup(follow.stop)

    def window(self):
        from view.main_window import MainWindow
        window = MainWindow(["Aatrox", "Ahri"], {"Aatrox": 0, "Ahri": 1})
        self.addCleanup(window.close)
        return window

    def wait(self, window):
        from PyQt6.QtTest import QTest
        for _ in range(400):
            if not window._busy:
                break
            QTest.qWait(10)
        self.assertFalse(window._busy)

    def test_all_choices_and_named_pool_return_in_a_fresh_window(self):
        pools.save_library({"version": 1, "champions": {"103": ["played", "comfort", "learn"]}, "lists": {"Mid": ["103"]}})
        first = self.window()
        first.chk_follow.setChecked(True)
        first.chk_meta.setChecked(False)
        first.chk_effects.setChecked(True)
        first.chk_robust.setChecked(True)
        first.chk_personal.setChecked(False)
        first.chk_history_service.setChecked(True)
        first.chk_pool_only.setChecked(True)
        first.cb_history_limit.setCurrentIndex(first.cb_history_limit.findData(1000))
        first.cb_sync_side.setCurrentIndex(first.cb_sync_side.findData("red"))
        first.cb_pool.setCurrentText("Mid")
        first.cb_rank.setCurrentText("GOLD")
        first.grp_role.button(2).click()
        first.close()
        self.assertTrue(preferences.load_ui_preferences()["auto_draft"])
        second = self.window()
        for control in (second.chk_follow, second.chk_effects, second.chk_robust, second.chk_history_service, second.chk_pool_only):
            self.assertTrue(control.isChecked())
        self.assertFalse(second.chk_meta.isChecked())
        self.assertFalse(second.chk_personal.isChecked())
        self.assertEqual(second.cb_history_limit.currentData(), 1000)
        self.assertEqual(second.cb_sync_side.currentData(), "red")
        self.assertEqual(second.cb_pool.currentText(), "Mid")
        self.assertEqual(second.cb_rank.currentText(), "GOLD")
        self.assertEqual(second.grp_role.checkedId(), 2)
        self.assertEqual(second._pool_library["champions"]["103"], ["comfort", "learn", "played"])
        second.show()
        self.app.processEvents()
        second._draft_follower.set_enabled.assert_called_with(True)

    def test_close_during_a_task_does_not_save_operational_follow_shutdown(self):
        from PyQt6.QtGui import QCloseEvent
        window = self.window()
        window.chk_follow.setChecked(True)
        window._set_busy(True)
        event = QCloseEvent()
        window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        window._set_busy(False)
        window.close()
        self.assertTrue(preferences.load_ui_preferences()["auto_draft"])

    def test_preferences_save_automatically_without_closing(self):
        from PyQt6.QtTest import QTest
        window = self.window()
        window.chk_meta.setChecked(False)
        QTest.qWait(350)
        self.assertFalse(preferences.load_ui_preferences()["off_meta"])

    def test_startup_restores_remote_choice_but_only_manual_refresh_can_use_it(self):
        from services.personal_history import PersonalHistory
        preferences.save_ui_preferences({"history_service": True})
        window = self.window()
        with patch("services.personal_history.LcuHistoryClient") as client:
            client.return_value.read_history.return_value = PersonalHistory("test", "Test", ())
            window._on_startup_history()
            self.wait(window)
            self.assertNotIn("allow_client_service", client.return_value.read_history.call_args.kwargs)
            self.assertTrue(window.chk_history_service.isChecked())
            window.btn_history.click()
            self.wait(window)
            self.assertTrue(client.return_value.read_history.call_args.kwargs["allow_client_service"])

    def test_failed_save_keeps_pending_changes_for_retry(self):
        window = self.window()
        window.chk_meta.setChecked(False)
        with patch("view.main_window.save_ui_preferences", return_value=False):
            self.assertFalse(window._flush_ui_preferences())
        self.assertEqual(window._pending_ui_preferences, {"off_meta": False})
        self.assertTrue(window._flush_ui_preferences())
        self.assertFalse(preferences.load_ui_preferences()["off_meta"])

    def test_played_champions_survive_in_the_account_cache_and_local_pool(self):
        from services import personal_history as history
        from services.update_progress import atomic_json
        from test_personal_history import ACCOUNT, profile, MAPPING
        with patch.object(history, "settings_path", return_value=self.path):
            first = history.LcuHistoryClient()
            source = profile(3)
            atomic_json(first._cache_path(ACCOUNT["puuid"]), {"version": 1, "puuid": ACCOUNT["puuid"], "games": source.games})
            cached = history.LcuHistoryClient()._cached(ACCOUNT)
        self.assertEqual(len(cached), 3)
        restored = history.PersonalHistory(ACCOUNT["puuid"], "Test", tuple(cached.values()))
        pools.save_library(pools.include_history(pools.empty_library(), restored, MAPPING))
        self.assertEqual(pools.load_library()["champions"], {"266": ["played"]})


if __name__ == "__main__":
    unittest.main()
