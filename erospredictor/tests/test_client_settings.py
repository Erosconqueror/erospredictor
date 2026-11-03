"""Per-computer client discovery and persistence; no real preferences are written."""

import json
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services import client_settings as settings
from services import draft_import as importer


class ClientLocationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.preferences = self.root / "preferences/settings.json"
        location = patch.object(settings, "settings_path", return_value=self.preferences)
        location.start()
        self.addCleanup(location.stop)
        environment = patch.dict(os.environ, {"EROSPREDICTOR_LCU_LOCKFILE": ""})
        environment.start()
        self.addCleanup(environment.stop)

    def lockfile(self, folder="League"):
        path = self.root / folder / "lockfile"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("LeagueClient:123:4567:temporary-auth-token:https", encoding="utf-8")
        return path

    def test_default_points_to_standard_c_drive_installation(self):
        self.assertEqual(settings.preferred_lockfile(), Path("C:/Riot Games/League of Legends/lockfile"))

    def test_path_roundtrips_without_storing_credentials(self):
        path = self.lockfile()
        self.assertTrue(settings.save_lockfile_path(path))
        self.assertEqual(settings.saved_lockfile(), path)
        self.assertEqual(settings.preferred_lockfile(), path)
        contents = self.preferences.read_text(encoding="utf-8")
        self.assertEqual(json.loads(contents), {"lcu_lockfile": str(path)})
        self.assertNotIn("temporary-auth-token", contents)
        self.assertNotIn("4567", contents)

    def test_saved_path_is_reused_without_expensive_machine_scan(self):
        path = self.lockfile()
        settings.save_lockfile_path(path)
        with patch.object(importer, "client_install_roots") as scan:
            self.assertEqual(importer.discover_lockfiles(), [path])
        scan.assert_not_called()

    def test_moved_install_falls_back_when_saved_file_disappears(self):
        old = self.root / "old/lockfile"
        settings.save_lockfile_path(old)
        actual = self.lockfile("new")
        with patch.object(importer, "DEFAULT_LOCKFILE", self.root / "absent/lockfile"), \
                patch.object(importer, "client_install_roots", return_value=[actual.parent]):
            self.assertEqual(importer.discover_lockfiles(), [actual])

    def test_installer_saves_location_when_client_is_closed(self):
        installation = self.root / "custom-drive/League of Legends"
        installation.mkdir(parents=True)
        (installation / "LeagueClient.exe").touch()
        with patch.object(importer, "DEFAULT_LOCKFILE", self.root / "absent/lockfile"), \
                patch.object(importer, "client_install_roots", return_value=[installation]):
            chosen, saved = importer.configure_client_location()
        self.assertTrue(saved)
        self.assertEqual(chosen, installation / "lockfile")
        self.assertEqual(settings.saved_lockfile(), chosen)
        self.assertFalse(chosen.exists())

    def test_reinstall_keeps_remembered_location_while_client_is_closed(self):
        path = self.lockfile("custom")
        settings.save_lockfile_path(path)
        path.unlink()
        with patch.object(importer, "DEFAULT_LOCKFILE", self.root / "absent/lockfile"), \
                patch.object(importer, "client_install_roots", return_value=[]):
            chosen, saved = importer.configure_client_location()
        self.assertTrue(saved)
        self.assertEqual(chosen, path)

    def test_installer_records_default_when_league_is_not_installed_yet(self):
        with patch.object(importer, "DEFAULT_LOCKFILE", self.root / "absent/lockfile"), \
                patch.object(importer, "client_install_roots", return_value=[]):
            chosen, saved = importer.configure_client_location()
        self.assertTrue(saved)
        self.assertEqual(chosen, self.root / "absent/lockfile")
        self.assertEqual(settings.saved_lockfile(), chosen)

    def test_successfully_read_lockfile_is_remembered_even_before_champ_select(self):
        path = self.lockfile()
        from unittest.mock import MagicMock
        transport = MagicMock()
        transport.get.return_value.status_code = 404
        with self.assertRaises(importer.DraftImportError):
            importer.LcuDraftClient(path, {}, transport).read_draft()
        self.assertEqual(settings.saved_lockfile(), path)

    def test_environment_override_has_priority_over_saved_path(self):
        path = self.lockfile("saved")
        override = self.lockfile("override")
        settings.save_lockfile_path(path)
        with patch.dict(os.environ, {"EROSPREDICTOR_LCU_LOCKFILE": str(override)}):
            self.assertEqual(settings.preferred_lockfile(), override)
            self.assertEqual(importer.discover_lockfiles(), [override])

    def test_corrupt_settings_and_write_failure_do_not_break_client_defaults(self):
        self.preferences.parent.mkdir()
        self.preferences.write_text("{invalid json", encoding="utf-8")
        self.assertEqual(settings.preferred_lockfile(), settings.DEFAULT_LOCKFILE)
        self.preferences.write_text(json.dumps({"lcu_lockfile": "bad\0path"}), encoding="utf-8")
        self.assertIsNone(settings.saved_lockfile())
        with patch.object(settings.os, "replace", side_effect=PermissionError("read-only preferences")):
            self.assertFalse(settings.save_lockfile_path(self.root / "League/lockfile"))
        self.assertEqual(list(self.preferences.parent.glob("settings-*.tmp")), [])

    def test_saving_location_preserves_other_preferences(self):
        self.preferences.parent.mkdir()
        self.preferences.write_text('{"theme":"dark"}', encoding="utf-8")
        settings.save_lockfile_path(self.root / "League/lockfile")
        self.assertEqual(json.loads(self.preferences.read_text(encoding="utf-8"))["theme"], "dark")

    @unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 is not installed")
    def test_manual_choice_is_reused_by_a_new_application_window(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from view.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        path = self.lockfile("manually-selected")
        first = MainWindow(["Aatrox"], {"Aatrox": 0})
        second = None
        try:
            with patch("view.main_window.QFileDialog.getOpenFileName", return_value=(str(path), "")):
                first._choose_lockfile()
            self.assertEqual(settings.saved_lockfile(), path)
            second = MainWindow(["Aatrox"], {"Aatrox": 0})
            with patch("view.main_window.QFileDialog.getOpenFileName", return_value=("", "")) as dialog:
                second._choose_lockfile()
            self.assertEqual(dialog.call_args.args[2], str(path))
            self.assertIn(str(path), second.btn_lockfile.toolTip())
        finally:
            first.close()
            if second is not None:
                second.close()


if __name__ == "__main__":
    unittest.main()
