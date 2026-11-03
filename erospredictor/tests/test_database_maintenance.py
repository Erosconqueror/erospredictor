"""Retention boundaries and transactional backup/delete use only mocked DB rows."""

from datetime import datetime
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import database_maintenance as maintenance

VERSIONS = [f"16.{i}.1" for i in range(21, 0, -1)] + [f"15.{i}.1" for i in range(24, 0, -1)]
ROWS = [("old-1", "euw1", "GOLD", "15.19", True, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], datetime(2025, 1, 1))]


class RetentionPlanTests(unittest.TestCase):
    def test_exactly_twenty_six_distinct_released_patches_are_retained_across_years(self):
        plan = maintenance.retention_plan([("15.19", 5), ("15.20", 7), ("16.21", 10), ("26.1.1", 3)], VERSIONS + ["16.21.2"])
        self.assertEqual(plan["cutoff_inclusive"], "15.20")
        self.assertEqual(len(plan["kept_patches"]), 26)
        self.assertEqual(plan["candidate_count"], 5)
        self.assertEqual(plan["old_patches"], [{"patch": "15.19", "matches": 5}])

    def test_unknown_patch_records_are_preserved_and_public_year_aliases_are_normalized(self):
        plan = maintenance.retention_plan([("UNKNOWN", 2), (None, 3), ("25.19", 4), ("16.22", 1)], VERSIONS)
        self.assertEqual(plan["candidate_count"], 4)
        self.assertEqual(len(plan["preserved_unknown"]), 2)
        self.assertEqual(plan["total_matches"], 10)

    def test_incomplete_or_untrusted_release_lists_cannot_authorize_deletion(self):
        for versions in ([], ["16.21.1"] * 40, ["invalid"] * 40):
            with self.assertRaises(ValueError):
                maintenance.retention_plan([("15.1", 100)], versions)
        with self.assertRaises(ValueError):
            maintenance.retention_plan([], VERSIONS, keep=0)


class RetentionTransactionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.backups = Path(folder.name)
        self.connection = MagicMock()
        self.cursor = self.connection.cursor.return_value.__enter__.return_value
        self.cursor.fetchall.side_effect = [[("15.19", 1), ("16.21", 5)], ROWS]
        self.cursor.rowcount = 1

    def run_maintenance(self, **kwargs):
        return maintenance.maintain(self.connection, VERSIONS, backup_directory=self.backups, **kwargs)

    def test_default_is_readonly_preview_without_deletes_backups_or_commits(self):
        result = self.run_maintenance()
        self.connection.set_session.assert_called_once_with(readonly=True, autocommit=False)
        self.connection.commit.assert_not_called()
        self.assertEqual(result["plan"]["candidate_count"], 1)
        self.assertEqual(result["deleted"], 0)
        self.assertFalse(any("DELETE" in call.args[0] for call in self.cursor.execute.call_args_list))
        self.assertEqual(list(self.backups.iterdir()), [])

    def test_apply_locks_exact_old_rows_then_verifies_backup_before_deleting_and_committing(self):
        def execute(query, params=()):
            if query.startswith("DELETE"):
                archives = list(self.backups.glob("*.json.gz"))
                self.assertEqual(len(archives), 1)
                with gzip.open(archives[0], "rt", encoding="utf-8") as file:
                    saved = json.load(file)
                self.assertEqual(saved["row_count"], 1)
                self.assertEqual(saved["rows"][0][:7], list(ROWS[0][:7]))
                self.assertEqual(params, (["old-1"], ["15.19"]))
        self.cursor.execute.side_effect = execute
        result = self.run_maintenance(apply=True, region="euw1")
        self.assertEqual(result["deleted"], 1)
        self.assertTrue(Path(result["backup"]).is_file())
        self.assertTrue(any("FOR UPDATE" in call.args[0] for call in self.cursor.execute.call_args_list))
        self.connection.commit.assert_called_once()
        self.connection.rollback.assert_not_called()

    def test_backup_failure_prevents_all_destructive_sql(self):
        with patch.object(maintenance, "write_backup", side_effect=OSError("full disk")):
            with self.assertRaises(OSError):
                self.run_maintenance(apply=True)
        self.assertFalse(any("DELETE" in call.args[0] for call in self.cursor.execute.call_args_list))
        self.connection.rollback.assert_called_once()
        self.connection.commit.assert_not_called()

    def test_delete_count_mismatch_rolls_back_and_preserves_the_archive(self):
        self.cursor.rowcount = 0
        with self.assertRaisesRegex(ValueError, "eltér"):
            self.run_maintenance(apply=True)
        self.connection.commit.assert_not_called()
        self.connection.rollback.assert_called_once()
        self.assertEqual(len(list(self.backups.glob("*.json.gz"))), 1)

    def test_manual_interruption_rolls_back_the_whole_delete(self):
        def execute(query, params=()):
            if query.startswith("DELETE"):
                raise KeyboardInterrupt()
        self.cursor.execute.side_effect = execute
        with self.assertRaises(KeyboardInterrupt):
            self.run_maintenance(apply=True)
        self.connection.rollback.assert_called_once()
        self.connection.commit.assert_not_called()

    def test_no_old_rows_never_creates_a_backup_or_issues_delete(self):
        self.cursor.fetchall.side_effect = [[("16.3", 400000)]]
        result = self.run_maintenance(apply=True)
        self.assertEqual(result["deleted"], 0)
        self.assertIsNone(result["backup"])
        self.assertEqual(list(self.backups.iterdir()), [])
        self.assertFalse(any("DELETE" in call.args[0] for call in self.cursor.execute.call_args_list))


if __name__ == "__main__":
    unittest.main()
