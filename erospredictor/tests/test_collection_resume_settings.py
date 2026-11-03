"""Collection cursors survive harmless training/QA edits, without relaxing scope."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services.update_progress import UpdateProgress, file_digest


TRAINING_ONLY = ("golden_dataset.json", "training_config", "epochs", "qa_min_pass", "min_matches")


def request():
    return {"last_patches": 5, "end_patch": None, "patches": None,
            "tiers": ["GOLD", "SILVER"], "region": "euw1", "continent": "europe",
            "database": {"DB_NAME": "test", "DB_HOST": "localhost", "DB_PORT": 5432},
            "matches_per_tier": None, "tier_target": [], "matches_per_player": 50,
            "players_per_tier": None, "pages_per_division": None,
            "skip_scrape": False, "no_catalog": False,
            "epochs": None, "qa_min_pass": .25, "min_matches": 200,
            "training_config": {"role_weights": {"TOP": 1}, "division_weights": {"GOLD": 1}},
            "golden_dataset.json": "old-QA-hash"}


class CollectionResumeSettingsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.original = request()
        self.progress = UpdateProgress.open(self.root, self.original, lambda: ["16.17", "16.18", "16.19", "16.20", "16.21"])
        catalog = self.progress.staging / "catalog.json"
        catalog.write_text('{"mapping":{"266":0},"version":"16.21.1"}')
        self.progress.state.update(
            catalog_sha256=file_digest(catalog),
            opaque_metadata={"must": "survive"},
            collection={
                "GOLD": {"done": True, "new_matches": 22000},
                "SILVER": {"done": False, "new_matches": 15637, "window_matches": 15637,
                    "player_index": 114, "match_index": 22, "players": ["fixture-player"],
                    "visited": ["fixture-visited"], "examined_ids": ["fixture-completed-match"],
                    "pending": {"id": "fixture-pending-match", "data": {
                        "tier": "SILVER", "patch": "16.21", "blue_win": True,
                        "blue_team": [1, 2, 3, 4, 5], "red_team": [6, 7, 8, 9, 10]}}}})
        self.progress.save()

    def resume(self, current, **kwargs):
        return UpdateProgress.open(self.root, current,
            lambda: self.fail("A resumed collection must keep its original patch window"),
            collection_only_changes=TRAINING_ONLY, **kwargs)

    def test_golden_change_keeps_exact_run_catalog_pending_write_and_collection_cursor(self):
        before = deepcopy(self.progress.state)
        catalog = (self.progress.staging / "catalog.json").read_bytes()
        current = dict(self.original, **{"golden_dataset.json": "new-QA-hash"})
        resumed = self.resume(current)
        self.assertEqual(resumed.staging, self.progress.staging)
        for key in before:
            if key != "request":
                self.assertEqual(resumed.state[key], before[key], key)
        self.assertEqual(resumed.state["request"], current)
        self.assertEqual((resumed.staging / "catalog.json").read_bytes(), catalog)
        audit = json.dumps(resumed.state.get("collection_request_updates"))
        self.assertIn("golden_dataset.json", audit)
        self.assertNotIn("old-QA-hash", audit)
        self.assertNotIn("new-QA-hash", audit)
        self.assertEqual(len(list(self.root.glob("run-*"))), 1)
        persisted = json.loads((self.root / "state.json").read_text())
        self.assertEqual(persisted, resumed.state)
        again = self.resume(current)
        self.assertEqual(again.state, resumed.state)

    def test_each_allowed_training_setting_can_change_before_training_starts(self):
        changes = {"training_config": {"role_weights": {"TOP": 2}, "division_weights": {"GOLD": 1}},
                   "epochs": 20, "qa_min_pass": .5, "min_matches": 300}
        for key, value in changes.items():
            with self.subTest(setting=key):
                # Restore the same pristine temporary journal for each scenario.
                self.progress.save()
                resumed = self.resume(dict(self.original, **{key: value}))
                self.assertEqual(resumed.state["run_id"], self.progress.state["run_id"])
                self.assertEqual(resumed.state["collection"], self.progress.state["collection"])
                self.assertEqual(resumed.state["request"][key], value)

    def test_critical_or_unknown_parameter_changes_leave_journal_byte_identical(self):
        changes = {"region": "eun1", "continent": "asia", "database": {"DB_NAME": "other"},
            "last_patches": 4, "patches": ["16.21"], "end_patch": "16.20", "tiers": ["GOLD"],
            "matches_per_player": 100, "matches_per_tier": 10, "players_per_tier": 5,
            "pages_per_division": 2, "skip_scrape": True, "unclassified_new_setting": True}
        before = (self.root / "state.json").read_bytes()
        for key, value in changes.items():
            with self.subTest(setting=key):
                current = dict(self.original, **{"golden_dataset.json": "new-QA-hash", key: value})
                with self.assertRaisesRegex(ValueError, "different parameters"):
                    self.resume(current)
                self.assertEqual((self.root / "state.json").read_bytes(), before)
        self.assertEqual(len(list(self.root.glob("run-*"))), 1)

    def test_training_snapshot_or_publication_never_uses_collection_only_exception(self):
        stages = {"training": {"GOLD:gnn": {"path": "models/gold.pth"}},
                  "snapshot_sha256": "fixture-snapshot-digest",
                  "publication": ["models/gold.pth"]}
        for key, value in stages.items():
            with self.subTest(stage=key):
                old = deepcopy(self.progress.state)
                self.progress.state[key] = value
                self.progress.save()
                before = (self.root / "state.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "different parameters"):
                    self.resume(dict(self.original, **{"golden_dataset.json": "new-QA-hash"}))
                self.assertEqual((self.root / "state.json").read_bytes(), before)
                self.progress.state = old
                self.progress.save()

    def test_atomic_write_failure_preserves_saved_request_and_cursor(self):
        before = (self.root / "state.json").read_bytes()
        with patch("services.update_progress.os.replace", side_effect=PermissionError("fixture locked file")):
            with self.assertRaises(PermissionError):
                self.resume(dict(self.original, **{"golden_dataset.json": "new-QA-hash"}))
        self.assertEqual((self.root / "state.json").read_bytes(), before)
        self.assertEqual(list(self.root.glob(".checkpoint-*")), [])
        self.assertEqual(len(list(self.root.glob("run-*"))), 1)

    def test_collection_only_relaxation_requires_explicit_classification(self):
        before = (self.root / "state.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "different parameters"):
            UpdateProgress.open(self.root, dict(self.original, **{"golden_dataset.json": "new-QA-hash"}),
                                lambda: self.fail("Do not resolve patches"))
        self.assertEqual((self.root / "state.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
