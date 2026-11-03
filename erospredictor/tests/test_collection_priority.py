"""Scheduling and lowered goals preserve accumulated data and per-tier cursors."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import configs
import update_pipeline as pipeline
from services.dataset_profile import collection_order
from services.update_progress import UpdateProgress
from test_update_pipeline import FakeDB, record


class CollectionPriorityTests(unittest.TestCase):
    def test_priority_only_changes_order_and_rejects_unselected_tier(self):
        self.assertEqual(collection_order(["GOLD", "BRONZE", "IRON"], ["IRON", "IRON"]), ["IRON", "GOLD", "BRONZE"])
        self.assertEqual(collection_order(["GOLD", "BRONZE"], []), ["GOLD", "BRONZE"])
        with self.assertRaises(ValueError):
            collection_order(["GOLD"], ["IRON"])

    def test_saved_priority_collects_iron_before_missing_bronze_without_removing_saved_matches(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "BRONZE", "IRON",
            "--priority-tier", "IRON", "--tier-target", "BRONZE=3", "--tier-target", "IRON=2"])
        db = FakeDB([record("old-b1", tier="BRONZE"), record("old-b2", tier="BRONZE")])
        visited = []
        def players(tier, *args, **kwargs):
            visited.append(tier)
            return [{"puuid": tier}]
        riot = SimpleNamespace(get_league_exp_players=players,
            get_match_ids=lambda tier, limit: [tier + str(index) for index in range(3)],
            get_match_data=lambda match_id, tier: record(match_id, tier=tier))
        with tempfile.TemporaryDirectory() as folder:
            progress = UpdateProgress.open(folder, {}, lambda: ["16.7"])
            pipeline._save_collection_priority(args, progress, log=lambda _: None)
            resumed = UpdateProgress.open(folder, {}, lambda: self.fail("Patch window changed"))
            args.priority_tier = []  # The saved order survives an ordinary relaunch.
            result = pipeline.collect_matches(args, ["16.7"], riot, db, log=lambda _: None, progress=resumed)
        self.assertEqual(visited, ["IRON", "BRONZE"])
        self.assertEqual(db.saved, ["IRON0", "IRON1", "BRONZE0"])
        self.assertEqual({row["match_id"] for row in db.records}, {"old-b1", "old-b2", "IRON0", "IRON1", "BRONZE0"})
        self.assertEqual(result["BRONZE"]["window_matches"], 3)

    def test_lower_goals_keep_the_same_run_and_exact_saved_cursor_before_training(self):
        old_targets = {"SILVER": 20000, "BRONZE": 14000, "IRON": 5000}
        new_targets = {"SILVER": 18000, "BRONZE": 7000, "IRON": 5000}
        with tempfile.TemporaryDirectory() as folder:
            progress = UpdateProgress.open(folder, {"targets": old_targets}, lambda: ["16.21"])
            progress.state["catalog_sha256"] = "saved-catalog"
            progress.state["collection"] = {"BRONZE": dict(pipeline._collection_state(), window_matches=6356,
                new_matches=6356, player_index=12, match_index=8, active_player="fixture")}
            progress.save()
            before = deepcopy(progress.state)
            resumed = UpdateProgress.open(folder, {"targets": new_targets}, lambda: self.fail("New window"),
                                          collection_only_changes=("targets", "tier_target"))
        self.assertEqual(resumed.state["run_id"], before["run_id"])
        self.assertEqual(resumed.state["catalog_sha256"], before["catalog_sha256"])
        self.assertEqual(resumed.state["collection"], before["collection"])
        self.assertEqual(resumed.state["request"]["targets"], new_targets)

    def test_finished_silver_above_new_goal_is_not_scraped_or_deleted(self):
        rows = [record(str(index), tier="SILVER") for index in range(5)]
        db = FakeDB(rows)
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "SILVER", "--tier-target", "SILVER=3"])
        riot = SimpleNamespace(get_league_exp_players=lambda *a, **kw: self.fail("Silver should already be sufficient"))
        result = pipeline.collect_matches(args, ["16.7"], riot, db, log=lambda _: None)
        self.assertEqual(len(db.records), 5)
        self.assertEqual(db.saved, [])
        self.assertEqual(result["SILVER"]["window_matches"], 5)

    def test_old_published_profile_cannot_override_new_default_goals(self):
        old_profile = dict(configs.DATASET_TARGETS, SILVER=20000, BRONZE=14000)
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path.name == "runtime_config.json":
                return json.dumps({"DATASET_TARGETS": old_profile})
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", new=read):
            loaded = runpy.run_path(configs.__file__)
        self.assertEqual(loaded["DATASET_TARGETS"]["SILVER"], 18000)
        self.assertEqual(loaded["DATASET_TARGETS"]["BRONZE"], 7000)
        self.assertEqual(loaded["DATASET_TARGETS"]["IRON"], 5000)


if __name__ == "__main__":
    unittest.main()
