"""Population quotas, frozen samples and safe adoption of older collections."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import configs
import update_pipeline as pipeline
from services.dataset_profile import balanced_snapshot, rank_targets
from services.update_progress import UpdateProgress
from test_update_pipeline import FakeDB, record, install_test_catalog


class ProfileTests(unittest.TestCase):
    def test_approved_default_profile_and_overrides(self):
        args = pipeline.parser().parse_args([])
        targets = rank_targets(args, configs.DATASET_TARGETS)
        self.assertEqual(targets, {"CHALLENGER": 2000, "GRANDMASTER": 3000, "MASTER": 5000,
            "DIAMOND": 8000, "EMERALD": 12000, "PLATINUM": 16000, "GOLD": 22000,
            "SILVER": 20000, "BRONZE": 14000, "IRON": 5000})
        self.assertEqual(sum(targets.values()), 107000)
        self.assertEqual(args.last_patches, 5)
        self.assertIsNone(args.players_per_tier)
        self.assertIsNone(args.pages_per_division)
        args.tier_target = ["GOLD=24000"]
        self.assertEqual(rank_targets(args, targets)["GOLD"], 24000)
        for invalid in ("GOLD", "GOLD=0", "OTHER=100", "GOLD=abc"):
            args.tier_target = [invalid]
            with self.assertRaises(ValueError):
                rank_targets(args, targets)

    def test_snapshot_caps_large_historical_tiers_and_spreads_over_five_patches(self):
        patches = [f"16.{i}" for i in range(17, 22)]
        rows = [record(f"{tier}-{version}-{i}", tier=tier, patch_version=version)
                for tier, size in (("MASTER", 50), ("GOLD", 20), ("SILVER", 20))
                for version in patches for i in range(size)]
        before = deepcopy(rows)
        targets = {"MASTER": 5, "GOLD": 22, "SILVER": 20}
        selected = balanced_snapshot(rows, targets, patches)
        self.assertEqual(Counter(row["tier"] for row in selected), targets)
        for tier in targets:
            self.assertEqual({row["patch"] for row in selected if row["tier"] == tier}, set(patches))
        self.assertEqual(selected, balanced_snapshot(list(reversed(rows)), targets, patches))
        self.assertEqual(rows, before)

    def test_sparse_patches_redistribute_available_samples(self):
        rows = [record("old", patch_version="16.6")] + [record(str(i)) for i in range(10)]
        selected = balanced_snapshot(rows, {"GOLD": 8}, ["16.5", "16.6", "16.7"])
        self.assertEqual(Counter(row["patch"] for row in selected), {"16.6": 1, "16.7": 7})
        self.assertEqual(len({row["match_id"] for row in selected}), 8)

    def test_profile_upgrade_keeps_cursor_but_invalidates_old_training_snapshot(self):
        previous = {"last_patches": 5, "matches_per_tier": 2000,
                    "players_per_tier": 200, "pages_per_division": 2,
                    "champion_id.json": "installed hash", "training_config": {"champion_count": 172}}
        current = {"last_patches": 5, "matches_per_tier": None, "players_per_tier": None,
                   "pages_per_division": None, "tier_target": [], "no_catalog": False,
                   "targets": configs.DATASET_TARGETS, "training_config": {}}
        with tempfile.TemporaryDirectory() as directory:
            first = UpdateProgress.open(directory, previous, lambda: ["16.17", "16.21"])
            cursor = dict(pipeline._collection_state(), new_matches=701, page=4, active_player="active")
            first.state.update(collection={"GRANDMASTER": cursor}, snapshot_sha256="obsolete")
            first.state["training"]["old"] = {"sha256": "obsolete"}
            first.save()
            resumed = UpdateProgress.open(directory, current, lambda: self.fail("Window changed"),
                                          upgrade=pipeline.compatible_profile_upgrade)
            self.assertEqual(resumed.state["patches"], first.state["patches"])
            self.assertEqual(resumed.state["collection"]["GRANDMASTER"], cursor)
            self.assertEqual(resumed.state["training"], {})
            self.assertNotIn("snapshot_sha256", resumed.state)
            self.assertEqual(resumed.state["upgraded_from"], first.state["run_id"])
            changed = dict(current, last_patches=3)
            with self.assertRaisesRegex(ValueError, "different parameters"):
                UpdateProgress.open(directory, changed, lambda: [], upgrade=pipeline.compatible_profile_upgrade)

    def test_profile_upgrade_keeps_sources_for_interrupted_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = {"last_patches": 5, "matches_per_tier": 2000}
            current = {"last_patches": 5, "matches_per_tier": None, "targets": configs.DATASET_TARGETS}
            progress = UpdateProgress.open(directory, previous, lambda: ["16.21"])
            progress.state["publication"] = ["models/ready.pth"]
            progress.state["publication_sha256"] = {"models/ready.pth": "digest"}
            progress.save()
            resumed = UpdateProgress.open(directory, current, lambda: [], upgrade=pipeline.compatible_profile_upgrade)
            self.assertEqual(resumed.state["run_id"], progress.state["run_id"])
            self.assertEqual(resumed.state["publication_sha256"], progress.state["publication_sha256"])


class ProfileCollectionTests(unittest.TestCase):
    def arguments(self, target=4):
        return pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD",
                                            "--tier-target", f"GOLD={target}"])

    def test_counts_valid_existing_matches_and_skips_download_when_target_met(self):
        db = FakeDB([record(str(i)) for i in range(4)] + [record("old", patch_version="16.2")])
        riot = SimpleNamespace(get_league_exp_players=lambda *a, **kw: self.fail("Unnecessary collection"))
        result = pipeline.collect_matches(self.arguments(), ["16.7"], riot, db, log=lambda _: None)
        self.assertEqual(result["GOLD"]["window_matches"], 4)
        self.assertEqual(result["GOLD"]["new_matches"], 0)

    def test_default_collection_continues_beyond_old_200_player_cap(self):
        pages = []
        riot = SimpleNamespace(
            get_league_exp_players=lambda tier, div, page: pages.append(page) or
                ([{"puuid": str(i)} for i in range((page - 1) * 80, page * 80)] if page <= 3 else []),
            get_match_ids=lambda puuid, limit: [puuid],
            get_match_data=lambda mid, tier: record(mid))
        db = FakeDB()
        result = pipeline.collect_matches(self.arguments(202), ["16.7"], riot, db, log=lambda _: None)
        self.assertEqual(result["GOLD"]["window_matches"], 202)
        self.assertEqual(result["GOLD"]["players"], 202)
        self.assertEqual(pages, [1, 2, 3])

    def test_reopens_a_finished_tier_when_its_quota_increases(self):
        db = FakeDB([record("old1"), record("old2")])
        riot = SimpleNamespace(get_league_exp_players=lambda *a, **kw: [{"puuid": "one"}],
            get_match_ids=lambda *a, **kw: ["old1", "new1", "new2"],
            get_match_data=lambda mid, tier: record(mid))
        with tempfile.TemporaryDirectory() as directory:
            progress = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            progress.state["collection"]["GOLD"] = dict(pipeline._collection_state(), done=True,
                                                        new_matches=2, division=4)
            result = pipeline.collect_matches(self.arguments(), ["16.7"], riot, db,
                                              log=lambda _: None, progress=progress)
        self.assertEqual(db.saved, ["new1", "new2"])
        self.assertEqual(result["GOLD"]["window_matches"], 4)

    def test_lost_commit_acknowledgement_does_not_double_count_population(self):
        class DB(FakeDB):
            fail_once = True
            def save_match(self, *args):
                inserted = super().save_match(*args)
                if self.fail_once:
                    self.fail_once = False
                    raise ConnectionError("lost commit acknowledgement")
                return inserted
        db = DB([record(str(i)) for i in range(3)])
        downloads = []
        riot = SimpleNamespace(get_league_exp_players=lambda *a, **kw: [{"puuid": "one"}],
            get_match_ids=lambda *a, **kw: ["new", "unnecessary"],
            get_match_data=lambda mid, tier: downloads.append(mid) or record(mid))
        with tempfile.TemporaryDirectory() as directory:
            progress = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            with self.assertRaises(ConnectionError):
                pipeline.collect_matches(self.arguments(), ["16.7"], riot, db, log=lambda _: None, progress=progress)
            resumed = UpdateProgress.open(directory, {}, lambda: [])
            result = pipeline.collect_matches(self.arguments(), ["16.7"], riot, db,
                                              log=lambda _: None, progress=resumed)
        self.assertEqual(result["GOLD"]["window_matches"], 4)
        self.assertEqual(db.saved, ["new"])
        self.assertEqual(downloads, ["new"])

    def test_skip_scrape_refuses_to_silently_train_below_population_target(self):
        args = self.arguments(5)
        args.min_matches, args.no_catalog, args.skip_scrape = 4, True, True
        db = FakeDB([record(str(i)) for i in range(4)])
        with tempfile.TemporaryDirectory() as directory:
            install_test_catalog(directory)
            with patch.object(pipeline, "APP_DIR", Path(directory)), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db)}), \
                    patch.object(pipeline, "train_outputs", side_effect=AssertionError("Training started")):
                with self.assertRaisesRegex(pipeline.PermanentUpdateError, "Population targets not reached"):
                    pipeline.run(args, log=lambda _: None)
            self.assertFalse((Path(directory) / "data/runtime_config.json").exists())
        self.assertTrue(db.closed)

    def test_exhausted_uncapped_collection_retries_and_later_publishes_exact_sample(self):
        args = self.arguments(5)
        args.min_matches, args.no_catalog = 4, True
        db = FakeDB([record(str(i)) for i in range(4)])
        def collect(args, patches, riot, db, log, progress):
            progress.state["collection"]["GOLD"] = dict(pipeline._collection_state(), done=True)
            return {}
        def fit(args, patches, db, records, stage, log):
            self.assertEqual(len(records), 5)
            (stage / "models").mkdir()
            (stage / "models/test.pth").write_bytes(b"test model")
            return {}, [Path("models/test.pth")]
        with tempfile.TemporaryDirectory() as directory:
            install_test_catalog(directory)
            with patch.object(pipeline, "APP_DIR", Path(directory)), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: SimpleNamespace(api_key="test"))}), \
                    patch.object(pipeline, "collect_matches", side_effect=collect), \
                    patch.object(pipeline, "train_outputs", side_effect=fit):
                with self.assertRaisesRegex(RuntimeError, "Retrying available players"):
                    pipeline.run(args, log=lambda _: None)
                state = json.loads((Path(directory) / ".model_update/state.json").read_text())
                self.assertFalse(state["collection"]["GOLD"]["done"])
                db.records.extend([record("enough"), record("excess")])
                self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            runtime = json.loads((Path(directory) / "data/runtime_config.json").read_text())
            self.assertEqual(runtime["DATASET_TARGETS"]["GOLD"], 5)
            manifest = json.loads((Path(directory) / "data/update_manifest.json").read_text())
            self.assertEqual(manifest["available_matches"]["GOLD"], 6)


if __name__ == "__main__":
    unittest.main()
