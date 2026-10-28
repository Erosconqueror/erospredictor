"""Offline pipeline checks: no Riot requests, PostgreSQL, or model training."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import update_pipeline as pipeline


def record(match_id="one", tier="GOLD", patch_version="16.7", region="euw1"):
    return {"match_id": match_id, "tier": tier, "patch": patch_version,
            "region": region, "blue_team": list(range(1, 6)),
            "red_team": list(range(6, 11)), "blue_win": True}


class FakeDB:
    def __init__(self, records=None):
        self.records = records or []
        self.saved = []
        self.closed = False

    def get_all_match_ids(self):
        return [row["match_id"] for row in self.records]

    def get_all_matches(self, **kwargs):
        self.query = kwargs
        return self.records

    def get_champion_mapping(self):
        return {str(i): i - 1 for i in range(1, 11)}

    def save_match(self, match_id, region, data):
        if match_id in self.get_all_match_ids():
            return False
        self.saved.append(match_id)
        self.records.append(dict(data, match_id=match_id, region=region))
        return True

    def close(self):
        self.closed = True


def install_test_catalog(app, mapping=None, names=None):
    folder = Path(app) / "data"
    folder.mkdir(parents=True, exist_ok=True)
    mapping = mapping or FakeDB().get_champion_mapping()
    names = names or {str(value): "Champion" + str(value) for value in mapping.values()}
    (folder / "champion_id.json").write_text(json.dumps(mapping), encoding="utf-8")
    (folder / "champion_names.json").write_text(json.dumps(names), encoding="utf-8")


class PatchSelectionTests(unittest.TestCase):
    def test_public_and_internal_patch_names_share_one_window(self):
        args = pipeline.parser().parse_args(["--patches", "26.7", "16.5.1", "16.7", "16.6"])
        self.assertEqual(pipeline.resolve_patch_window(args), ["16.5", "16.6", "16.7"])
        for invalid in ("latest", "16", "16.0", "-1.7"):
            with self.assertRaises(ValueError):
                pipeline.normalize_patch(invalid)

    def test_actual_releases_handle_duplicates_and_year_boundary(self):
        args = pipeline.parser().parse_args(["--last-patches", "3", "--end-patch", "26.1"])
        versions = ["16.2.1", "16.1.2", "16.1.1", "15.24.1", "15.23.1"]
        self.assertEqual(pipeline.resolve_patch_window(args, versions), ["15.23", "15.24", "16.1"])

    def test_explicit_dry_run_needs_no_network_database_or_ml(self):
        args = pipeline.parser().parse_args(["--patches", "16.6", "16.7", "--dry-run"])
        with patch.object(pipeline, "fetch_versions", side_effect=AssertionError("Network used")):
            self.assertEqual(pipeline.run(args, log=lambda _: None), 0)


class CollectionTests(unittest.TestCase):
    def test_collection_is_finite_deduplicated_and_stops_at_quota(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD",
                                            "--matches-per-tier", "2", "--players-per-tier", "3"])
        db = FakeDB([record("existing")])
        calls = []

        class Riot:
            def get_league_exp_players(self, *args, **kwargs):
                return [{"puuid": "p"}, {"puuid": "p"}, {"puuid": "q"}]

            def get_match_ids(self, puuid, limit):
                calls.append(puuid)
                return ["existing", "old", "new1", "new1", "new2", "new3"]

            def get_match_data(self, match_id, tier):
                return record(match_id, tier=tier, patch_version="16.5" if match_id == "old" else "16.7")

        report = pipeline.collect_matches(args, ["16.7"], Riot(), db, log=lambda _: None)
        self.assertEqual(db.saved, ["new1", "new2"])
        self.assertEqual(calls, ["p"])
        self.assertEqual(report["GOLD"]["new_matches"], 2)
        self.assertEqual(report["GOLD"]["outside_window_or_invalid"], 1)

    def test_empty_pages_finish_without_continuous_loop(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "MASTER"])
        riot = SimpleNamespace(get_league_exp_players=lambda *a, **kw: [])
        report = pipeline.collect_matches(args, ["16.7"], riot, FakeDB(), log=lambda _: None)
        self.assertEqual(report["MASTER"]["new_matches"], 0)


class SnapshotTests(unittest.TestCase):
    def test_same_clean_window_excludes_new_champions_and_bad_records(self):
        valid = record()
        unknown, duplicate = record("unknown"), record("duplicate")
        unknown["blue_team"][0] = 999
        duplicate["red_team"][0] = 1
        data = [valid, record("old", patch_version="16.2"), record("other", region="kr"), unknown, duplicate]
        before = deepcopy(data)
        rows, rejected = pipeline.usable_matches(data, FakeDB().get_champion_mapping(), ["16.7"], ["GOLD"], "euw1")
        self.assertEqual(rows, [valid])
        self.assertEqual(rejected, {"outside_window": 2, "unknown_champion": 1, "invalid_teams": 1})
        self.assertEqual(data, before)

    def test_cache_and_meta_writes_are_staged(self):
        with tempfile.TemporaryDirectory() as temp:
            view = pipeline.WindowData(FakeDB(), [record()], temp)
            view.save_stats_cache({"GOLD": {"sample": 1}})
            view.save_meta_champs({"GOLD": {"strict": {}}})
            self.assertEqual(view.get_all_matches(), [record()])
            cache = Path(temp) / "data/stats_cache.json"
            self.assertEqual(json.loads(cache.read_text()), {"GOLD": {"sample": 1}})
            self.assertTrue((Path(temp) / "data/meta_champs.json").is_file())

    def test_all_preprocessor_representations_share_patch_filter_and_accept_zero_index(self):
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(),
            "model.data_manager": SimpleNamespace(DataManager=FakeDB),
            "model.gnn_predictor": SimpleNamespace(create_graph=lambda blue, red, win, weight: (blue, red, win, weight)),
        }):
            spec = importlib.util.spec_from_file_location("_pipeline_test_preprocessor", Path(__file__).parent / "model/preprocessor.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        records = [record("old", patch_version="16.5"), record("previous", patch_version="16.6"), record("current")]
        with tempfile.TemporaryDirectory() as temp:
            prep = module.Preprocessor(db=pipeline.WindowData(FakeDB(), records, temp), allowed_patches=["16.6", "16.7"])
            weighted = prep.process_matches()
            aware = prep.process_matches_ra()
            graphs, _ = prep.process_matches_gnn()
            self.assertEqual(len(weighted[0]), 2)
            self.assertEqual(len(aware[0]), 2)
            self.assertEqual(len(graphs), 2)
            self.assertAlmostEqual(weighted[3][0], 0.8 * pipeline.cfg.DIVISION_WEIGHTS["GOLD"])
            self.assertAlmostEqual(weighted[3][1], pipeline.cfg.DIVISION_WEIGHTS["GOLD"])
            self.assertEqual(weighted[0][0][0], pipeline.cfg.ROLE_WEIGHTS["TOP"])
            self.assertEqual(aware[0][0][0], 1.0)
            self.assertEqual(graphs[0][0][0], 0)
            prep.gen_meta_champs()
            self.assertEqual(set(json.loads((Path(temp) / "data/meta_champs.json").read_text())), {"GOLD", "MIXED"})


class PublicationTests(unittest.TestCase):
    def test_missing_output_never_changes_published_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staging, dest = root / "stage", root / "app"
            staging.mkdir()
            dest.mkdir()
            (dest / "old.pth").write_text("original")
            with self.assertRaises(RuntimeError):
                pipeline.publish_outputs(staging, dest, ["old.pth"])
            self.assertEqual((dest / "old.pth").read_text(), "original")

    def test_failure_during_publication_restores_previous_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staging, dest = root / "stage", root / "app"
            staging.mkdir()
            dest.mkdir()
            for name in ("one", "two"):
                (dest / name).write_text("old-" + name)
                (staging / name).write_text("new-" + name)
            original_replace = pipeline.os.replace
            failed = False

            def replace(source, target):
                nonlocal failed
                if Path(target) == dest / "two" and not failed:
                    failed = True
                    raise OSError("simulated full disk")
                return original_replace(source, target)

            with patch.object(pipeline.os, "replace", side_effect=replace):
                with self.assertRaises(OSError):
                    pipeline.publish_outputs(staging, dest, ["one", "two"])
            self.assertEqual((dest / "one").read_text(), "old-one")
            self.assertEqual((dest / "two").read_text(), "old-two")

    def test_skip_scrape_update_filters_then_publishes_only_after_training(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--skip-scrape", "--min-matches", "4",
                                            "--matches-per-tier", "4", "--no-catalog"])
        records = [record(str(i)) for i in range(4)] + [record("old", patch_version="16.5")]
        db = FakeDB(records)
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp)
            (app / "data").mkdir()
            install_test_catalog(app)
            (app / "data/stats_cache.json").write_text(json.dumps({"IRON": {"old": True}, "GOLD": {"old": True}}))
            (app / "data/meta_champs.json").write_text(json.dumps({"IRON": {"old": True}, "MIXED": {"old": True}}))

            def fit(args, patches, snapshot_db, selected, stage, log):
                self.assertEqual(len(selected), 4)
                self.assertFalse((app / "models/done.pth").exists())
                (stage / "models").mkdir()
                (stage / "data").mkdir(exist_ok=True)
                (stage / "models/done.pth").write_text("new")
                (stage / "data/stats_cache.json").write_text(json.dumps({"GOLD": {"new": True}}))
                (stage / "data/meta_champs.json").write_text(json.dumps({"GOLD": {"new": True}, "MIXED": {"new": True}}))
                return {"GOLD": {"matches": 4}}, [Path("models/done.pth"), Path("data/stats_cache.json"), Path("data/meta_champs.json")]

            with patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db)}), \
                    patch.object(pipeline, "APP_DIR", app), patch.object(pipeline, "train_outputs", side_effect=fit):
                self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertEqual((app / "models/done.pth").read_text(), "new")
            manifest = json.loads((app / "data/update_manifest.json").read_text())
            self.assertEqual(manifest["plan"]["patches"], ["16.7"])
            self.assertEqual(db.query["patches"], ["16.7"])
            self.assertTrue(db.closed)
            cache = json.loads((app / "data/stats_cache.json").read_text())
            self.assertEqual(cache["IRON"], {"old": True})
            self.assertEqual(cache["GOLD"], {"new": True})
            self.assertEqual(cache["__scope__"], {"mixed_tiers": ["GOLD"], "patches": ["16.7"]})
            meta = json.loads((app / "data/meta_champs.json").read_text())
            self.assertEqual(meta["IRON"], {"old": True})
            self.assertEqual(meta["MIXED"], {"new": True})

    def test_insufficient_tier_keeps_outputs_and_closes_database(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--skip-scrape", "--min-matches", "4",
                                            "--matches-per-tier", "4", "--no-catalog"])
        db = FakeDB([record()])
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db)}), \
                patch.object(pipeline, "APP_DIR", Path(temp)), \
                patch.object(pipeline, "train_outputs", side_effect=AssertionError("Training should not start")):
            install_test_catalog(temp)
            with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                pipeline.run(args, log=lambda _: None)
        self.assertTrue(db.closed)


class UpdateLockTests(unittest.TestCase):
    def test_overlapping_launch_is_rejected_and_exception_releases_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "update.lock"
            with self.assertRaisesRegex(ValueError, "interruption"):
                with pipeline.UpdateLock(path):
                    with self.assertRaisesRegex(RuntimeError, "Another model update"):
                        with pipeline.UpdateLock(path):
                            self.fail("Second update acquired the lock")
                    raise ValueError("simulated interruption")
            with pipeline.UpdateLock(path):
                pass
            self.assertIn(str(pipeline.os.getpid()), path.read_text())


class RiotCleaningTests(unittest.TestCase):
    def test_cleaner_sorts_roles_and_rejects_remakes(self):
        # RiotWatcher is optional for these offline data-cleaning tests.
        with patch.dict("sys.modules", {"riotwatcher": SimpleNamespace(LolWatcher=object),
                                        "requests": SimpleNamespace()}):
            spec = importlib.util.spec_from_file_location("_pipeline_test_riot", Path(__file__).parent / "model/riot.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        client = module.Riot.__new__(module.Riot)
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        participants = [{"teamId": team, "teamPosition": role, "championId": offset + i}
                        for team, offset in ((100, 1), (200, 6)) for i, role in enumerate(roles)]
        raw = {"info": {"participants": list(reversed(participants)), "queueId": 420,
                        "gameDuration": 1000, "gameVersion": "16.7.123.1",
                        "teams": [{"teamId": 200, "win": False}, {"teamId": 100, "win": True}]}}
        result = client._clean_data(raw, "GOLD")
        self.assertEqual(result["blue_team"], [1, 2, 3, 4, 5])
        self.assertEqual(result["red_team"], [6, 7, 8, 9, 10])
        self.assertEqual(result["patch"], "16.7")
        self.assertTrue(result["blue_win"])
        raw["info"]["gameDuration"] = 150
        self.assertIsNone(client._clean_data(raw, "GOLD"))


if __name__ == "__main__":
    unittest.main()
