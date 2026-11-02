"""Scrape-only mode stays separate from model updates and survives restarts."""

from collections import Counter
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import configs
import update_pipeline as pipeline
from services.dataset_profile import collection_steps
from test_catalog_update import document
from test_update_pipeline import FakeDB, record, install_test_catalog


class AvailableRiot:
    api_key = "offline-test"

    def __init__(self, available, fail_at=None):
        self.available, self.fail_at = available, fail_at
        self.error_type = ConnectionError
        self.pages, self.players, self.downloads = [], [], []

    def get_league_exp_players(self, tier, division, page):
        self.pages.append((tier, division, page))
        return [{"puuid": tier}] if division == "I" and page == 1 else []

    def get_match_ids(self, puuid, limit):
        self.players.append(puuid)
        return [f"{puuid}-{i}" for i in range(self.available[puuid])][:limit]

    def get_match_data(self, match_id, tier):
        self.downloads.append(match_id)
        if self.fail_at == match_id:
            self.fail_at = None
            raise self.error_type("simulated interruption")
        return record(match_id, tier=tier)


class ScrapeOnlyTests(unittest.TestCase):
    def arguments(self, *extra):
        return pipeline.parser().parse_args(["--scrape-only", "--patches", "16.7", "--no-catalog",
            "--tiers", "CHALLENGER", "GOLD", "--tier-target", "CHALLENGER=40", "--tier-target", "GOLD=80", *extra])

    def test_default_is_full_update_and_scrape_only_requires_its_explicit_flag(self):
        self.assertFalse(pipeline.parser().parse_args([]).scrape_only)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parser().parse_args(["--scrape-only", "--skip-scrape"])
        args = self.arguments("--matches-per-tier", "2", "--dry-run")
        with self.assertRaisesRegex(ValueError, "--tier-target"):
            pipeline.run(args, log=lambda _: None)

    def test_round_steps_follow_the_approved_profile(self):
        steps = collection_steps(configs.DATASET_TARGETS)
        self.assertEqual(steps, {tier: goal // 20 for tier, goal in configs.DATASET_TARGETS.items()})
        self.assertEqual(steps["CHALLENGER"], 100)
        self.assertEqual(steps["GOLD"], 1100)
        self.assertEqual(steps["SILVER"], 900)
        self.assertEqual(collection_steps({"GOLD": 21, "IRON": 1}), {"GOLD": 2, "IRON": 1})
        self.assertGreater(steps["GOLD"], steps["SILVER"])
        self.assertGreater(steps["SILVER"], steps["PLATINUM"])
        self.assertAlmostEqual(steps["GOLD"] / steps["CHALLENGER"], 11, delta=0.3)

    def test_dry_run_uses_no_database_ml_network_or_checkpoint(self):
        messages = []
        args = self.arguments("--dry-run")
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pipeline, "APP_DIR", Path(directory)), \
                patch.object(pipeline, "fetch_versions", side_effect=AssertionError("Network")), \
                patch.object(pipeline, "prepare_catalog", side_effect=AssertionError("Catalog")), \
                patch.dict("sys.modules", {"model.data_manager": None, "torch": None}):
            self.assertEqual(pipeline.run(args, log=messages.append), 0)
            self.assertEqual(list(Path(directory).iterdir()), [])
        plan = json.loads(messages[0])
        self.assertEqual(plan["stages"], ["collect"])
        self.assertIsNone(plan["fixed_match_limit"])
        self.assertNotIn("MIXED", plan["tiers"])

    def test_collects_beyond_base_quota_in_balanced_rounds_without_changing_app(self):
        db = FakeDB()
        riot = AvailableRiot({"CHALLENGER": 5, "GOLD": 13})
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            (app / ".model_update").mkdir()
            originals = {".model_update/state.json": b"unfinished training checkpoint",
                "models/original.pth": b"original model", "data/stats_cache.json": b"{}",
                "data/meta_champs.json": b"{}", "data/runtime_config.json": b"{}"}
            for name, value in originals.items():
                target = app / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
            for name in ("data/champion_id.json", "data/champion_names.json"):
                originals[name] = (app / name).read_bytes()
            with patch.object(pipeline, "APP_DIR", app), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot), "torch": None}), \
                    patch.object(pipeline, "train_outputs", side_effect=AssertionError("Training")), \
                    patch.object(pipeline, "publish_outputs", side_effect=AssertionError("Publishing")):
                self.assertEqual(pipeline.run(self.arguments(), log=lambda _: None), 0)
            self.assertEqual(Counter(row["tier"] for row in db.records), {"CHALLENGER": 5, "GOLD": 13})
            self.assertEqual(Counter(mid.split("-")[0] for mid in db.saved[:6]), {"CHALLENGER": 2, "GOLD": 4})
            self.assertEqual(Counter(mid.split("-")[0] for mid in db.saved[:12]), {"CHALLENGER": 4, "GOLD": 8})
            self.assertEqual(Counter(riot.players), {"CHALLENGER": 1, "GOLD": 1})
            self.assertEqual(len(riot.downloads), len(set(riot.downloads)))
            for name, value in originals.items():
                self.assertEqual((app / name).read_bytes(), value)
            self.assertEqual(json.loads((app / ".scrape_only/state.json").read_text())["status"], "complete")
            report = json.loads((app / ".scrape_only/last_collection.json").read_text())
            self.assertEqual(report["collection"]["GOLD"]["new_matches"], 13)
        self.assertTrue(db.closed)

    def test_connection_loss_resumes_same_match_and_keeps_cached_player_histories(self):
        db = FakeDB()
        riot = AvailableRiot({"CHALLENGER": 5, "GOLD": 13}, fail_at="GOLD-2")
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            with patch.object(pipeline, "APP_DIR", app), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                with self.assertRaises(ConnectionError):
                    pipeline.run(self.arguments(), log=lambda _: None)
                state = json.loads((app / ".scrape_only/state.json").read_text())
                self.assertEqual(state["collection"]["CHALLENGER"]["match_index"], 2)
                self.assertEqual(state["collection"]["GOLD"]["match_index"], 2)
                # An older rotation policy must not force us to finish its old quota.
                state.pop("scrape_steps")
                state["scrape_goals"] = {"CHALLENGER": 100, "GOLD": 100}
                pipeline.atomic_json(app / ".scrape_only/state.json", state)
                self.assertEqual(pipeline.run(self.arguments(), log=lambda _: None), 0)
                resumed = json.loads((app / ".scrape_only/state.json").read_text())
                self.assertEqual(resumed["run_id"], state["run_id"])
                self.assertEqual(resumed["scrape_steps"], {"CHALLENGER": 2, "GOLD": 4})
            self.assertEqual(len(db.saved), 18)
            self.assertEqual(Counter(mid.split("-")[0] for mid in db.saved[:6]), {"CHALLENGER": 2, "GOLD": 4})
            self.assertEqual(riot.downloads.count("GOLD-2"), 2)
            self.assertEqual(Counter(riot.players), {"CHALLENGER": 1, "GOLD": 1})

    def test_user_cancellation_exits_without_retry_and_next_launch_resumes(self):
        db = FakeDB()
        riot = AvailableRiot({"GOLD": 5}, fail_at="GOLD-1")
        riot.error_type = KeyboardInterrupt
        argv = ["--scrape-only", "--patches", "16.7", "--tiers", "GOLD", "--no-catalog",
                "--tier-target", "GOLD=2"]
        with tempfile.TemporaryDirectory() as directory:
            install_test_catalog(directory)
            with patch.object(pipeline, "APP_DIR", Path(directory)), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}), \
                    patch.object(pipeline, "wait_for_retry", side_effect=AssertionError("Cancellation retried")):
                with patch("builtins.print"), redirect_stderr(io.StringIO()):
                    self.assertEqual(pipeline.main(argv), 130)
                    self.assertEqual(len(db.saved), 1)
                    self.assertEqual(pipeline.main(argv), 0)
            self.assertEqual(len(db.saved), 5)
            self.assertEqual(riot.downloads.count("GOLD-1"), 2)

    def test_crash_after_last_cursor_save_reconstructs_final_report_on_restart(self):
        db = FakeDB()
        riot = AvailableRiot({"CHALLENGER": 0, "GOLD": 3})
        save = pipeline.UpdateProgress.save
        fail_once = True
        def last_cursor_then_crash(progress):
            nonlocal fail_once
            save(progress)
            states = progress.state["collection"]
            if fail_once and len(states) == 2 and all(value.get("exhausted") for value in states.values()):
                fail_once = False
                raise ConnectionError("crash after final cursor save")
        with tempfile.TemporaryDirectory() as directory:
            install_test_catalog(directory)
            with patch.object(pipeline, "APP_DIR", Path(directory)), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}), \
                    patch.object(pipeline.UpdateProgress, "save", new=last_cursor_then_crash):
                with self.assertRaises(ConnectionError):
                    pipeline.run(self.arguments(), log=lambda _: None)
                self.assertEqual(pipeline.run(self.arguments(), log=lambda _: None), 0)
            report = json.loads((Path(directory) / ".scrape_only/last_collection.json").read_text())["collection"]
            self.assertEqual(report["GOLD"]["new_matches"], 3)
            self.assertEqual(report["GOLD"]["window_matches"], 3)

    def test_scrape_only_respects_the_same_lock_as_full_model_updates(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "APP_DIR", Path(directory)):
            with pipeline.UpdateLock(Path(directory) / ".model_update.lock"):
                with self.assertRaises(pipeline.UpdateLockError):
                    pipeline.run(self.arguments(), log=lambda _: None)
            self.assertFalse((Path(directory) / ".scrape_only").exists())

    def test_lost_commit_acknowledgement_does_not_duplicate_records_or_counts(self):
        class DB(FakeDB):
            fail_once = True
            def save_match(self, *args):
                inserted = super().save_match(*args)
                if self.fail_once:
                    self.fail_once = False
                    raise ConnectionError("COMMIT succeeded; acknowledgement lost")
                return inserted
        db = DB()
        riot = AvailableRiot({"CHALLENGER": 3, "GOLD": 7})
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            with patch.object(pipeline, "APP_DIR", app), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                with self.assertRaises(ConnectionError):
                    pipeline.run(self.arguments(), log=lambda _: None)
                self.assertEqual(pipeline.run(self.arguments(), log=lambda _: None), 0)
            report = json.loads((app / ".scrape_only/last_collection.json").read_text())["collection"]
            self.assertEqual(report["CHALLENGER"]["window_matches"], 3)
            self.assertEqual(report["GOLD"]["window_matches"], 7)
            self.assertEqual(report["CHALLENGER"]["reused_matches"], 0)
            self.assertEqual(report["GOLD"]["reused_matches"], 0)
            self.assertEqual(len(db.saved), 10)
            self.assertEqual(len(riot.downloads), 10)

    def test_counts_are_loaded_once_then_existing_excess_is_balanced_by_other_tiers(self):
        class DB(FakeDB):
            queries = 0
            def get_all_matches(self, **kwargs):
                self.queries += 1
                return super().get_all_matches(**kwargs)
        db = DB([record(f"CHALLENGER-{i}", tier="CHALLENGER") for i in range(4)])
        riot = AvailableRiot({"CHALLENGER": 5, "GOLD": 13})
        with tempfile.TemporaryDirectory() as directory:
            install_test_catalog(directory)
            with patch.object(pipeline, "APP_DIR", Path(directory)), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                self.assertEqual(pipeline.run(self.arguments(), log=lambda _: None), 0)
            self.assertEqual(Counter(mid.split("-")[0] for mid in db.saved[:8]), {"GOLD": 8})
            self.assertEqual(db.queries, 1)

    def test_search_cap_and_empty_tier_finish_without_training_minimum(self):
        db = FakeDB()
        riot = AvailableRiot({"CHALLENGER": 0, "GOLD": 3})
        with tempfile.TemporaryDirectory() as directory:
            install_test_catalog(directory)
            with patch.object(pipeline, "APP_DIR", Path(directory)), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                self.assertEqual(pipeline.run(self.arguments("--players-per-tier", "1"), log=lambda _: None), 0)
            self.assertEqual(len(db.saved), 3)

    def test_new_champions_are_collected_using_staged_catalog_without_icons_or_publication(self):
        db = FakeDB()
        riot = AvailableRiot({"CHALLENGER": 1, "GOLD": 0})
        original = riot.get_match_data
        def match(mid, tier):
            row = original(mid, tier)
            row["blue_team"][0] = 11
            return row
        riot.get_match_data = match
        args = self.arguments()
        args.no_catalog = False
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            before = (app / "data/champion_id.json").read_bytes()
            with patch.object(pipeline, "APP_DIR", app), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}), \
                    patch("services.catalog_update.fetch_json", side_effect=[["16.7.1"], document(range(1, 12))]), \
                    patch("services.catalog_update.urlopen", side_effect=AssertionError("Icon download")):
                self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertEqual(db.records[0]["blue_team"][0], 11)
            self.assertEqual((app / "data/champion_id.json").read_bytes(), before)
            self.assertFalse((app / "assets").exists())


if __name__ == "__main__":
    unittest.main()
