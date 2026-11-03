"""Routing, credentials, isolation and all-region training without network access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import regional_collector as collector
import update_pipeline as pipeline
from services.api_keys import get_api_key
from test_update_pipeline import record, FakeDB, install_test_catalog
from test_scrape_only import AvailableRiot


class RegionalTests(unittest.TestCase):
    def test_independent_routing_budgets_and_shared_europe_group(self):
        self.assertEqual(collector.selected_routes(["euw1", "kr", "na1"]), ["euw1", "kr", "na1"])
        self.assertEqual(collector.selected_routes(["kr", "kr"]), ["kr"])
        self.assertEqual(collector.route_groups(["euw1", "eun1", "kr", "na1"]),
                         [["euw1", "eun1"], ["kr"], ["na1"]])

    def test_europe_rotates_by_saved_match_counts_and_preserves_one_process_budget(self):
        counts = {"euw1": 0, "eun1": 0}
        order = []
        def collect(values):
            region = pipeline.parser().parse_args(values).region
            order.append(region)
            counts[region] += 100
            return pipeline.ROUND_YIELDED if len(order) < 4 else 0
        with patch.object(collector, "collected_count", side_effect=counts.get), \
                patch.object(pipeline, "main", side_effect=collect), \
                patch.object(pipeline, "REGIONAL_ROUND", False), patch.object(pipeline, "REGIONAL_WORKER", None):
            collector.collect_group(["euw1", "eun1"], [])
        self.assertEqual(order[:4], ["euw1", "eun1", "euw1", "eun1"])

    def test_worker_route_cannot_be_overridden_and_never_trains(self):
        values = collector.worker_arguments("kr", ["--region", "euw1", "--continent", "europe"])
        args = pipeline.parser().parse_args(values)
        self.assertEqual((args.region, args.continent), ("kr", "asia"))
        self.assertTrue(args.scrape_only)
        with self.assertRaises(ValueError):
            collector.worker_arguments("kr", ["--skip-scrape"])

    def test_credentials_have_explicit_profiles_and_environment_priority(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {}, clear=True):
            path = Path(folder) / "keys.json"
            path.write_text(json.dumps({"riot": {"default": "test-primary", "development": "test-dev"}}))
            self.assertEqual(get_api_key(path=path), "test-primary")
            self.assertEqual(get_api_key("development", path), "test-dev")
            with patch.dict("os.environ", {"RIOT_API_KEY": "test-environment"}):
                self.assertEqual(get_api_key(path=path), "test-environment")

    def test_supervisor_launches_all_routes_without_putting_keys_in_command(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "APP_DIR", Path(folder)), \
                patch.object(collector, "get_api_key", return_value="test-secret"), \
                patch.object(pipeline, "resolve_patch_window", return_value=["16.20", "16.21"]), \
                patch.object(collector.subprocess, "Popen") as spawn, patch.object(collector.time, "sleep"):
            spawn.side_effect = lambda *a, **kw: MagicMock(poll=lambda: 0)
            self.assertEqual(collector.supervise(["euw1", "kr", "na1"], "default", []), 0)
            self.assertEqual(spawn.call_count, 3)
            for call in spawn.call_args_list:
                self.assertNotIn("test-secret", " ".join(call.args[0]))
                self.assertEqual(call.kwargs["env"]["RIOT_API_KEY"], "test-secret")

    def test_all_regions_filter_keeps_matches_without_changing_single_region_filter(self):
        mapping = {str(cid): cid - 1 for cid in range(1, 11)}
        rows = [record("a", region="euw1"), record("b", region="kr"), record("c", region="na1")]
        self.assertEqual(len(pipeline.usable_matches(rows, mapping, ["16.7"], ["GOLD"], None)[0]), 3)
        self.assertEqual(len(pipeline.usable_matches(rows, mapping, ["16.7"], ["GOLD"], "kr")[0]), 1)

    def test_regional_lock_cannot_enable_training(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "APP_DIR", Path(folder)), \
                patch.object(pipeline, "REGIONAL_WORKER", "kr"):
            with self.assertRaisesRegex(ValueError, "scope"):
                pipeline.run(pipeline.parser().parse_args(["--skip-scrape"]))

    def test_real_collection_uses_only_its_regional_journal(self):
        from types import SimpleNamespace
        db = FakeDB()
        riot = AvailableRiot({"GOLD": 3})
        args = pipeline.parser().parse_args(["--scrape-only", "--patches", "16.7", "--region", "kr",
            "--continent", "asia", "--tiers", "GOLD", "--tier-target", "GOLD=20", "--no-catalog"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            install_test_catalog(root)
            with patch.object(pipeline, "APP_DIR", root), patch.object(pipeline, "REGIONAL_WORKER", "kr"), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertFalse((root / ".scrape_only").exists())
            state = json.loads((root / ".scrape_regions/kr/state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(len(db.saved), 3)

    def test_training_lock_remains_held_until_workers_stop(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "APP_DIR", Path(folder)):
            def stop(workers):
                with self.assertRaises(pipeline.UpdateLockError):
                    with pipeline.UpdateLock(Path(folder) / ".model_update.lock"):
                        pass
            with patch.object(collector, "stop_workers", side_effect=stop) as stopped:
                with collector.collector_lock({}):
                    pass
                stopped.assert_called_once()
            with pipeline.UpdateLock(Path(folder) / ".model_update.lock"):
                pass

    def test_europe_slice_yields_after_one_hundred_commits_with_goal_and_cursor_kept(self):
        from types import SimpleNamespace
        class TwoPlayers(AvailableRiot):
            def get_league_exp_players(self, tier, division, page):
                return [{"puuid": "first"}, {"puuid": "second"}] if division == "I" and page == 1 else []
        riot = TwoPlayers({"first": 100, "second": 100})
        db = FakeDB()
        args = pipeline.parser().parse_args(["--scrape-only", "--patches", "16.7", "--region", "euw1",
            "--tiers", "GOLD", "--tier-target", "GOLD=4000", "--matches-per-player", "100", "--no-catalog"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            install_test_catalog(root)
            with patch.object(pipeline, "APP_DIR", root), patch.object(pipeline, "REGIONAL_WORKER", "euw1"), \
                    patch.object(pipeline, "REGIONAL_ROUND", True), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                self.assertEqual(pipeline.run(args, log=lambda _: None), pipeline.ROUND_YIELDED)
                state = json.loads((root / ".scrape_regions/euw1/state.json").read_text())
                self.assertEqual(len(db.saved), 100)
                self.assertEqual(state["scrape_goals"], {"GOLD": 200})
                self.assertEqual(state["collection"]["GOLD"]["match_index"], 100)
                self.assertEqual(pipeline.run(args, log=lambda _: None), pipeline.ROUND_YIELDED)
                resumed = json.loads((root / ".scrape_regions/euw1/state.json").read_text())
                self.assertEqual(resumed["run_id"], state["run_id"])
                self.assertEqual(len(db.saved), 200)
                self.assertEqual(len(set(db.saved)), 200)
