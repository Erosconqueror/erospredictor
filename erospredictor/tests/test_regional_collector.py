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
    def test_independent_routing_budgets_and_shared_europe_rejection(self):
        self.assertEqual(collector.selected_routes(["euw1", "kr", "na1"]), ["euw1", "kr", "na1"])
        self.assertEqual(collector.selected_routes(["kr", "kr"]), ["kr"])
        with self.assertRaisesRegex(ValueError, "share"):
            collector.selected_routes(["euw1", "eun1"])

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
