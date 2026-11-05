"""Daily planning, patch rollover, SEA routing and resumable deeper history."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import collection_day as day
import regional_collector as regional
import update_pipeline as pipeline
from services.api_keys import get_api_key
from test_update_pipeline import FakeDB, record, install_test_catalog


class DayTests(unittest.TestCase):
    def test_plan_has_four_profiles_disjoint_platforms_and_no_split_budgets(self):
        plan = day.validate_plan(day.DEFAULT_ASSIGNMENTS)
        platforms = [r for regions in plan.values() for r in regions]
        self.assertEqual(len(platforms), len(set(platforms)))
        self.assertEqual(len(plan), 4)
        self.assertIn("vn2", platforms)
        self.assertIn("br1", platforms)
        self.assertEqual(regional.ROUTES["vn2"], "sea")
        self.assertEqual(regional.ROUTES["br1"], "americas")
        for invalid in ({"default": ["euw1", "eun1"]}, {"default": ["euw1"], "development": ["euw1"]}):
            with self.assertRaises(ValueError):
                day.validate_plan(invalid)

    def test_dry_day_plan_needs_no_keys_and_bounds_duration(self):
        with patch.object(day, "get_api_key", side_effect=AssertionError("Secret lookup")), patch("builtins.print"):
            self.assertEqual(day.run(day.DEFAULT_ASSIGNMENTS, 24, ["--patches", "16.22"], True), 0)
        for hours in (0, 25, float("nan")):
            with self.assertRaises(ValueError):
                day.run(day.DEFAULT_ASSIGNMENTS, hours, [], True)

    def test_expired_run_starts_no_worker(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "APP_DIR", Path(folder)), \
                patch.object(regional, "get_api_key", return_value="test-key"), \
                patch.object(pipeline, "resolve_patch_window", return_value=["16.22"]), \
                patch.object(regional.subprocess, "Popen") as popen:
            self.assertEqual(regional.supervise(["vn2"], "default", [], deadline=0), 0)
            popen.assert_not_called()

    def test_day_starts_disjoint_profiles_with_deep_history_and_duration(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "APP_DIR", Path(folder)), \
                patch.object(day, "get_api_key", side_effect=lambda p: "test-secret-" + p), \
                patch.object(day.subprocess, "Popen") as popen:
            popen.side_effect = lambda *a, **kw: MagicMock(poll=lambda: 0)
            self.assertEqual(day.run(day.DEFAULT_ASSIGNMENTS, 24, []), 0)
            self.assertEqual(popen.call_count, 4)
            for call in popen.call_args_list:
                command = call.args[0]
                self.assertIn("--run-hours", command)
                self.assertIn("--history-pages", command)
                self.assertNotIn("test-secret", " ".join(command))

    def test_missing_or_identical_keys_stop_before_launching_any_profile(self):
        for value in ("", "same-test-key"):
            with patch.object(day, "get_api_key", return_value=value), patch.object(day.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    day.run(day.DEFAULT_ASSIGNMENTS, 24, [])
                popen.assert_not_called()

    def test_immediate_profile_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "APP_DIR", Path(folder)), \
                patch.object(day, "get_api_key", side_effect=lambda p: "test-" + p), \
                patch.object(day.subprocess, "Popen", return_value=MagicMock(poll=lambda: 2)):
            self.assertEqual(day.run(day.DEFAULT_ASSIGNMENTS, 24, []), 2)

    def test_new_patch_uses_new_namespace_and_sea_worker_route(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, "APP_DIR", Path(folder)), \
                patch.object(regional, "get_api_key", return_value="test-key"), \
                patch.object(pipeline, "resolve_patch_window", return_value=["16.21", "16.22"]), \
                patch.object(regional.subprocess, "Popen") as popen, patch.object(regional.time, "sleep"):
            popen.return_value = MagicMock(poll=lambda: 0)
            self.assertEqual(regional.supervise(["vn2"], "default", []), 0)
            call = popen.call_args
            self.assertEqual(call.kwargs["env"]["ERO_COLLECT_WINDOW"], "patches-16.21_16.22")
            self.assertIn("sea", call.args[0])

    def test_deeper_history_paginates_and_stops_at_old_patch_or_repeated_page(self):
        for mode, expected_starts, expected_saved in (("all", [0, 2, 4], 5), ("old", [0, 2], 2), ("repeat", [0, 2], 2)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                install_test_catalog(root)
                db, starts = FakeDB(), []
                def ids(puuid, limit, start=0):
                    starts.append(start)
                    offset = 0 if mode == "repeat" else start
                    return [str(i) for i in range(offset, min(offset + limit, 5))]
                def match(mid, tier):
                    return record(mid, tier=tier, patch_version="16.6" if mode == "old" and int(mid) >= 2 else "16.7")
                riot = SimpleNamespace(api_key="test", get_league_exp_players=lambda tier, div, page:
                    [{"puuid": "one"}] if div == "I" and page == 1 else [], get_match_ids=ids, get_match_data=match)
                args = pipeline.parser().parse_args(["--scrape-only", "--no-catalog", "--patches", "16.7",
                    "--tiers", "GOLD", "--tier-target", "GOLD=1000", "--matches-per-player", "2", "--history-pages", "3"])
                with patch.object(pipeline, "APP_DIR", root), patch.dict("sys.modules", {
                        "model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                        "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                    self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
                self.assertEqual(starts, expected_starts)
                self.assertEqual(len(db.saved), expected_saved)

    def test_foreign_platform_match_ids_are_not_downloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            install_test_catalog(root)
            db, downloads = FakeDB(), []
            def fetch(mid, tier):
                downloads.append(mid)
                return record(mid, tier=tier)
            riot = SimpleNamespace(api_key="test", get_league_exp_players=lambda tier, div, page:
                [{"puuid": "moved-account"}] if div == "I" and page == 1 else [],
                get_match_ids=lambda *a, **kw: ["BR1_123", "EUW1_123"], get_match_data=fetch)
            args = pipeline.parser().parse_args(["--scrape-only", "--no-catalog", "--patches", "16.7", "--region", "euw1",
                "--tiers", "GOLD", "--tier-target", "GOLD=20"])
            with patch.object(pipeline, "APP_DIR", root), patch.dict("sys.modules", {
                    "model.data_manager": SimpleNamespace(DataManager=lambda **kw: db),
                    "model.riot": SimpleNamespace(Riot=lambda **kw: riot)}):
                self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertEqual(downloads, ["EUW1_123"])
