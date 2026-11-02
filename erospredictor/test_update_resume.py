"""Abrupt-stop and restart regressions with local fixtures, never real APIs/DBs."""

import json
import importlib.util
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import update_pipeline as pipeline
from services.update_progress import UpdateProgress, atomic_json
from services.update_retry import wait_for_retry
from test_update_pipeline import FakeDB, record, install_test_catalog


class ProgressTests(unittest.TestCase):
    def test_resume_freezes_patch_window_and_rejects_incompatible_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            first = UpdateProgress.open(directory, {"last_patches": 3}, lambda: ["16.5", "16.6", "16.7"])
            first.state["collection"]["GOLD"] = {"done": True}
            first.save()
            second = UpdateProgress.open(directory, {"last_patches": 3}, lambda: self.fail("Patch list must stay frozen"))
            self.assertEqual(second.state["patches"], ["16.5", "16.6", "16.7"])
            self.assertTrue(second.state["collection"]["GOLD"]["done"])
            with self.assertRaisesRegex(ValueError, "different parameters"):
                UpdateProgress.open(directory, {"last_patches": 4}, lambda: [])

    def test_completed_crash_resume_does_not_start_another_update(self):
        with tempfile.TemporaryDirectory() as directory:
            first = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            first.complete()
            resumed = UpdateProgress.open(directory, {}, lambda: self.fail("Already completed"), resume_completed=True)
            self.assertEqual(resumed.state["status"], "complete")
            fresh = UpdateProgress.open(directory, {}, lambda: ["16.8"])
            self.assertNotEqual(fresh.state["run_id"], first.state["run_id"])

    def test_only_verified_artifacts_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            artifact = progress.staging / "model.pth"
            artifact.write_bytes(b"complete model")
            progress.finish_unit("GOLD:gnn", artifact, {"score": 1})
            self.assertIsNotNone(progress.completed_unit("GOLD:gnn"))
            artifact.write_bytes(b"damaged model")
            self.assertIsNone(progress.completed_unit("GOLD:gnn"))
            with self.assertRaises(ValueError):
                progress.artifact("../outside.pth")

    def test_new_run_intent_survives_metadata_disconnect_after_previous_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            first = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            first.complete()
            with self.assertRaises(ConnectionError):
                UpdateProgress.open(directory, {}, lambda: (_ for _ in ()).throw(ConnectionError("version server offline")))
            resumed = UpdateProgress.open(directory, {}, lambda: ["16.8"], resume_completed=True)
            self.assertEqual(resumed.state["status"], "running")
            self.assertEqual(resumed.state["patches"], ["16.8"])
            self.assertNotEqual(resumed.state["run_id"], first.state["run_id"])


class CollectionResumeTests(unittest.TestCase):
    def arguments(self):
        return pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "CHALLENGER", "MASTER",
                                             "--matches-per-tier", "2", "--players-per-tier", "4"])

    def test_connection_loss_resumes_exact_match_and_skips_finished_tier(self):
        args = self.arguments()
        db = FakeDB()
        seen = []
        fail_once = True

        class Riot:
            def get_league_exp_players(self, tier, *args, **kwargs):
                seen.append(("page", tier))
                return [{"puuid": tier}]

            def get_match_ids(self, puuid, limit):
                seen.append(("player", puuid))
                return [puuid + "1", puuid + "2"]

            def get_match_data(self, match_id, tier):
                nonlocal fail_once
                seen.append(("match", match_id))
                if match_id == "MASTER1" and fail_once:
                    fail_once = False
                    raise ConnectionError("remote closed connection")
                return record(match_id, tier=tier)

        with tempfile.TemporaryDirectory() as directory:
            progress = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            with self.assertRaises(ConnectionError):
                pipeline.collect_matches(args, ["16.7"], Riot(), db, log=lambda _: None, progress=progress)
            self.assertEqual(db.saved, ["CHALLENGER1", "CHALLENGER2"])
            progress = UpdateProgress.open(directory, {}, lambda: self.fail("Resume"))
            reports = pipeline.collect_matches(args, ["16.7"], Riot(), db, log=lambda _: None, progress=progress)
        self.assertEqual(reports["CHALLENGER"]["new_matches"], 2)
        self.assertEqual(reports["MASTER"]["new_matches"], 2)
        self.assertEqual(seen.count(("page", "CHALLENGER")), 1)
        self.assertEqual(seen.count(("player", "MASTER")), 1)
        self.assertEqual(seen.count(("match", "MASTER1")), 2)

    def test_commit_then_disconnect_replays_write_without_losing_quota(self):
        args = self.arguments()
        args.tiers = ["MASTER"]

        class CommittedDB(FakeDB):
            fail_once = True
            def save_match(self, match_id, region, data):
                if match_id not in self.saved:
                    self.saved.append(match_id)
                    self.records.append(dict(data, match_id=match_id))
                    if self.fail_once:
                        self.fail_once = False
                        raise ConnectionError("COMMIT succeeded but acknowledgement was lost")
                    return True
                return False

        calls = []
        riot = SimpleNamespace(
            get_league_exp_players=lambda *a, **kw: [{"puuid": "one"}],
            get_match_ids=lambda *a, **kw: ["m1", "m2"],
            get_match_data=lambda mid, tier: calls.append(mid) or record(mid, tier=tier))
        db = CommittedDB()
        with tempfile.TemporaryDirectory() as directory:
            progress = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            with self.assertRaises(ConnectionError):
                pipeline.collect_matches(args, ["16.7"], riot, db, log=lambda _: None, progress=progress)
            progress = UpdateProgress.open(directory, {}, lambda: [])
            result = pipeline.collect_matches(args, ["16.7"], riot, db, log=lambda _: None, progress=progress)
        self.assertEqual(db.saved, ["m1", "m2"])
        self.assertEqual(calls, ["m1", "m2"])
        self.assertEqual(result["MASTER"]["new_matches"], 2)

    def test_first_upgraded_run_adopts_the_already_completed_challenger_collection(self):
        args = self.arguments()
        db = FakeDB([record("old1", tier="CHALLENGER"), record("old2", tier="CHALLENGER")])
        visited = []
        class Riot:
            def get_league_exp_players(self, tier, *args, **kwargs):
                visited.append(tier)
                return [{"puuid": "one"}]
            def get_match_ids(self, *args, **kwargs):
                return ["new1", "new2"]
            def get_match_data(self, match_id, tier):
                return record(match_id, tier=tier)
        with tempfile.TemporaryDirectory() as directory:
            progress = UpdateProgress.open(directory, {}, lambda: ["16.7"])
            reports = pipeline.collect_matches(args, ["16.7"], Riot(), db, log=lambda _: None, progress=progress)
        self.assertNotIn("CHALLENGER", visited)
        self.assertEqual(reports["CHALLENGER"]["reused_matches"], 2)
        self.assertEqual(reports["CHALLENGER"]["new_matches"], 0)
        self.assertEqual(reports["MASTER"]["new_matches"], 2)


class RestartTests(unittest.TestCase):
    def test_main_restarts_unexpected_failure_but_ctrl_c_stops_immediately(self):
        with patch.object(pipeline, "run", side_effect=[ConnectionError("lost"), 0]) as run, \
                patch.object(pipeline, "wait_for_retry") as wait:
            self.assertEqual(pipeline.main(["--patches", "16.7"]), 0)
        self.assertEqual(run.call_count, 2)
        wait.assert_called_once()
        with patch.object(pipeline, "run", side_effect=ConnectionError("lost")) as run, \
                patch.object(pipeline, "wait_for_retry", side_effect=KeyboardInterrupt):
            self.assertEqual(pipeline.main(["--patches", "16.7"]), 130)
        self.assertEqual(run.call_count, 1)

    def test_auth_and_invalid_configuration_stop_instead_of_hot_loop(self):
        import requests
        response = requests.Response()
        response.status_code = 403
        with patch.object(pipeline, "run", side_effect=requests.HTTPError(response=response)), \
                patch.object(pipeline, "wait_for_retry") as wait:
            self.assertEqual(pipeline.main(["--patches", "16.7"]), 2)
        wait.assert_not_called()

    def test_retry_delay_is_interruptible_and_uses_short_sleeps(self):
        delays = []
        wait_for_retry(125, log=None, sleeper=delays.append)
        self.assertEqual(sum(delays), 125)
        self.assertLessEqual(max(delays), 5)

    def test_hard_crash_mid_publication_can_roll_forward_from_intact_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage, destination = root / "stage", root / "destination"
            stage.mkdir(); destination.mkdir()
            for name in ("one", "two"):
                (stage / name).write_text("new-" + name)
                (destination / name).write_text("old-" + name)
            code = (
                "from pathlib import Path\nimport os,sys\n"
                f"sys.path.insert(0,{str(Path(pipeline.__file__).parent)!r})\n"
                "import update_pipeline as p\noriginal=p.os.replace\n"
                "def crash(source,target):\n original(source,target)\n"
                f" if Path(target)==Path({str(destination / 'one')!r}): os._exit(99)\n"
                "p.os.replace=crash\n"
                f"p.publish_outputs(Path({str(stage)!r}),Path({str(destination)!r}),['one','two'])\n")
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=15,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual(result.returncode, 99, result.stderr)
            self.assertEqual((destination / "one").read_text(), "new-one")
            self.assertEqual((destination / "two").read_text(), "old-two")
            pipeline.publish_outputs(stage, destination, ["one", "two"])
            self.assertEqual((destination / "one").read_text(), "new-one")
            self.assertEqual((destination / "two").read_text(), "new-two")
            self.assertEqual((stage / "backup/one").read_text(), "old-one")

    def test_resume_at_publication_does_not_collect_or_train_again(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--skip-scrape", "--min-matches", "4",
                                            "--matches-per-tier", "4", "--no-catalog"])
        db = FakeDB([record(str(i)) for i in range(4)])
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            def fit(args, patches, db, records, stage, log):
                (stage / "data").mkdir(exist_ok=True)
                (stage / "models").mkdir()
                (stage / "models/done.pth").write_text("complete new model")
                return {}, [Path("models/done.pth")]
            with patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db)}), \
                    patch.object(pipeline, "APP_DIR", app), patch.object(pipeline, "train_outputs", side_effect=fit) as train:
                with patch.object(pipeline, "publish_outputs", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        pipeline.run(args, log=lambda _: None)
                with patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: self.fail("DB reopened"))}):
                    self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertEqual(train.call_count, 1)
            self.assertEqual((app / "models/done.pth").read_text(), "complete new model")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"), "ML packages required")
class TrainingPipelineResumeTests(unittest.TestCase):
    def test_ready_model_is_reused_after_crash_and_training_snapshot_is_frozen(self):
        import torch
        from model.train_model import DynamicTrainer
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        source = Path(__file__).parent
        names = {int(key): value for key, value in json.loads((source / "data/champion_names.json").read_text(encoding="utf-8")).items()}
        class DB(FakeDB):
            queries = 0
            def get_champion_names(self):
                return names
            def get_all_matches(self, **kwargs):
                DB.queries += 1
                return self.records
        modes = []
        def create_db(dev_mode):
            modes.append(dev_mode)
            return DB([record(str(i)) for i in range(8)])
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--skip-scrape",
                                            "--min-matches", "4", "--epochs", "1", "--qa-min-pass", "0",
                                            "--matches-per-tier", "8", "--no-catalog"])
        finish = UpdateProgress.finish_unit
        training = DynamicTrainer.train_single_model
        trained = []
        fail_once = True
        def save_then_crash(item, key, path, report):
            nonlocal fail_once
            finish(item, key, path, report)
            if key == "GOLD:roleweighted" and fail_once:
                fail_once = False
                raise ConnectionError("crashed after a verified model checkpoint")
        def track_training(item, X, y, name, *args, **kwargs):
            trained.append(name)
            return training(item, X, y, name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            (app / "data").mkdir()
            for name in ("champion_id.json", "champion_names.json", "golden_dataset.json"):
                shutil.copy2(source / "data" / name, app / "data" / name)
            with patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=create_db)}), \
                    patch.object(pipeline, "APP_DIR", app), \
                    patch("trainer_script.quick_evaluate_model", return_value=1.0), \
                    patch.object(UpdateProgress, "finish_unit", new=save_then_crash), \
                    patch.object(DynamicTrainer, "train_single_model", new=track_training):
                with self.assertRaises(ConnectionError):
                    pipeline.run(args, log=lambda _: None)
                self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertEqual(trained.count("GOLD_roleweighted"), 1)
            self.assertEqual(DB.queries, 1)
            self.assertEqual(modes, [True, False])
            self.assertEqual(len(list((app / "models").glob("*.pth"))), 6)
            self.assertEqual(json.loads((app / ".model_training/state.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
