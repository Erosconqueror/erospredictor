"""Frozen-window interaction artifacts, train-only priors and resume policies."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import trainer_script as trainer
import update_pipeline as pipeline
from model.gnn_predictor import create_graph
from model.preprocessor import Preprocessor
from model.train_model import DynamicTrainer
from model.meta_calibrator import MetaLearningCalibrator
from model.statistical import StatisticalModel
from services.interaction_statistics import (build_interaction_statistics, InteractionStatistics,
                                             save_interaction_statistics)
from services.blind_pick import build_blind_profiles
from services.update_progress import UpdateProgress
from test_update_pipeline import FakeDB, record


class TrainingDB(FakeDB):
    def get_champion_names(self):
        path = Path(__file__).resolve().parents[1] / "data/champion_names.json"
        return {int(cid): name for cid, name in json.loads(path.read_text(encoding="utf-8")).items()}


class InteractionPreparationTests(unittest.TestCase):
    def test_prior_factory_excludes_validation_rows_including_their_eligibility(self):
        rows = [record(str(index)) for index in range(30)]
        for row in rows[24:]:
            row["blue_team"][0] = 11
        mapping = {str(index): index - 1 for index in range(1, 12)}
        indices = list(range(24))
        factory = trainer.make_interaction_prior_factory(rows, mapping, "GOLD", patches=["16.7"], region="euw1")
        with patch("services.interaction_statistics.build_interaction_statistics", wraps=build_interaction_statistics) as build:
            prior = factory(indices)
            self.assertEqual([row["match_id"] for row in build.call_args.args[0]], [str(index) for index in indices])
            self.assertEqual(build.call_args.kwargs["source"], "gnn_training_subset")
            self.assertNotIn(10, prior["eligible"])
            self.assertIs(factory(indices), prior)
            build.assert_called_once()
        changed = deepcopy(rows)
        for row in changed[24:]:
            row["blue_win"] = not row["blue_win"]
        other = trainer.make_interaction_prior_factory(changed, mapping, "GOLD", patches=["16.7"], region="euw1")
        self.assertEqual(other(indices), prior)
        with self.assertRaises(ValueError):
            factory([0, 0])

    def test_graph_alignment_retains_true_patch_metadata_and_zero_internal_champion(self):
        rows = [record("old", patch_version="16.6"), record("current")]
        invalid = record("invalid")
        invalid["blue_team"][0] = 999
        graphs = [create_graph(list(range(5)), list(range(5, 10)), True) for _ in rows]
        mapping = FakeDB().get_champion_mapping()
        aligned = trainer.graph_training_records(graphs, ["GOLD"] * 2, mapping, [invalid] + rows)
        self.assertEqual(aligned, rows)
        self.assertEqual(aligned[0]["blue_team"][0], 1)  # Riot ID 1 corresponds to internal index 0.
        self.assertEqual([row["patch"] for row in aligned], ["16.6", "16.7"])
        with self.assertRaises(ValueError):
            trainer.graph_training_records(graphs, ["GOLD"] * 2, mapping, rows[:1])

    def test_interactive_action_freezes_one_sql_read_for_all_representations(self):
        class ChangingDB(FakeDB):
            calls = 0
            def get_all_matches(self, **kwargs):
                self.calls += 1
                return [record(str(index)) for index in range(self.calls)]
        db = ChangingDB()
        initial = Preprocessor(db=db, allowed_patches=["16.7"])
        frozen = trainer.freeze_training_window(initial)
        self.assertEqual(len(frozen.process_matches()[0]), 1)
        self.assertEqual(len(frozen.process_matches_ra()[0]), 1)
        self.assertEqual(len(frozen.process_matches_gnn()[0]), 1)
        self.assertEqual(len(frozen._get_matches()), 1)
        self.assertEqual(db.calls, 1)
        refreshed = trainer.freeze_training_window(frozen)
        self.assertEqual(len(refreshed._get_matches()), 2)
        self.assertEqual(db.calls, 2)
        self.assertEqual(len(frozen._get_matches()), 1)

    def test_fingerprint_ignores_clock_but_detects_evidence_and_compact_save_is_atomic(self):
        first = build_interaction_statistics([record()], FakeDB().get_champion_mapping(), patches=["16.7"])
        second = deepcopy(first)
        second["generated_at"] = "different-clock"
        self.assertEqual(trainer.interaction_fingerprint(first), trainer.interaction_fingerprint(second))
        second["tiers"]["GOLD"]["matches"] += 1
        self.assertNotEqual(trainer.interaction_fingerprint(first, "GOLD"), trainer.interaction_fingerprint(second, "GOLD"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            trainer.compact_json(path, {"previous": True})
            self.assertEqual(path.read_text(), '{"previous":true}')
            with patch.object(trainer.os, "replace", side_effect=OSError("full disk")):
                with self.assertRaises(OSError):
                    trainer.compact_json(path, {"replacement": True})
            self.assertEqual(json.loads(path.read_text()), {"previous": True})
            self.assertEqual(list(path.parent.glob(".interactions-*")), [])

    def test_interactive_statistics_and_profiles_use_the_same_frozen_patch_window(self):
        class DB(FakeDB):
            def save_stats_cache(self, value, path=None):
                trainer.compact_json("data/stats_cache.json", value)
            def save_meta_champs(self, value, path=None):
                trainer.compact_json("data/meta_champs.json", value)
        db = DB([record("a"), record("b"), record("outside", patch_version="16.2")])
        prep = trainer.freeze_training_window(Preprocessor(db=db, allowed_patches=["16.7"]))
        stat = StatisticalModel(db)
        with tempfile.TemporaryDirectory() as directory:
            root, previous = Path(directory), Path.cwd()
            try:
                os.chdir(root)
                with patch.object(trainer, "ROLE_DISTRIBUTION_PATH", root / "data/champion_roles.json"):
                    selected = trainer.refresh_training_statistics(prep, stat)
                self.assertEqual([row["match_id"] for row in selected], ["a", "b"])
                self.assertEqual(stat.interactions.document["scope"]["matches"], 2)
                for name in ("champion_roles.json", "interaction_stats.json", "blind_profiles.json"):
                    document = json.loads((root / "data" / name).read_text())
                    self.assertEqual(document["scope"]["matches"], 2)
                    self.assertEqual(document["scope"]["patches"], ["16.7"])
            finally:
                os.chdir(previous)


class InteractionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.app = self.root / "app"
        (self.app / "data").mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / "data/golden_dataset.json"
        (self.app / "data/golden_dataset.json").write_bytes(source.read_bytes())
        self.db = TrainingDB([record(str(index)) for index in range(8)])
        self.args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--epochs", "1", "--qa-min-pass", "0"])
        location = patch.object(pipeline, "APP_DIR", self.app)
        location.start()
        self.addCleanup(location.stop)
        qa = patch("trainer_script.quick_evaluate_model", return_value=1.0)
        qa.start()
        self.addCleanup(qa.stop)

    def fit(self, stage):
        return pipeline.train_outputs(self.args, ["16.7"], self.db, self.db.records, stage, log=lambda _: None)

    def test_artifacts_precede_training_and_publication_preserves_other_rank_scopes(self):
        mapping = self.db.get_champion_mapping()
        old = build_interaction_statistics([record(f"silver:{index}", tier="SILVER", patch_version="16.2")
            for index in range(24)], mapping, patches=["16.2"], tiers=["SILVER"], region="euw1")
        save_interaction_statistics(self.app / "data/interaction_stats.json", old)
        profiles = build_blind_profiles(InteractionStatistics(old, mapping), ["SILVER"], mapping)
        profiles["tier_scopes"] = {tier: value["scope"] for tier, value in old["tiers"].items()}
        trainer.compact_json(self.app / "data/blind_profiles.json", profiles)
        stage = self.root / "stage"
        original_init = DynamicTrainer.__init__
        checked = []
        def initialize(instance, *args, **kwargs):
            self.assertTrue((stage / "data/interaction_stats.json").is_file())
            self.assertTrue((stage / "data/blind_profiles.json").is_file())
            checked.append(args[0])
            return original_init(instance, *args, **kwargs)
        original_calibrate = MetaLearningCalibrator.calibrate
        def calibrate(instance, models, tier):
            self.assertEqual(models["statistical"].interactions.document["scope"]["matches"], 8)
            return original_calibrate(instance, models, tier)
        with patch.object(DynamicTrainer, "__init__", new=initialize), \
             patch.object(MetaLearningCalibrator, "calibrate", new=calibrate):
            reports, outputs = self.fit(stage)
        self.assertEqual(checked, ["GOLD", "MIXED"])
        self.assertIn(Path("data/interaction_stats.json"), outputs)
        self.assertIn(Path("data/blind_profiles.json"), outputs)
        interactions = json.loads((stage / "data/interaction_stats.json").read_text())
        blinds = json.loads((stage / "data/blind_profiles.json").read_text())
        self.assertEqual(interactions["tiers"]["SILVER"]["scope"], old["tiers"]["SILVER"]["scope"])
        self.assertEqual(blinds["tier_scopes"]["SILVER"], old["tiers"]["SILVER"]["scope"])
        self.assertEqual(interactions["tiers"]["GOLD"]["scope"]["patches"], ["16.7"])
        self.assertEqual(reports["GOLD"]["qa"]["gnn"]["interaction_policy"], trainer.GNN_PRIOR_POLICY)
        pipeline.publish_outputs(stage, self.app, outputs)
        self.assertEqual(json.loads((self.app / "data/blind_profiles.json").read_text())["tier_scopes"], blinds["tier_scopes"])

    def test_old_gnn_and_calibration_policies_rebuild_then_matching_evidence_resumes(self):
        progress = UpdateProgress.open(self.root / "journal", {}, lambda: ["16.7"])
        self.args._progress = progress
        self.fit(progress.staging)
        for tier in ("GOLD", "MIXED"):
            progress.state["training"][f"{tier}:gnn"]["report"].pop("interaction_policy")
            calibration = progress.state["training"][f"{tier}:calibration"]
            calibration["report"] = calibration["report"]["weights"]
        progress.save()
        original_gnn, original_calibrate = DynamicTrainer.train_gnn_model, MetaLearningCalibrator.calibrate
        retrained, recalibrated = [], []
        def fit_gnn(instance, graphs, name, *args, **kwargs):
            retrained.append(name)
            self.assertTrue(callable(kwargs.get("interaction_prior_factory")))
            return original_gnn(instance, graphs, name, *args, **kwargs)
        def calibrate(instance, models, tier):
            recalibrated.append(tier)
            return original_calibrate(instance, models, tier)
        with patch.object(DynamicTrainer, "train_gnn_model", new=fit_gnn), \
             patch.object(MetaLearningCalibrator, "calibrate", new=calibrate), \
             patch.object(DynamicTrainer, "train_single_model", side_effect=AssertionError("Unchanged baseline retrained")):
            self.fit(progress.staging)
            self.assertEqual(retrained, ["GOLD_gnn", "MIXED_gnn"])
            self.assertEqual(recalibrated, ["GOLD", "MIXED"])
            retrained.clear()
            recalibrated.clear()
            self.fit(progress.staging)
        self.assertEqual(retrained, [])
        self.assertEqual(recalibrated, [])


if __name__ == "__main__":
    unittest.main()
