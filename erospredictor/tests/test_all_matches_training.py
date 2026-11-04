"""Training-only full snapshots retain every valid selected-patch record."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import update_pipeline as pipeline
from test_update_pipeline import FakeDB, record, install_test_catalog


class FullTrainingTests(unittest.TestCase):
    def args(self, *extra):
        return pipeline.parser().parse_args(["--skip-scrape", "--all-regions", "--all-matches",
            "--patches", "16.7", "--tiers", "GOLD", "--min-matches", "4", "--no-catalog", *extra])

    def test_every_valid_match_reaches_training_and_statistics_without_profile_sampling(self):
        rows = [record(str(i), region="kr" if i % 2 else "euw1") for i in range(13)]
        rows += [record("old", patch_version="16.6"), record("other", tier="BRONZE"), dict(record("invalid"), blue_win=None)]
        db = FakeDB(rows)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            install_test_catalog(root)
            old_journal = root / ".model_training_all/state.json"
            old_journal.parent.mkdir()
            old_journal.write_bytes(b"old capped checkpoint")
            def fit(args, patches, database, selected, stage, log):
                self.assertEqual({row["match_id"] for row in selected}, {str(i) for i in range(13)})
                self.assertEqual(len(selected), 13)
                self.assertEqual({row["region"] for row in selected}, {"euw1", "kr"})
                (stage / "models").mkdir()
                (stage / "models/full.pth").write_bytes(b"test model")
                return {"GOLD": {"matches": len(selected)}}, [Path("models/full.pth")]
            with patch.object(pipeline, "APP_DIR", root), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db), "model.riot": None}), \
                    patch.object(pipeline, "balanced_snapshot", side_effect=AssertionError("Downsampling")), \
                    patch.object(pipeline, "train_outputs", side_effect=fit):
                self.assertEqual(pipeline.run(self.args(), log=lambda _: None), 0)
            manifest = json.loads((root / "data/update_manifest.json").read_text())
            self.assertEqual(manifest["training_patch_counts"], {"16.7": 13})
            self.assertEqual(manifest["plan"]["training_selection"], "all_available")
            self.assertIsNone(manifest["plan"]["dataset_targets"])
            roles = json.loads((root / "data/champion_roles.json").read_text())
            self.assertEqual(roles["tiers"]["GOLD"]["1"]["total"], 13)
            self.assertEqual(old_journal.read_bytes(), b"old capped checkpoint")
            self.assertEqual(db.query["region"], None)

    def test_full_training_rejects_collection_and_contradictory_caps(self):
        for values in (["--all-matches"], ["--skip-scrape", "--all-matches", "--tier-target", "GOLD=10"],
                       ["--skip-scrape", "--all-matches", "--matches-per-tier", "10"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                pipeline._run(pipeline.parser().parse_args(values), log=lambda _: None)

    def test_dry_plan_has_no_rank_cap_or_database_access(self):
        messages = []
        with patch.dict("sys.modules", {"model.data_manager": None}):
            self.assertEqual(pipeline.run(self.args("--dry-run"), log=messages.append), 0)
        plan = json.loads(messages[0])
        self.assertEqual(plan["training_selection"], "all_available")
        self.assertIsNone(plan["dataset_targets"])
