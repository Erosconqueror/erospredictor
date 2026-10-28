"""Offline catalog staging, stable indices and training with new champions."""

import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zlib

import configs
import update_pipeline as pipeline
from services import catalog_update as catalogs
from services.update_progress import UpdateProgress
from test_update_pipeline import FakeDB, record, install_test_catalog


def png_chunk(kind, payload):
    return len(payload).to_bytes(4, "big") + kind + payload + zlib.crc32(kind + payload).to_bytes(4, "big")


PNG = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", bytes.fromhex("00000001000000010802000000"))
       + png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + png_chunk(b"IEND", b""))


def document(ids):
    return {"data": {str(cid): {"key": str(cid), "name": f"Champion{cid}",
            "image": {"full": f"Champion{cid}.png"}} for cid in ids}}


class CatalogTests(unittest.TestCase):
    def test_icon_validation_detects_truncated_payload_and_bad_crc(self):
        self.assertTrue(catalogs.complete_png(PNG))
        for bad in (b"", PNG[:8], PNG[:-1], PNG[:-12], PNG[:45] + b"broken" + PNG[51:]):
            self.assertFalse(catalogs.complete_png(bad))

    def test_append_preserves_every_existing_index_and_updates_names(self):
        mapping = {"266": 0, "103": 1}
        names = {"0": "Aatrox", "1": "Ahri"}
        result = catalogs.merge_catalog(mapping, names, document([999, 103, 888]))
        self.assertEqual(result["mapping"], {"266": 0, "103": 1, "888": 2, "999": 3})
        self.assertEqual(result["names"]["0"], "Aatrox")
        self.assertEqual(result["names"]["1"], "Champion103")
        self.assertEqual(result["count"], 4)
        self.assertEqual(mapping, {"266": 0, "103": 1})
        self.assertEqual(names, {"0": "Aatrox", "1": "Ahri"})

    def test_bad_catalogs_cannot_change_ids_or_escape_icon_directory(self):
        mapping, names = {"266": 0}, {"0": "Aatrox"}
        for bad in ({}, {"data": {}}, document([-1])):
            with self.assertRaises(ValueError):
                catalogs.merge_catalog(mapping, names, bad)
        bad = document([266])
        bad["data"]["266"]["image"]["full"] = "../outside.png"
        with self.assertRaises(ValueError):
            catalogs.merge_catalog(mapping, names, bad)
        bad = document([266, 103])
        bad["data"]["103"]["key"] = "266"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            catalogs.merge_catalog(mapping, names, bad)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            catalogs.merge_catalog({"266": 0, "103": 2}, {"0": "Aatrox", "2": "Ahri"}, document([266]))

    def test_catalog_uses_latest_build_of_the_frozen_patch(self):
        self.assertEqual(catalogs.version_for_patch("16.7", ["16.8.1", "16.7.1", "16.7.10", "16.7.2"]), "16.7.10")
        with self.assertRaises(ValueError):
            catalogs.version_for_patch("16.7", ["16.8.1"])

    def test_catalog_and_missing_icons_are_staged_and_frozen_across_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            (app / "assets/icons").mkdir(parents=True)
            for index in range(10):
                (app / f"assets/icons/{index}.png").write_bytes(PNG)
            before = (app / "data/champion_id.json").read_bytes()
            progress = UpdateProgress.open(app / ".model_update", {}, lambda: ["16.7"])
            with patch.object(catalogs, "fetch_json", side_effect=[["16.7.1"], document(range(1, 12))]) as metadata, \
                    patch.object(catalogs, "urlopen", return_value=io.BytesIO(PNG)) as images:
                catalog, outputs = catalogs.prepare_catalog(app, progress, ["16.7"])
            self.assertEqual(metadata.call_count, 2)
            self.assertEqual(images.call_count, 1)
            self.assertEqual(catalog["count"], 11)
            self.assertEqual(catalog["installed_count"], 10)
            self.assertIn(Path("assets/icons/10.png"), outputs)
            self.assertEqual((app / "data/champion_id.json").read_bytes(), before)
            self.assertFalse((app / "assets/icons/10.png").exists())
            resumed = UpdateProgress.open(app / ".model_update", {}, lambda: [])
            with patch.object(catalogs, "fetch_json", side_effect=AssertionError("Catalog refetched")), \
                    patch.object(catalogs, "urlopen", side_effect=AssertionError("Icon refetched")):
                saved, saved_outputs = catalogs.prepare_catalog(app, resumed, ["16.7"])
            self.assertEqual(saved, catalog)
            self.assertEqual(saved_outputs, outputs)
            (resumed.staging / "catalog.json").write_text("damaged")
            with self.assertRaisesRegex(ValueError, "damaged"):
                catalogs.prepare_catalog(app, resumed, ["16.7"])

    def test_partial_icon_download_retries_from_frozen_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            progress = UpdateProgress.open(app / ".model_update", {}, lambda: ["16.7"])
            with patch.object(catalogs, "fetch_json", side_effect=[["16.7.1"], document([11])]) as metadata:
                with patch.object(catalogs, "urlopen", return_value=io.BytesIO(PNG[:30])):
                    with self.assertRaisesRegex(ConnectionError, "Incomplete champion icon"):
                        catalogs.prepare_catalog(app, progress, ["16.7"])
                with patch.object(catalogs, "urlopen", return_value=io.BytesIO(PNG)):
                    catalog, outputs = catalogs.prepare_catalog(app, progress, ["16.7"])
            self.assertEqual(metadata.call_count, 2)
            self.assertEqual(catalog["count"], 11)
            self.assertIn(Path("assets/icons/10.png"), outputs)
            self.assertFalse((app / "assets/icons/10.png").exists())

    def test_new_champion_cannot_publish_with_only_some_tiers_retrained(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--skip-scrape"])
        with tempfile.TemporaryDirectory() as directory:
            install_test_catalog(directory)
            with patch.object(pipeline, "APP_DIR", Path(directory)), \
                    patch.object(catalogs, "fetch_json", side_effect=[["16.7.1"], document([1, 11])]), \
                    patch.object(catalogs, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(PNG)), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: self.fail("DB opened"))}):
                with self.assertRaisesRegex(pipeline.PermanentUpdateError, "every tier"):
                    pipeline.run(args, log=lambda _: None)
            self.assertEqual(len(json.loads((Path(directory) / "data/champion_id.json").read_text())), 10)

    def test_runtime_metadata_updates_patches_and_count_without_overriding_credentials(self):
        # Load a separate configs module; never change the running app's config.
        spec = importlib.util.spec_from_file_location("_catalog_test_configs", Path(configs.__file__))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            module.__file__ = str(Path(directory) / "configs.py")
            install_test_catalog(directory)
            before = {key: getattr(module, key) for key in ("RIOT_API_KEY", "DB_PASS", "DB_HOST")}
            runtime = {"ALLOWED_PATCHES": ["16.20", "16.21"], "CHAMPION_COUNT": 999,
                       "CHAMPION_CATALOG_VERSION": "16.21.1", "RIOT_API_KEY": "ignored", "DB_HOST": "ignored"}
            (Path(directory) / "data/runtime_config.json").write_text(json.dumps(runtime))
            module._load_runtime_config()
            self.assertEqual(module.ALLOWED_PATCHES, ["16.20", "16.21"])
            self.assertEqual(module.CHAMPION_COUNT, 10)
            self.assertEqual(module.CHAMPION_CATALOG_VERSION, "16.21.1")
            self.assertTrue(all(getattr(module, key) == value for key, value in before.items()))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"), "ML packages required")
class NewChampionTrainingTests(unittest.TestCase):
    def test_all_architectures_accept_new_index_and_scope_restores_after_exception(self):
        import torch
        from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
        from model.gnn_predictor import LeagueGNN, create_graph
        from torch_geometric.data import Batch
        previous = configs.CHAMPION_COUNT
        count = previous + 1
        with self.assertRaisesRegex(RuntimeError, "scope exits"):
            with catalogs.champion_dimensions(count):
                rw = ChampionPredictor().eval()
                ra = RoleAwareEmbeddingPredictor().eval()
                gnn = LeagueGNN().eval()
                self.assertEqual(rw.fc1.in_features, 2 * count)
                self.assertEqual(ra.c_emb.num_embeddings, count)
                self.assertEqual(gnn.champ_emb.num_embeddings, count)
                raise RuntimeError("scope exits")
        self.assertEqual(configs.CHAMPION_COUNT, previous)
        x = torch.zeros((1, 10 * count))
        x[0, count - 1] = 1
        graph = Batch.from_data_list([create_graph([count - 1, 1, 2, 3, 4], [5, 6, 7, 8, 9], True, 1)])
        with torch.no_grad():
            self.assertTrue(torch.isfinite(ra(x)).all().item())
            self.assertTrue(torch.isfinite(rw(torch.zeros((1, 2 * count)))).all().item())
            self.assertTrue(torch.isfinite(gnn(graph.x, graph.edge_index, graph.batch)).all().item())

    def test_grown_catalog_trains_resumes_and_publishes_with_models_and_config(self):
        import torch
        from model.train_model import DynamicTrainer
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        source = Path(__file__).parent
        count = configs.CHAMPION_COUNT + 1
        rows = [record(str(i)) for i in range(8)]
        for row in rows:
            row["blue_team"][0] = 999999
        db = FakeDB(rows)
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--skip-scrape",
                                            "--min-matches", "4", "--epochs", "1", "--tier-target", "GOLD=8"])
        finish, fit = UpdateProgress.finish_unit, DynamicTrainer.train_single_model
        trained, fail_once = [], True
        def checkpoint_then_crash(progress, key, path, report):
            nonlocal fail_once
            finish(progress, key, path, report)
            if key == "GOLD:roleweighted" and fail_once:
                fail_once = False
                raise ConnectionError("simulated interrupted new-catalog training")
        def track(item, X, y, name, *a, **kw):
            trained.append(name)
            return fit(item, X, y, name, *a, **kw)
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            (app / "data").mkdir()
            for name in ("champion_id.json", "champion_names.json", "golden_dataset.json"):
                shutil.copy2(source / "data" / name, app / "data" / name)
            original_names = (app / "data/champion_names.json").read_bytes()
            with patch.object(pipeline, "APP_DIR", app), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db)}), \
                    patch.object(configs, "TARGET_TIERS", ["GOLD"]), \
                    patch.object(catalogs, "fetch_json", side_effect=[["16.7.1"], document([266, 999999])]) as metadata, \
                    patch.object(catalogs, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(PNG)), \
                    patch("trainer_script.quick_evaluate_model", return_value=1.0), \
                    patch.object(UpdateProgress, "finish_unit", new=checkpoint_then_crash), \
                    patch.object(DynamicTrainer, "train_single_model", new=track):
                with self.assertRaises(ConnectionError):
                    pipeline.run(args, log=lambda _: None)
                self.assertEqual((app / "data/champion_names.json").read_bytes(), original_names)
                self.assertFalse((app / "data/runtime_config.json").exists())
                self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertEqual(metadata.call_count, 2)
            self.assertEqual(trained.count("GOLD_roleweighted"), 1)
            self.assertEqual(len(list((app / "models").glob("*.pth"))), 6)
            ids = json.loads((app / "data/champion_id.json").read_text())
            self.assertEqual(ids["266"], 0)
            self.assertEqual(ids["999999"], count - 1)
            runtime = json.loads((app / "data/runtime_config.json").read_text())
            self.assertEqual(runtime["CHAMPION_COUNT"], count)
            self.assertEqual(runtime["ALLOWED_PATCHES"], ["16.7"])
            self.assertEqual(runtime["DATASET_TARGETS"]["GOLD"], 8)
            weights = torch.load(app / "models/GOLD_roleaware.pth", weights_only=True)
            self.assertEqual(weights["c_emb.weight"].shape[0], count)
            self.assertTrue((app / f"assets/icons/{count - 1}.png").is_file())
            self.assertEqual(json.loads((app / ".model_update/state.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
