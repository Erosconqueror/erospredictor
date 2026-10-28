"""Small real CPU training/checkpoint roundtrips; never trains shipped weights."""

import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import torch
    import torch_geometric
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch and PyTorch Geometric are required")
class NeuralTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_architectures_train_and_reload_in_isolated_output(self):
        from configs import CHAMPION_COUNT
        from model.train_model import DynamicTrainer
        from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
        from model.gnn_predictor import LeagueGNN, create_graph, predict_gnn
        labels = [float(i % 2) for i in range(8)]
        teams = [([i, 10, 11, 12, 13], [20, 21, 22, 23, 24]) for i in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("roleweighted", "roleaware", "gnn"):
                with self.subTest(kind=kind):
                    trainer = DynamicTrainer("MIXED", output_dir=directory, dataset_size=8)
                    trainer.config.update(epochs=1, batch_size=4)
                    if kind == "gnn":
                        graphs = [create_graph(b, r, bool(y)) for (b, r), y in zip(teams, labels)]
                        result = trainer.train_gnn_model(graphs, kind)
                        model = LeagueGNN()
                    else:
                        factor = 2 if kind == "roleweighted" else 10
                        features = []
                        for blue, red in teams:
                            row = [0.0] * (CHAMPION_COUNT * factor)
                            for side, champions in enumerate((blue, red)):
                                for role, cid in enumerate(champions):
                                    segment = side if factor == 2 else side * 5 + role
                                    row[segment * CHAMPION_COUNT + cid] = 1.0
                            features.append(row)
                        result = trainer.train_single_model(
                            features, labels, kind, CHAMPION_COUNT * factor,
                            "standard" if factor == 2 else "roleaware", [1.0] * 8)
                        model = ChampionPredictor() if factor == 2 else RoleAwareEmbeddingPredictor()
                    self.assertEqual(result["status"], "success")
                    self.assertTrue(math.isfinite(result["best_loss"]))
                    model.load_state_dict(torch.load(Path(directory) / f"{kind}.pth", weights_only=True))
                    model.eval()
                    with torch.no_grad():
                        probability = (predict_gnn(model, *teams[0], "cpu") if kind == "gnn"
                                       else model(torch.tensor(features[:1])).item())
                    self.assertTrue(0 <= probability <= 1)

    def test_epoch_resume_restores_optimizer_and_rng_for_identical_final_weights(self):
        from configs import CHAMPION_COUNT
        from model.train_model import DynamicTrainer
        from model.gnn_predictor import create_graph
        labels = [float(i % 2) for i in range(8)]
        teams = [([i, 10, 11, 12, 13], [20, 21, 22, 23, 24]) for i in range(8)]
        for kind in ("roleweighted", "roleaware", "gnn"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                features = []
                if kind != "gnn":
                    factor = 2 if kind == "roleweighted" else 10
                    for blue, red in teams:
                        row = [0.0] * (CHAMPION_COUNT * factor)
                        for side, champions in enumerate((blue, red)):
                            for role, cid in enumerate(champions):
                                segment = side if factor == 2 else side * 5 + role
                                row[segment * CHAMPION_COUNT + cid] = 1.0
                        features.append(row)
                graphs = [create_graph(b, r, bool(y)) for (b, r), y in zip(teams, labels)]

                def trainer(folder, resume):
                    item = DynamicTrainer("GOLD", output_dir=root / folder, dataset_size=8,
                                          checkpoint_dir=root / "epochs" if resume else None)
                    item.config.update(epochs=3, batch_size=4)
                    return item

                def fit(item):
                    return (item.train_gnn_model(graphs, kind) if kind == "gnn" else
                            item.train_single_model(features, labels, kind, CHAMPION_COUNT * factor,
                                "standard" if factor == 2 else "roleaware", [1.0] * 8))

                torch.manual_seed(123)
                fit(trainer("baseline", False))
                interrupted = trainer("resumed", True)
                save_epoch = interrupted._save_epoch

                def crash_after_saved_epoch(*args, **kwargs):
                    save_epoch(*args, **kwargs)
                    raise KeyboardInterrupt

                torch.manual_seed(123)
                with patch.object(interrupted, "_save_epoch", side_effect=crash_after_saved_epoch):
                    with self.assertRaises(KeyboardInterrupt):
                        fit(interrupted)
                resumed = trainer("resumed", True)
                torch.manual_seed(999)
                with patch.object(resumed, "_save_epoch", wraps=resumed._save_epoch) as save:
                    self.assertEqual(fit(resumed)["epochs_trained"], 3)
                self.assertEqual(save.call_count, 2)
                expected = torch.load(root / f"baseline/{kind}.pth", weights_only=True)
                actual = torch.load(root / f"resumed/{kind}.pth", weights_only=True)
                self.assertEqual(expected.keys(), actual.keys())
                self.assertTrue(all(torch.equal(expected[key], actual[key]) for key in expected))

    def test_failed_checkpoint_write_keeps_the_previous_file(self):
        from model.train_model import DynamicTrainer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(b"previous checkpoint")
            def fail(value, output):
                output.write(b"partial new checkpoint")
                raise OSError("simulated write failure")
            with patch("model.train_model.torch.save", side_effect=fail):
                with self.assertRaises(OSError):
                    DynamicTrainer._atomic_torch_save({}, path)
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).glob(".training-*")), [])


if __name__ == "__main__":
    unittest.main()
