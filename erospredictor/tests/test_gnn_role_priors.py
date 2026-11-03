"""Positive trainable relevance gates and legacy-safe GNN checkpoint loading."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import mock_open

try:
    import torch
    from torch_geometric.loader import DataLoader
except ImportError:
    torch = None


def prior():
    return {"counter_weights": [[.6 + .2 * row + .03 * column for column in range(5)] for row in range(5)],
            "synergy_weights": [[0 if row == column else .8 + .1 * row + .02 * column for column in range(5)] for row in range(5)],
            "eligible": {cid: list(range(5)) for cid in range(30)}}


@unittest.skipUnless(torch is not None, "PyTorch and PyTorch Geometric are required")
class GNNRolePriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self):
        from model.gnn_predictor import LeagueGNN
        return LeagueGNN()

    def test_old_state_load_is_strict_and_preserves_the_exact_unweighted_path(self):
        from model.gnn_predictor import predict_gnn
        expected = self.model()
        old = {name: value.clone() for name, value in expected.state_dict().items() if not name.startswith("role_prior_")}
        restored = self.model()
        restored.load_state_dict(old)
        self.assertFalse(restored._role_priors_active)
        self.assertFalse(restored.role_prior_logits.requires_grad)
        blue, red = list(range(5)), list(range(5, 10))
        with patch.object(restored, "_edge_weights", side_effect=AssertionError("Legacy inference must be unweighted")):
            self.assertEqual(predict_gnn(expected, blue, red, "cpu"), predict_gnn(restored, blue, red, "cpu"))
        corrupt = dict(old)
        del corrupt["fc2.weight"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(corrupt)

    def test_edges_follow_target_row_source_column_and_off_meta_or_unknown_nodes_stay_neutral(self):
        from model.gnn_predictor import create_graph
        document = prior()
        document["eligible"][2] = [0]  # Champion 2 in Mid is off-meta.
        model = self.model()
        model.initialize_role_priors(document)
        graph = create_graph([0, 1, 2, 3, -1], [5, 6, 7, 8, 9])
        weights = model._edge_weights(graph.x, graph.edge_index)
        def edge(source, target):
            index = ((graph.edge_index[0] == source) & (graph.edge_index[1] == target)).nonzero().item()
            return weights[index].item()
        self.assertAlmostEqual(edge(8, 0), document["counter_weights"][0][3], places=6)
        self.assertAlmostEqual(edge(0, 8), document["counter_weights"][3][0], places=6)
        self.assertAlmostEqual(edge(1, 0), document["synergy_weights"][0][1], places=6)
        self.assertAlmostEqual(edge(0, 1), document["synergy_weights"][1][0], places=6)
        self.assertEqual(edge(2, 0), 1)
        self.assertEqual(edge(0, 2), 1)
        self.assertEqual(edge(4, 0), 1)
        self.assertEqual(edge(0, 4), 1)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights > 0).all())

    def test_priors_receive_gradients_and_survive_a_strict_checkpoint_roundtrip(self):
        from model.gnn_predictor import create_graph, predict_gnn
        torch.manual_seed(41)
        model = self.model()
        model.initialize_role_priors(prior())
        batch = next(iter(DataLoader([create_graph(list(range(5)), list(range(5, 10))),
                                      create_graph(list(range(1, 6)), list(range(6, 11)))], batch_size=2)))
        model.train()
        model(batch.x, batch.edge_index, batch.batch).sum().backward()
        self.assertTrue(model.role_prior_logits.requires_grad)
        self.assertIsNotNone(model.role_prior_logits.grad)
        self.assertTrue(torch.isfinite(model.role_prior_logits.grad).all())
        self.assertGreater(model.role_prior_logits.grad.abs().sum().item(), 0)
        clone = self.model()
        clone.load_state_dict(model.state_dict())
        self.assertTrue(clone._role_priors_active)
        self.assertTrue(clone.role_prior_logits.requires_grad)
        blue, red = list(range(5)), list(range(5, 10))
        self.assertEqual(predict_gnn(model, blue, red, "cpu"), predict_gnn(clone, blue, red, "cpu"))

    def test_learned_edge_weights_remain_positive_bounded_and_finite(self):
        from model.gnn_predictor import create_graph
        model = self.model()
        model.initialize_role_priors(prior())
        graph = create_graph(list(range(5)), list(range(5, 10)))
        with torch.no_grad():
            model.role_prior_logits[0].fill_(-1000)
            model.role_prior_logits[1].fill_(1000)
        weights = model._edge_weights(graph.x, graph.edge_index)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= .25).all())
        self.assertTrue((weights <= 4).all())

    def test_invalid_priors_cannot_partly_mutate_a_model(self):
        invalid = []
        for field, value in (("counter_weights", [[1] * 5] * 4), ("counter_weights", [[float("nan")] * 5] * 5),
                             ("counter_weights", [[-1] * 5] * 5), ("eligible", {0: [True]}),
                             ("eligible", {1.5: [0]}), ("eligible", None)):
            item = prior()
            item[field] = value
            invalid.append(item)
        model = self.model()
        fingerprint = model.role_prior_fingerprint
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                model.initialize_role_priors(item)
            self.assertEqual(model.role_prior_fingerprint, fingerprint)

    def test_partly_missing_new_checkpoint_metadata_is_not_silently_treated_as_legacy(self):
        model = self.model()
        model.initialize_role_priors(prior())
        state = dict(model.state_dict())
        del state["role_prior_eligible"]
        with self.assertRaises(RuntimeError):
            self.model().load_state_dict(state)

    def test_small_training_uses_the_prior_and_preserves_existing_signature_fields(self):
        from model.train_model import DynamicTrainer
        from model.gnn_predictor import create_graph
        graphs = [create_graph([index, 10, 11, 12, 13], [20, 21, 22, 23, 24], bool(index % 2))
                  for index in range(8)]
        document = prior()
        before = deepcopy(document)
        with tempfile.TemporaryDirectory() as directory:
            trainer = DynamicTrainer("GOLD", output_dir=directory, dataset_size=8,
                                     checkpoint_dir=Path(directory) / "epochs")
            trainer.config.update(epochs=1, batch_size=4)
            result = trainer.train_gnn_model(graphs, "GOLD_gnn", interaction_prior=document)
            self.assertEqual(result["status"], "success")
            checkpoint = torch.load(Path(directory) / "epochs/GOLD_gnn_normal.pth", weights_only=True)
            if "dropout" in checkpoint["signature"]:
                self.assertEqual(checkpoint["signature"]["dropout"], self.model().dropout.p)
            self.assertIn("interaction_prior", checkpoint["signature"])
            model = self.model()
            model.load_state_dict(torch.load(Path(directory) / "GOLD_gnn.pth", weights_only=True))
            self.assertTrue(model._role_priors_active)
            self.assertTrue(model.role_prior_logits.requires_grad)
            self.assertEqual(document, before)

    def test_prior_factory_receives_only_training_indices_before_initialization(self):
        import numpy as np
        from model.train_model import DynamicTrainer
        from model.gnn_predictor import create_graph
        graphs = [create_graph([index, 10, 11, 12, 13], [20, 21, 22, 23, 24], bool(index % 2))
                  for index in range(8)]
        received = []
        derived = []
        def factory(indices):
            received.extend(indices)
            item = prior()
            # This value deliberately depends on labels, so held-out graphs must
            # never be passed to this builder.
            item["counter_weights"][0][0] = .5 + sum(graphs[index].y.item() for index in indices) / len(indices)
            derived.append(item)
            return item
        with tempfile.TemporaryDirectory() as directory:
            trainer = DynamicTrainer("GOLD", output_dir=directory, dataset_size=8,
                                     checkpoint_dir=Path(directory) / "epochs")
            trainer.config.update(epochs=1, batch_size=4)
            result = trainer.train_gnn_model(graphs, "GOLD_gnn", interaction_prior_factory=factory)
            order = np.random.default_rng(trainer.seed).permutation(len(graphs)).tolist()
            self.assertEqual(result["status"], "success")
            self.assertEqual(received, order[1:])
            self.assertNotIn(order[0], received)
            expected = self.model()
            expected.initialize_role_priors(derived[0])
            saved = torch.load(Path(directory) / "epochs/GOLD_gnn_normal.pth", weights_only=True)
            self.assertEqual(saved["signature"]["interaction_prior"], expected.role_prior_fingerprint)
            self.assertEqual(result["interaction_prior"], expected.role_prior_fingerprint)
            self.assertEqual(result["interaction_policy"], "training-only-role-priors-v1")

    def test_neutral_factory_still_reports_the_training_only_policy(self):
        from model.train_model import DynamicTrainer
        from model.gnn_predictor import create_graph
        graphs = [create_graph([index, 10, 11, 12, 13], [20, 21, 22, 23, 24], bool(index % 2))
                  for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            trainer = DynamicTrainer("GOLD", output_dir=directory, dataset_size=8)
            trainer.config.update(epochs=1, batch_size=4)
            report = trainer.train_gnn_model(graphs, "GOLD_gnn", interaction_prior_factory=lambda indices: None)
            self.assertEqual(report["status"], "success")
            self.assertEqual(report["interaction_policy"], "training-only-role-priors-v1")
            self.assertNotIn("interaction_prior", report)

    def test_calibration_applies_the_current_per_model_floor_policy(self):
        import numpy as np
        from types import SimpleNamespace
        from model.train_model import MetaLearningTrainer
        trainer = MetaLearningTrainer()
        predictions, truth = [.4, .6] * 4, [0, 1] * 4
        estimator = SimpleNamespace(fit=lambda features, labels: None, coef_=np.zeros((1, 4)),
                                    score=lambda features, labels: 1)
        scaler = SimpleNamespace(fit_transform=lambda features: features)
        modules = {"sklearn": SimpleNamespace(),
                   "sklearn.linear_model": SimpleNamespace(LogisticRegression=lambda **kwargs: estimator),
                   "sklearn.preprocessing": SimpleNamespace(StandardScaler=lambda: scaler)}
        for enhanced in (False, True):
            with self.subTest(enhanced=enhanced), \
                 patch.dict("sys.modules", modules), \
                 patch.object(trainer, "_softmax", return_value=np.array([0, 1, 0, 0])), \
                 patch("model.train_model.open", mock_open(), create=True), \
                 patch("pickle.dump"), \
                 patch("model.train_model.os.makedirs"):
                report = trainer.calibrate_ensemble(predictions, predictions, predictions, predictions,
                                                     truth, "GOLD", enhanced_statistical=enhanced)
            self.assertEqual(report["status"], "success")
            self.assertAlmostEqual(report["weights"]["gnn"], .16 if enhanced else .24)
            self.assertAlmostEqual(report["weights"]["statistical"], .16)
            self.assertAlmostEqual(report["weights"]["roleaware"], .08)
            self.assertAlmostEqual(sum(report["weights"].values()), 1)

    def test_legacy_epoch_optimizer_can_resume_without_a_prior(self):
        from model.train_model import DynamicTrainer
        from model.gnn_predictor import create_graph
        graphs = [create_graph([index, 10, 11, 12, 13], [20, 21, 22, 23, 24], bool(index % 2))
                  for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = DynamicTrainer("GOLD", output_dir=root, dataset_size=8, checkpoint_dir=root / "epochs")
            trainer.config.update(epochs=2, batch_size=4)
            save = trainer._save_epoch
            def interrupt(*args):
                save(*args)
                raise KeyboardInterrupt
            with patch.object(trainer, "_save_epoch", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                trainer.train_gnn_model(graphs, "GOLD_gnn")
            path = root / "epochs/GOLD_gnn_normal.pth"
            checkpoint = torch.load(path, weights_only=True)
            for key in ("model", "best_state"):
                checkpoint[key] = {name: value for name, value in checkpoint[key].items() if not name.startswith("role_prior_")}
            torch.save(checkpoint, path)
            resumed = DynamicTrainer("GOLD", output_dir=root, dataset_size=8, checkpoint_dir=root / "epochs")
            resumed.config.update(epochs=2, batch_size=4)
            self.assertEqual(resumed.train_gnn_model(graphs, "GOLD_gnn")["epochs_trained"], 2)
            with self.assertRaisesRegex(ValueError, "parameters changed"):
                resumed.train_gnn_model(graphs, "GOLD_gnn", interaction_prior=prior())


if __name__ == "__main__":
    unittest.main()
