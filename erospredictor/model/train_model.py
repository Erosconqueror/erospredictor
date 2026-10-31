import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import tempfile
import pickle
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from model.gnn_predictor import LeagueGNN
from configs import MODELS_DIR, CHAMPION_COUNT

class DynamicTrainer:
    """Trainer with dynamic hyperparameters, early stopping, and fallback generalization."""
    
    DATASET_SIZES = {
        "GRANDMASTER": 10799, "CHALLENGER": 12011, "MASTER": 89630,
        "DIAMOND": 34917, "PLATINUM": 29099, "EMERALD": 23812,
        "GOLD": 25912, "SILVER": 26362, "BRONZE": 21428, "IRON": 18814
    }
    
    def __init__(self, div: str, output_dir=None, dataset_size=None, seed=42, checkpoint_dir=None):
        self.div = div
        self.output_dir = os.fspath(output_dir) if output_dir is not None else MODELS_DIR
        self.dataset_size = dataset_size
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.config = self._get_config(div)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_config(self, div: str) -> dict:
        """Get adaptive hyperparameters based on dataset size."""
        size = self.dataset_size if self.dataset_size is not None else self.DATASET_SIZES.get(div, 20000)
        
        if size < 5000:
            bs, ep, lr = 16, 50, 0.0005
        elif size < 20000:
            bs, ep, lr = 32, 40, 0.001
        elif size < 50000:
            bs, ep, lr = 64, 25, 0.0015
        else:
            bs, ep, lr = 128, 15, 0.0015
        
        return {"batch_size": bs, "epochs": ep, "learning_rate": lr, "dataset_size": size}

    @staticmethod
    def _atomic_torch_save(value, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".training-", delete=False) as output:
                temporary = Path(output.name)
                torch.save(value, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _resume(self, name, fallback, model, opt, signature):
        self.best_loss, self.patience_counter = float('inf'), 0
        if self.checkpoint_dir is None:
            return 0, None
        path = self.checkpoint_dir / f"{name}_{'fallback' if fallback else 'normal'}.pth"
        if not path.is_file():
            return 0, None
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except (EOFError, pickle.UnpicklingError, RuntimeError):
            print(f"[{name}] A korszakmentés sérült; csak ezt a modellt kezdi újra.")
            return 0, None
        if state.get("signature") != signature:
            raise ValueError(f"Training checkpoint parameters changed for {name}; use --fresh.")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        self.best_loss, self.patience_counter = state["best_loss"], state["patience"]
        torch.set_rng_state(state["rng"])
        if self.device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"[{name}] Folytatás a(z) {state['next_epoch']}. mentett korszak után.")
        return state["next_epoch"], state["best_state"]

    def _save_epoch(self, name, fallback, model, opt, signature, epoch, best_state):
        if self.checkpoint_dir is None:
            return
        self._atomic_torch_save({"signature": signature, "next_epoch": epoch + 1,
            "model": model.state_dict(), "optimizer": opt.state_dict(), "best_state": best_state,
            "best_loss": self.best_loss, "patience": self.patience_counter,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None},
            self.checkpoint_dir / f"{name}_{'fallback' if fallback else 'normal'}.pth")
    
    def train_single_model(self, X: list, y: list, name: str, in_size: int, 
                           m_type: str = "standard", w: list = None, fallback: bool = False) -> dict:
        """Trains model with dynamic hyperparameters, early stopping, and optional fallback generalization."""
        if len(X) < 4:
            return {"status": "failed", "reason": "no_data"}
        
        config = self.config
        ep = 15 if fallback else config["epochs"]
        bs = config["batch_size"]
        lr = config["learning_rate"]
        weight_decay = 1e-3 if fallback else 1e-5
        
        print(f"[{name}] BS={bs}, EP={ep}, LR={lr:.6f}, WD={weight_decay}, Fallback={fallback}, Dataset={len(X)}")
        
        x_t = torch.tensor(np.array(X, dtype=np.float32))
        y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(1)
        w_t = torch.tensor(np.array(w, dtype=np.float32)).unsqueeze(1) if w else torch.ones_like(y_t)
        
        val_size = max(1, int(0.2 * len(X)))
        order = np.random.default_rng(self.seed).permutation(len(X))
        val_indices, train_indices = order[:val_size], order[val_size:]
        
        train_ds = TensorDataset(x_t[train_indices], y_t[train_indices], w_t[train_indices])
        loader = TorchDataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                                 pin_memory=self.device.type == "cuda", drop_last=len(train_ds) % bs == 1)
        
        model = ChampionPredictor(in_size).to(self.device) if m_type == "standard" \
                else RoleAwareEmbeddingPredictor().to(self.device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = nn.BCELoss(reduction='none')
        
        signature = {"kind": m_type, "count": len(X), "size": in_size,
                     "epochs": ep, "batch_size": bs, "lr": lr, "wd": weight_decay, "seed": self.seed}
        start, best_state = self._resume(name, fallback, model, opt, signature)
        epochs_trained = start
        for epoch in range(start, ep):
            if self.patience_counter >= 15:
                break
            model.train()
            train_loss = 0.0
            for bx, by, bw in loader:
                bx, by, bw = bx.to(self.device), by.to(self.device), bw.to(self.device)
                opt.zero_grad()
                out = model(bx)
                loss = (crit(out, by) * bw).mean()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            
            train_loss /= len(loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx in val_indices:
                    x_val = x_t[idx:idx+1].to(self.device)
                    y_val = y_t[idx:idx+1].to(self.device)
                    w_val = w_t[idx:idx+1].to(self.device)
                    out = model(x_val)
                    loss = (crit(out, y_val) * w_val).mean()
                    val_loss += loss.item()
            
            val_loss /= len(val_indices)
            
            if val_loss < self.best_loss - 1e-5:
                self.best_loss = val_loss
                self.patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"  Ep {epoch}/{ep} TL={train_loss:.4f} VL={val_loss:.4f} P={self.patience_counter}/15")
            
            if self.patience_counter >= 10:
                for pg in opt.param_groups:
                    pg['lr'] *= 0.5
            self._save_epoch(name, fallback, model, opt, signature, epoch, best_state)
            epochs_trained = epoch + 1
        
        if best_state:
            model.load_state_dict(best_state)
        
        if not best_state:
            return {"status": "failed", "reason": "non_finite_validation_loss"}
        os.makedirs(self.output_dir, exist_ok=True)
        self._atomic_torch_save(model.state_dict(), os.path.join(self.output_dir, f"{name}.pth"))
        
        return {"status": "success", "epochs_trained": epochs_trained, "best_loss": float(self.best_loss)}
    
    def train_gnn_model(self, graphs: list, name: str, fallback: bool = False,
                        interaction_prior=None, interaction_prior_factory=None) -> dict:
        """Trains GNN model with dynamic hyperparameters and optional fallback generalization."""
        if len(graphs) < 4:
            return {"status": "failed", "reason": "no_data"}
        
        config = self.config
        ep = 15 if fallback else config["epochs"]
        bs = max(8, config["batch_size"] // 4)
        lr = config["learning_rate"] * 0.3
        weight_decay = 1e-3 if fallback else 1e-5
        
        print(f"[{name}] BS={bs}, EP={ep}, LR={lr:.6f}, WD={weight_decay}, Fallback={fallback}")
        
        order = np.random.default_rng(self.seed).permutation(len(graphs))
        val_size = max(1, int(len(graphs) * 0.2))
        val_graphs = [graphs[i] for i in order[:val_size]]
        train_graphs = [graphs[i] for i in order[val_size:]]
        if interaction_prior_factory is not None:
            if interaction_prior is not None or not callable(interaction_prior_factory):
                raise ValueError("A GNN priorhoz egy előre számolt érték vagy egy tanítómintára szűrő factory adható meg.")
            # Labels from validation graphs must not influence prior learning.
            interaction_prior = interaction_prior_factory([int(index) for index in order[val_size:]])
        loader = GeoDataLoader(train_graphs, batch_size=bs, shuffle=True, num_workers=0,
                              pin_memory=self.device.type == "cuda", drop_last=len(train_graphs) % bs == 1)
        val_loader = GeoDataLoader(val_graphs, batch_size=bs, shuffle=False, num_workers=0)
        
        model = LeagueGNN().to(self.device)
        if interaction_prior is not None:
            model.initialize_role_priors(interaction_prior)
        # Frozen neutral priors are excluded, preserving legacy optimizer groups.
        opt = optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad),
                         lr=lr, weight_decay=weight_decay)
        crit = nn.BCELoss(reduction='none')
        
        signature = {"kind": "gnn", "count": len(graphs), "epochs": ep,
                     "batch_size": bs, "lr": lr, "wd": weight_decay, "seed": self.seed}
        if interaction_prior is not None:
            signature["interaction_prior"] = model.role_prior_fingerprint
        start, best_state = self._resume(name, fallback, model, opt, signature)
        epochs_trained = start
        for epoch in range(start, ep):
            if self.patience_counter >= 15:
                break
            model.train()
            train_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = (crit(out, batch.y) * batch.weight).mean()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            
            train_loss /= len(loader)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    val_loss += (crit(out, batch.y) * batch.weight).sum().item()
            val_loss /= len(val_graphs)
            if val_loss < self.best_loss - 1e-5:
                self.best_loss = val_loss
                self.patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"  Ep {epoch}/{ep} TL={train_loss:.4f} VL={val_loss:.4f} P={self.patience_counter}/15")
            
            if self.patience_counter >= 10:
                for pg in opt.param_groups:
                    pg['lr'] *= 0.5
            self._save_epoch(name, fallback, model, opt, signature, epoch, best_state)
            epochs_trained = epoch + 1
        
        if best_state:
            model.load_state_dict(best_state)
        
        if not best_state:
            return {"status": "failed", "reason": "non_finite_validation_loss"}
        os.makedirs(self.output_dir, exist_ok=True)
        self._atomic_torch_save(model.state_dict(), os.path.join(self.output_dir, f"{name}.pth"))
        
        report = {"status": "success", "epochs_trained": epochs_trained, "best_loss": float(self.best_loss)}
        if interaction_prior is not None:
            report["interaction_prior"] = signature["interaction_prior"]
        if interaction_prior_factory is not None:
            report["interaction_policy"] = "training-only-role-priors-v1"
        elif interaction_prior is not None:
            report["interaction_policy"] = "provided-role-priors-v1"
        return report

class MetaLearningTrainer:
    """Trains meta-model to learn optimal ensemble weights."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def calibrate_ensemble(self, gnn_preds: list, rw_preds: list, ra_preds: list, 
                          stat_preds: list, ground_truth: list, div: str, enhanced_statistical=False) -> dict:
        """Trains logistic regression meta-model to learn ensemble weights."""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"status": "failed", "reason": "sklearn_not_available"}
        
        X = np.column_stack([gnn_preds, rw_preds, ra_preds, stat_preds])
        y = np.array(ground_truth)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        meta_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        meta_model.fit(X_scaled, y)
        
        raw_weights = meta_model.coef_[0]
        weights = self._softmax(raw_weights)
        
        weights_dict = {
            "gnn": float(weights[0]),
            "roleweighted": float(weights[1]),
            "roleaware": float(weights[2]),
            "statistical": float(weights[3])
        }
        from services.ensemble_weights import bounded_weights, ensemble_minimums
        weights_dict = bounded_weights(weights_dict, minimums=ensemble_minimums(enhanced_statistical))
        
        os.makedirs("models", exist_ok=True)
        import pickle
        with open(f"models/{div}_meta_calibrator.pkl", 'wb') as f:
            pickle.dump({"weights": weights_dict, "scaler": scaler, "model": meta_model}, f)
        
        score = meta_model.score(X_scaled, y)
        
        return {
            "status": "success",
            "weights": weights_dict,
            "accuracy": float(score),
            "div": div
        }
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Converts raw weights to probabilities."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
