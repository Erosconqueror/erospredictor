import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from model.gnn_predictor import LeagueGNN
from configs import MODELS_DIR, CHAMPION_COUNT

class DynamicTrainer:
    """Trainer with dynamic hyperparameters, early stopping, and LR scheduling."""
    
    DATASET_SIZES = {
        "GRANDMASTER": 10799, "CHALLENGER": 12011, "MASTER": 89630,
        "DIAMOND": 34917, "PLATINUM": 29099, "EMERALD": 23812,
        "GOLD": 25912, "SILVER": 26362, "BRONZE": 21428, "IRON": 18814
    }
    
    def __init__(self, div: str):
        self.div = div
        self.config = self._get_config(div)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_config(self, div: str) -> dict:
        """Get adaptive hyperparameters based on dataset size."""
        size = self.DATASET_SIZES.get(div, 20000)
        
        if size < 5000:
            bs, ep, lr = 16, 50, 0.0005
        elif size < 20000:
            bs, ep, lr = 32, 40, 0.001
        elif size < 50000:
            bs, ep, lr = 64, 25, 0.0015
        else:
            bs, ep, lr = 128, 15, 0.0015
        
        return {"batch_size": bs, "epochs": ep, "learning_rate": lr, "dataset_size": size}
    
    def train_single_model(self, X: list, y: list, name: str, in_size: int, 
                          m_type: str = "standard", w: list = None) -> dict:
        """Train model with dynamic hyperparameters and early stopping."""
        if not X:
            return {"status": "failed", "reason": "no_data"}
        
        config = self.config
        ep, bs, lr = config["epochs"], config["batch_size"], config["learning_rate"]
        
        print(f"[{name}] BS={bs}, EP={ep}, LR={lr:.6f}, Dataset={len(X)}")
        
        x_t = torch.tensor(np.array(X, dtype=np.float32))
        y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(1)
        w_t = torch.tensor(np.array(w, dtype=np.float32)).unsqueeze(1) if w else torch.ones_like(y_t)
        
        val_size = max(1, int(0.2 * len(X)))
        val_indices = np.random.choice(len(X), val_size, replace=False)
        train_indices = np.array([i for i in range(len(X)) if i not in val_indices])
        
        train_ds = TensorDataset(x_t[train_indices], y_t[train_indices], w_t[train_indices])
        loader = TorchDataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        
        model = ChampionPredictor(in_size).to(self.device) if m_type == "standard" \
                else RoleAwareEmbeddingPredictor().to(self.device)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.BCELoss(reduction='none')
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        best_state = None
        
        for epoch in range(ep):
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
            
            if self.patience_counter >= 15:
                print(f"  Early stop at epoch {epoch}")
                break
            
            if self.patience_counter >= 10:
                for pg in opt.param_groups:
                    pg['lr'] *= 0.5
        
        if best_state:
            model.load_state_dict(best_state)
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pth"))
        
        return {"status": "success", "epochs_trained": epoch + 1, "best_loss": float(self.best_loss)}
    
    def train_gnn_model(self, graphs: list, name: str) -> dict:
        """Train GNN model with dynamic hyperparameters."""
        if not graphs:
            return {"status": "failed", "reason": "no_data"}
        
        config = self.config
        ep, bs, lr = config["epochs"], max(8, config["batch_size"] // 4), config["learning_rate"] * 0.3
        
        print(f"[{name}] BS={bs}, EP={ep}, LR={lr:.6f}")
        
        loader = GeoDataLoader(graphs, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        
        model = LeagueGNN().to(self.device)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.BCELoss(reduction='none')
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        best_state = None
        
        for epoch in range(ep):
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
            
            if train_loss < self.best_loss - 1e-5:
                self.best_loss = train_loss
                self.patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"  Ep {epoch}/{ep} TL={train_loss:.4f} P={self.patience_counter}/15")
            
            if self.patience_counter >= 15:
                print(f"  Early stop at epoch {epoch}")
                break
            
            if self.patience_counter >= 10:
                for pg in opt.param_groups:
                    pg['lr'] *= 0.5
        
        if best_state:
            model.load_state_dict(best_state)
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pth"))
        
        return {"status": "success", "epochs_trained": epoch + 1, "best_loss": float(self.best_loss)}


class MetaLearningTrainer:
    """Trains meta-model to learn optimal ensemble weights."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def calibrate_ensemble(self, gnn_preds: list, rw_preds: list, ra_preds: list, 
                          stat_preds: list, ground_truth: list, div: str) -> dict:
        """Train logistic regression meta-model to learn ensemble weights."""
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
        """Convert raw weights to probabilities."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
