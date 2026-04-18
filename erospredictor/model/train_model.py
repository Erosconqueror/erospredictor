import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from model.gnn_predictor import LeagueGNN
from configs import MODELS_DIR

def train_single_model(X: list, y: list, divs: list, name: str, in_size: int, ep: int, bs: int, lr: float, m_type: str = "standard", w: list = None):
    """
    Trains and saves a single MLP model using optimized data loading.
    """
    if not X:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] Training started on device: {device}")
    
    print(f"[{name}] Converting tensors via numpy...")
    x_t = torch.tensor(np.array(X, dtype=np.float32))
    y_t = torch.tensor(np.array(y, dtype=np.float32)).unsqueeze(1)
    
    if w:
        w_t = torch.tensor(np.array(w, dtype=np.float32)).unsqueeze(1)
    else:
        w_t = torch.ones_like(y_t)

    dataset = TensorDataset(x_t, y_t, w_t)
    
    print(f"[{name}] Initializing dataloader with 4 workers...")
    loader = TorchDataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    
    model = ChampionPredictor(in_size).to(device) if m_type == "standard" else RoleAwareEmbeddingPredictor().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss(reduction='none')

    print(f"[{name}] Starting training loop for {ep} epochs...")
    model.train()
    for e in range(ep):
        loss_tot = 0
        for bx, by, bw in loader:
            bx, by, bw = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), bw.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(bx)
            loss = (crit(out, by) * bw).mean()
            loss.backward()
            opt.step()
            loss_tot += loss.item()
            
        if e % 10 == 0 or e == ep - 1:
            print(f"Epoch {e}/{ep} - Loss: {loss_tot/len(loader):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pth"))
    print(f"[{name}] Training complete. Model saved to {MODELS_DIR}.")

def train_gnn_model(graphs: list, name: str, ep: int, bs: int, lr: float):
    """
    Trains and saves a Graph Neural Network model using geometric data loading.
    """
    if not graphs:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] GNN Training started on device: {device}")
    
    print(f"[{name}] Initializing geometric dataloader with 4 workers...")
    loader = GeoDataLoader(graphs, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    
    model = LeagueGNN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss(reduction='none')

    print(f"[{name}] Starting GNN training loop for {ep} epochs...")
    model.train()
    for e in range(ep):
        loss_tot = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = (crit(out, batch.y) * batch.weight).mean()
            loss.backward()
            opt.step()
            loss_tot += loss.item()
            
        if e % 10 == 0 or e == ep - 1:
            print(f"Epoch {e}/{ep} - Loss: {loss_tot/len(loader):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{name}.pth"))
    print(f"[{name}] GNN Training complete. Model saved to {MODELS_DIR}.")