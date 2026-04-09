import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from model.gnn_predictor import LeagueGNN
from configs import MODELS_DIR

def train_single_model(X: list, y: list, divs: list, name: str, in_size: int, ep: int, bs: int, lr: float, m_type: str = "standard", w: list = None):
    """Trains and saves a single model (either standard or role-aware) based on the provided training data."""
    if not X: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    w_t = torch.tensor(w, dtype=torch.float32).unsqueeze(1) if w else torch.ones_like(y_t)

    loader = TorchDataLoader(TensorDataset(x_t, y_t, w_t), batch_size=bs, shuffle=True)
    model = ChampionPredictor(in_size).to(device) if m_type == "standard" else RoleAwareEmbeddingPredictor().to(device)
    
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss(reduction='none')

    model.train()
    for e in range(ep):
        loss_tot = 0
        for bx, by, bw in loader:
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
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

def train_gnn_model(graphs: list, name: str, ep: int, bs: int, lr: float):
    """Trains and saves a Graph Neural Network model."""
    if not graphs: return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = GeoDataLoader(graphs, batch_size=bs, shuffle=True)
    model = LeagueGNN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss(reduction='none')

    model.train()
    for e in range(ep):
        loss_tot = 0
        for batch in loader:
            batch = batch.to(device)
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