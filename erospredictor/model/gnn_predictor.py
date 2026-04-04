import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from configs import CHAMPION_COUNT, CHAMPION_DATA_PATH

class LeagueGNN(nn.Module):
    """Graph Neural Network model for League of Legends matches."""
    
    def __init__(self, champ_dim: int = 32, role_dim: int = 8, hid_dim: int = 64):  
        super().__init__()
        
        self.champ_emb = nn.Embedding(CHAMPION_COUNT, champ_dim)
        self.role_emb = nn.Embedding(10, role_dim)
        
        node_dim = champ_dim + role_dim + 1
        self.conv1 = GCNConv(node_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        
        self.fc1 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc2 = nn.Linear(hid_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.5)  
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim // 2)

    def forward(self, x, edge_idx, batch):
        c_ids, r_ids, t_ids = x[:, 0], x[:, 1], x[:, 2].float().unsqueeze(1)
        
        x = torch.cat([self.champ_emb(c_ids), self.role_emb(r_ids), t_ids], dim=1)
        
        x = F.relu(self.bn1(self.conv1(x, edge_idx)))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_idx))
        x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

def create_graph(blue: list, red: list, win: bool = None, weight: float = 1.0) -> Data:
    """Constructs a graph representation of a match."""
    nodes = []
    for i, cid in enumerate(blue):
        nodes.append([cid, i, 0])
    for i, cid in enumerate(red):
        nodes.append([cid, i + 5, 1])
    
    edges = []
    for i in range(10):
        for j in range(i + 1, 10):
            edges.extend([[i, j], [j, i]])
    
    x = torch.tensor(nodes, dtype=torch.long)
    edge_idx = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    graph = Data(x=x, edge_index=edge_idx, num_nodes=10)
    
    if win is not None:
        graph.y = torch.tensor([[1.0]] if win else [[0.0]], dtype=torch.float32)
        graph.weight = torch.tensor([[weight]], dtype=torch.float32)
    
    return graph

def prep_gnn_matches(db, p_weights: dict) -> tuple:
    """Prepares all matches from the database into PyTorch geometric graphs."""
    matches = db.get_all_matches()
    graphs, divs = [], []
    
    print(f"GNN data prep: Building graphs from {len(matches)} matches...")
    
    with open(CHAMPION_DATA_PATH, 'r', encoding='utf-8') as f:
        c_map = json.load(f)
            
    for m in matches:
        b_raw, r_raw = m.get("blue_team", []), m.get("red_team", [])
        if isinstance(b_raw, str): b_raw = b_raw.strip("{}").split(",")
        if isinstance(r_raw, str): r_raw = r_raw.strip("{}").split(",")

        b_team = [int(c_map[str(cid)]) for cid in b_raw if str(cid) in c_map]
        r_team = [int(c_map[str(cid)]) for cid in r_raw if str(cid) in c_map]
        
        if len(b_team) != 5 or len(r_team) != 5:
            continue
            
        w = p_weights.get(m.get("patch", "UNKNOWN"), 0.5)
        graphs.append(create_graph(b_team, r_team, m.get("blue_win", False), w))
        divs.append(m.get("tier", "UNKNOWN"))
    
    return graphs, divs

def predict_gnn(model, blue: list, red: list, device) -> float:
    """Runs a prediction using the GNN model."""
    model.eval()
    with torch.no_grad():
        graph = create_graph(blue, red).to(device)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        return model(graph.x, graph.edge_index, batch).item()