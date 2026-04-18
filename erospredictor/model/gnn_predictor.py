import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from configs import CHAMPION_COUNT, CHAMPION_DATA_PATH, DIVISION_WEIGHTS

class LeagueGNN(nn.Module):
    """This GNN model represents each match as a graph where nodes correspond to champions in specific roles, and edges represent interactions between them.
       Each node's features are derived from champion embeddings, role embeddings, and a team indicator.
       It was chosen because it is the best representation of a LoL champion draft.
       The dropout rate is there to prevent the model from becoming too deterministic"""
    
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


def predict_gnn(model, blue: list, red: list, device) -> float:
    """Runs a prediction using the GNN model."""
    model.eval()
    with torch.no_grad():
        graph = create_graph(blue, red).to(device)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        return model(graph.x, graph.edge_index, batch).item()