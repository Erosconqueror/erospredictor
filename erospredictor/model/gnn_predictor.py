import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from configs import CHAMPION_COUNT, CHAMPION_DATA_PATH
import json
from pathlib import Path

class LeagueGNN(nn.Module):
    def __init__(self, champion_embedding_dim=32, role_embedding_dim=8, hidden_dim=64):  
        super(LeagueGNN, self).__init__()
        
        self.champ_embedding = nn.Embedding(CHAMPION_COUNT, champion_embedding_dim)
        self.role_embedding = nn.Embedding(10, role_embedding_dim)
        
        node_dim = champion_embedding_dim + role_embedding_dim + 1
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.5)  
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x, edge_index, batch):
        champ_ids = x[:, 0]
        role_ids = x[:, 1]
        team_ids = x[:, 2].float().unsqueeze(1)
        
        champ_emb = self.champ_embedding(champ_ids)
        role_emb = self.role_embedding(role_ids)
        
        x = torch.cat([champ_emb, role_emb, team_ids], dim=1)
        
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.bn2(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def create_match_graph(blue_team, red_team, blue_win=None, weight=1.0):
    num_nodes = 10
    node_features = []
    
    for i, champ_id in enumerate(blue_team):
        node_features.append([champ_id, i, 0])
    
    for i, champ_id in enumerate(red_team):
        node_features.append([champ_id, i + 5, 1])
    
    node_features = torch.tensor(node_features, dtype=torch.long)
    edge_list = []
    
    for i in range(10):
        for j in range(i + 1, 10):
            same_team = (i < 5 and j < 5) or (i >= 5 and j >= 5)
            if same_team:
                edge_list.append([i, j])
                edge_list.append([j, i])
    
    lane_opponents = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    for blue_idx, red_idx in lane_opponents:
        edge_list.append([blue_idx, red_idx])
        edge_list.append([red_idx, blue_idx])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    if blue_win is not None:
        graph_data.y = torch.tensor([[1.0]] if blue_win else [[0.0]], dtype=torch.float32)
        graph_data.weight = torch.tensor([[weight]], dtype=torch.float32)
    
    return graph_data

def preprocess_matches_for_gnn(data_manager, patch_weights):
    matches = data_manager.get_all_matches()
    graphs = []
    divisions = []
    
    print(f"GNN adatfeldolgozas: {len(matches)} meccsbol grafok epitese...")
    
    champ_mapping = {}
    path = Path(CHAMPION_DATA_PATH)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            champ_mapping = json.load(f)
            
    for match_data in matches:
        division = match_data.get("tier", "UNKNOWN")
        patch = match_data.get("patch", "UNKNOWN")
        weight = patch_weights.get(patch, 0.5)
        
        blue_team_raw = match_data.get("blue_team", [])
        red_team_raw = match_data.get("red_team", [])
        blue_win = match_data.get("blue_win", False)
        
        if isinstance(blue_team_raw, str):
            blue_team_raw = blue_team_raw.strip("{}").split(",")
        if isinstance(red_team_raw, str):
            red_team_raw = red_team_raw.strip("{}").split(",")

        blue_team = [int(champ_mapping[str(cid)]) for cid in blue_team_raw if str(cid) in champ_mapping]
        red_team = [int(champ_mapping[str(cid)]) for cid in red_team_raw if str(cid) in champ_mapping]
        
        if len(blue_team) != 5 or len(red_team) != 5:
            continue
            
        graph = create_match_graph(blue_team, red_team, blue_win, weight)
        graphs.append(graph)
        divisions.append(division)
    
    print(f"Sikeresen felepitve {len(graphs)} graf.")
    return graphs, divisions

def predict_match_gnn(model, blue_team, red_team, device):
    model.eval()
    with torch.no_grad():
        graph = create_match_graph(blue_team, red_team, blue_win=None)
        graph = graph.to(device)
        
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        prediction = model(graph.x, graph.edge_index, batch)
        return prediction.item()

def load_gnn_model(device, model_path="models/gnn_model.pth"):
    model = LeagueGNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model