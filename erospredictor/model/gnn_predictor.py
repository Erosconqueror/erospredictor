import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from configs import CHAMPION_COUNT

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

def create_match_graph(blue_team, red_team, blue_win=None):
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
    
    return graph_data

def preprocess_matches_for_gnn(data_manager):
    match_ids = data_manager.get_all_match_ids()
    graphs = []
    divisions = []
    
    print(f"Preprocessing {len(match_ids)} matches for GNN...")
    
    for match_id in match_ids:
        match_data = data_manager.get_match(match_id)
        if not match_data:
            continue
            
        data = match_data["data"]
        division = data.get("tier", "UNKNOWN")
        
        blue_team = []
        red_team = []
        
        for i, participant in enumerate(data["participants"]):
            champ_id = participant["championId"]
            champ_index = data_manager.get_champindex_by_id(champ_id)
            
            if champ_index is not None:
                if i < 5:
                    blue_team.append(champ_index)
                else:
                    red_team.append(champ_index)
        
        if len(blue_team) != 5 or len(red_team) != 5:
            continue
            
        blue_win = data["teams"][0]["win"]
        
        graph = create_match_graph(blue_team, red_team, blue_win)
        graphs.append(graph)
        divisions.append(division)
    
    print(f"Successfully created {len(graphs)} graphs")
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