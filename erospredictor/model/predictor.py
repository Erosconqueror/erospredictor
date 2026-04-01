import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import CHAMPION_COUNT

class ChampionPredictor(nn.Module):
    def __init__(self, input_size=CHAMPION_COUNT*2, hidden_size=256, dropout_rate=0.2):
        super(ChampionPredictor, self).__init__()
        self.input_norm = nn.BatchNorm1d(input_size) #check if good

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        x = self.input_norm(x) #check if good - is good probably xd
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

class RoleAwareEmbeddingPredictor(nn.Module):
    def __init__(self, champion_count=CHAMPION_COUNT, role_count=10, embedding_dim=64, hidden_dim=256):
        super(RoleAwareEmbeddingPredictor, self).__init__()
        
        self.champ_embedding = nn.Embedding(champion_count, embedding_dim)
        self.role_embedding = nn.Embedding(role_count, embedding_dim)
        
        self.fc1 = nn.Linear(10 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        team_embeddings = []
        
        for i in range(10): 

            role_idx = i
            role_embedding = self.role_embedding(torch.tensor([role_idx], device=device)).expand(batch_size, -1)
            
            start_idx = i * CHAMPION_COUNT
            end_idx = start_idx + CHAMPION_COUNT
            position_slice = x[:, start_idx:end_idx]
            
            champ_indices = torch.argmax(position_slice, dim=1)
            champ_embedding = self.champ_embedding(champ_indices)
            
            combined = champ_embedding + role_embedding
            team_embeddings.append(combined)
        
        x = torch.cat(team_embeddings, dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x