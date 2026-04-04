import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import CHAMPION_COUNT

class ChampionPredictor(nn.Module):
    """Standard multi-layer perceptron for match prediction."""
    
    def __init__(self, input_size: int = CHAMPION_COUNT * 2, hidden_size: int = 256, drop_rate: float = 0.2):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.drop = nn.Dropout(drop_rate)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        return torch.sigmoid(self.fc3(x))

class RoleAwareEmbeddingPredictor(nn.Module):
    """Predictor utilizing embeddings for both champions and roles."""
    
    def __init__(self, champ_count: int = CHAMPION_COUNT, role_count: int = 10, emb_dim: int = 64, hid_dim: int = 256):
        super().__init__()
        self.c_emb = nn.Embedding(champ_count, emb_dim)
        self.r_emb = nn.Embedding(role_count, emb_dim)
        
        self.fc1 = nn.Linear(10 * emb_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc3 = nn.Linear(hid_dim // 2, 1)
        
        self.drop = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim // 2)

    def forward(self, x):
        bs, device = x.size(0), x.device
        embs = []
        
        for i in range(10): 
            r_tensor = self.r_emb(torch.tensor([i], device=device)).expand(bs, -1)
            c_idx = torch.argmax(x[:, i * CHAMPION_COUNT : (i + 1) * CHAMPION_COUNT], dim=1)
            embs.append(self.c_emb(c_idx) + r_tensor)
        
        x = torch.cat(embs, dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.sigmoid(self.fc3(x))