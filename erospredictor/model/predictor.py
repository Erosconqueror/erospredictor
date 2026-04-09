import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import CHAMPION_COUNT

class ChampionPredictor(nn.Module):
    """Long explanation: This is a standard feedforward neural network designed to predict the outcome of a League of Legends match based on the champion picks for both teams.
    The input layer takes a flattened vector representing the presence and role-based weighting of each champion for both the blue and red teams (hence the input size of CHAMPION_COUNT * 2).
    This architecture was chosen for its simplicity and effectiveness in handling tabular data, and as well to serve as a strong baseline and to serve as a balancing factor against more complex models like the GNN.
    while the dropout and batch normalization help prevent overfitting and improve generalization to unseen matchups."""
    
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
    """This model uses separate embeddings for champions and their roles, allowing it to learn more nuanced interactions between specific champions in specific roles.
       The input is a flattened one-hot encoding of champion picks by role, which allows the model to capture positional information.
       The architecture is similar to the ChampionPredictor but incorporates embedding layers and sums the champion and role embeddings for each position before passing them through the fully connected layers.
       This design was an improvement on the previous model, and gave promising results, in some rare cases can give more valid result than the GNN (we are tlaking like 1 champion out of 10 recommendation),
       so it is kept with its predecessor to smoothen out the rough predictions and give more accurate and reliable data """
    
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