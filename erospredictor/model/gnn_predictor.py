import json
import hashlib
import math
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
       Dropout regularizes training; it is disabled during inference."""
    
    def __init__(self, champ_dim: int = 32, role_dim: int = 8, hid_dim: int = 64):  
        super().__init__()
        
        self.champ_emb = nn.Embedding(CHAMPION_COUNT, champ_dim)
        self.role_emb = nn.Embedding(10, role_dim)
        
        node_dim = champ_dim + role_dim + 1
        self.conv1 = GCNConv(node_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        
        self.fc1 = nn.Linear(hid_dim, hid_dim // 2)
        self.fc2 = nn.Linear(hid_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim // 2)

        # Neutral defaults leave old checkpoints on the exact unweighted path.
        self.role_prior_logits = nn.Parameter(torch.zeros(2, 5, 5), requires_grad=False)
        self.register_buffer("role_prior_eligible", torch.zeros(CHAMPION_COUNT, 5, dtype=torch.bool))
        self.register_buffer("role_prior_enabled", torch.tensor(False))
        self._role_priors_active = False

    def initialize_role_priors(self, prior):
        """Initialize trainable positive message gates from training-only evidence.

        Rows refer to the receiving champion's role, columns to the source's
        role. Relative percentages are relevance, never signed message weights.
        Missing/off-meta champion-role endpoints retain a neutral gate of one.
        """
        if not isinstance(prior, dict):
            raise ValueError("A GNN szerepkörprior formátuma érvénytelen.")
        matrices = []
        for name in ("counter_weights", "synergy_weights"):
            try:
                matrix = torch.as_tensor(prior.get(name), dtype=self.role_prior_logits.dtype)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("A GNN szerepkörprior mátrixa érvénytelen.") from error
            if matrix.shape != (5, 5) or not torch.isfinite(matrix).all():
                raise ValueError("A GNN szerepkörprior csak véges, 5×5-ös mátrix lehet.")
            matrix = matrix.clone()
            if name == "synergy_weights":
                matrix.fill_diagonal_(1.0)  # No same-role ally edges; self loops stay neutral.
            if (matrix <= 0).any():
                raise ValueError("A GNN üzenetsúlyai csak pozitívak lehetnek.")
            matrices.append(matrix.clamp(.25, 4.0))
        eligible = prior.get("eligible")
        if not isinstance(eligible, dict):
            raise ValueError("A GNN priorhoz szükséges a nem off-meta hősök szerepkörlistája.")
        mask = torch.zeros_like(self.role_prior_eligible)
        for champion, roles in eligible.items():
            if type(champion) not in (int, str):
                raise ValueError("Érvénytelen hősazonosító a GNN szerepkörpriorban.")
            try:
                cid = int(champion)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Érvénytelen hősazonosító a GNN szerepkörpriorban.") from error
            if not 0 <= cid < mask.shape[0] or not isinstance(roles, (list, tuple)):
                raise ValueError("Érvénytelen hős vagy szerepkörlista a GNN priorban.")
            for role in roles:
                if type(role) is not int or not 0 <= role < 5:
                    raise ValueError("Érvénytelen szerepkör a GNN priorban.")
                mask[cid, role] = True
        with torch.no_grad():
            self.role_prior_logits.copy_(torch.stack(matrices).log().to(self.role_prior_logits.device))
            self.role_prior_eligible.copy_(mask)
            self.role_prior_enabled.fill_(bool(mask.any()))
        self._role_priors_active = bool(mask.any())
        self.role_prior_logits.requires_grad_(self._role_priors_active)

    @property
    def role_prior_fingerprint(self):
        document = {"version": 1, "logits": self.role_prior_logits.detach().cpu().tolist(),
                    "eligible": self.role_prior_eligible.detach().cpu().tolist(),
                    "enabled": self._role_priors_active}
        return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Only the new keys are optional for legacy checkpoints. Other missing
        # weights still raise the usual strict-loading error.
        defaults = {"role_prior_logits": torch.zeros_like(self.role_prior_logits),
                    "role_prior_eligible": torch.zeros_like(self.role_prior_eligible),
                    "role_prior_enabled": torch.tensor(False, device=self.role_prior_enabled.device)}
        if not any(prefix + name in state_dict for name in defaults):
            for name, default in defaults.items():
                state_dict[prefix + name] = default
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._role_priors_active = bool(self.role_prior_enabled.item())
        self.role_prior_logits.requires_grad_(self._role_priors_active)

    def _edge_weights(self, nodes, edge_idx):
        source, target = edge_idx
        roles = nodes[:, 1].remainder(5)
        same_team = nodes[source, 2] == nodes[target, 2]
        logits = self.role_prior_logits[same_team.long(), roles[target], roles[source]]
        weights = logits.clamp(math.log(.25), math.log(4.0)).exp()
        champions = nodes[:, 0]
        known = (champions >= 0) & (champions < self.role_prior_eligible.shape[0])
        eligible = self.role_prior_eligible[champions.clamp(0, self.role_prior_eligible.shape[0] - 1), roles] & known
        return torch.where(eligible[source] & eligible[target], weights, torch.ones_like(weights))

    def forward(self, x, edge_idx, batch):
        edge_weight = self._edge_weights(x, edge_idx) if self._role_priors_active else None
        c_ids, r_ids, t_ids = x[:, 0], x[:, 1], x[:, 2].float().unsqueeze(1)
        
        # Keep checkpoint indices intact: champion 0 is Aatrox, -1 is an unknown pick.
        champion_features = self.champ_emb(c_ids.clamp(min=0))
        champion_features = champion_features * (c_ids >= 0).unsqueeze(1)
        x = torch.cat([champion_features, self.role_emb(r_ids), t_ids], dim=1)
        
        x = F.relu(self.bn1(self.conv1(x, edge_idx, edge_weight=edge_weight)))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_idx, edge_weight=edge_weight))
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
