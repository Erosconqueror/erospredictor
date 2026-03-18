import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from model.gnn_predictor import LeagueGNN
import os
from configs import MODELS_DIR

def train_single_model(X, y, divisions, model_name, input_size, epochs, batch_size, lr, model_type="standard", weights=None):
    if not X:
        print(f"No data available to train {model_name}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device} with {len(X)} samples...")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    if weights is not None:
        w_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    else:
        w_tensor = torch.ones_like(y_tensor)

    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model_type == "standard":
        model = ChampionPredictor(input_size=input_size).to(device)
    else:
        model = RoleAwareEmbeddingPredictor().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none')

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y, batch_w in dataloader:
            batch_X, batch_y, batch_w = batch_X.to(device), batch_y.to(device), batch_w.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            weighted_loss = (loss * batch_w).mean()
            
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}\n")

def train_gnn_model(graphs, model_name, epochs, batch_size, lr):
    if not graphs:
        print(f"No data available to train {model_name}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} (GNN) on {device} with {len(graphs)} graphs...")

    dataloader = GeoDataLoader(graphs, batch_size=batch_size, shuffle=True)
    model = LeagueGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.BCELoss(reduction='none')

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(outputs, batch.y)
            weighted_loss = (loss * batch.weight).mean()
            
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"GNN Model saved to {model_path}\n")