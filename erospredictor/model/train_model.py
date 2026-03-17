import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from configs import DIVISION_WEIGHTS, CHAMPION_COUNT, MODELS_DIR

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

def train_model(preprocessor, epochs_rw=50, batch_size_rw=32, lr_rw=0.001, epochs_ra=50, batch_size_ra=32, lr_ra=0.001):
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== Preprocessing roleweighted data ===")
    X_rw, y_rw, divisions_rw = preprocessor.preprocess_all_matches()
    print(f"Loaded {len(X_rw)} roleweighted samples")

    print("=== Preprocessing roleaware data ===")
    X_ra, y_ra, divisions_ra = preprocessor.preprocess_all_matches_roleaware()
    print(f"Loaded {len(X_ra)} roleaware samples")

    def train_for_variant(model_suffix, X_all, y_all, divisions, input_size, epochs, batch_size, lr, model_type="standard"):
        unique_divisions = sorted(set(divisions))
        print(f"\n=== Training models for variant: {model_suffix} ===")
        
        for div in unique_divisions:
            model_name = f"{div}_{model_suffix}"
            train_single_model(X_all, y_all, divisions, model_name, input_size, epochs, batch_size, lr, model_type)

        print(f"\n=== Training MIXED_{model_suffix} model ===")
        train_single_model(X_all, y_all, divisions, f"MIXED_{model_suffix}", input_size, epochs, batch_size, lr, model_type)

    train_for_variant("roleweighted", X_rw, y_rw, divisions_rw, CHAMPION_COUNT * 2, epochs_rw, batch_size_rw, lr_rw, "standard")
    train_for_variant("roleaware", X_ra, y_ra, divisions_ra, CHAMPION_COUNT * 10, epochs_ra, batch_size_ra, lr_ra, "roleaware")

    print("\n=== All models trained and saved successfully ===")