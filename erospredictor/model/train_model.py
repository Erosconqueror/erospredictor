import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from configs import DIVISION_WEIGHTS, CHAMPION_COUNT, MODELS_DIR

def train_single_model(X_all, y_all, divisions_all, model_name, input_size, epochs=50, batch_size=32, lr=0.001, model_type="standard"):
    """Train a single model for a specific division"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_division = model_name.split('_')[0]
    
    if target_division == "MIXED":

        weighted_X, weighted_y = [], []
        for x, y, div in zip(X_all, y_all, divisions_all):
            weight = DIVISION_WEIGHTS.get(div, 1.0)
            repeat = max(1, int(weight * 2))
            for _ in range(repeat):
                weighted_X.append(x)
                weighted_y.append(y)
        X = weighted_X
        y = weighted_y
    else:

        X = [x for x, d in zip(X_all, divisions_all) if d == target_division]
        y = [y for y, d in zip(y_all, divisions_all) if d == target_division]
    
    if not X:
        print(f"Skipping {model_name} — no data for division {target_division}.")
        return
    
    print(f"Training {model_name} with {len(X)} samples")
    

    def augment_data(X, y, data_type="standard"):
        X_augmented = []
        y_augmented = []
        
        for features, label in zip(X, y):
            X_augmented.append(features)
            y_augmented.append(label)
            
            if data_type == "standard" and len(features) == CHAMPION_COUNT * 2:
                blue_side = features[:CHAMPION_COUNT]
                red_side = features[CHAMPION_COUNT:]
                flipped = red_side + blue_side
                X_augmented.append(flipped)
                y_augmented.append(1 - label)
                
            elif data_type == "roleaware" and len(features) == CHAMPION_COUNT * 10:
                blue_side = features[:CHAMPION_COUNT * 5]
                red_side = features[CHAMPION_COUNT * 5:]
                flipped = red_side + blue_side
                X_augmented.append(flipped)
                y_augmented.append(1 - label)
        
        print(f"Data augmentation: {len(X)} -> {len(X_augmented)} samples")
        return X_augmented, y_augmented

    if model_type == "roleaware":
        X_aug, y_aug = augment_data(X, y, "roleaware")
    else:
        X_aug, y_aug = augment_data(X, y, "standard")


    X_tensor = torch.tensor(X_aug, dtype=torch.float32)
    y_tensor = torch.tensor(y_aug, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model_type == "roleaware":
        model = RoleAwareEmbeddingPredictor().to(device)
    else:
        model = ChampionPredictor(input_size=input_size).to(device)
        
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 10

    print(f"=== Training {model_name} ({model_type}) ===")
    print(f"Data: {len(X_aug)} samples, {sum(y_aug)/len(y_aug)*100:.1f}% positive")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
    
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
            total_loss += loss.item()
            predicted = (out > 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        scheduler.step()
    
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{model_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Saved best model -> {model_path} (loss: {best_loss:.4f})")

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