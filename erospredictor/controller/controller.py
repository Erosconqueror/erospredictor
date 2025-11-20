import os
import time
from xml.parsers.expat import model
import torch
import torch.nn as nn
from model.statistical import StatisticalModel
from model.riot import Riot
from model.data_manager import DataManager
from model.preprocessor import Preprocessor
from model.train_model import train_model, train_single_model
from model.predictor import ChampionPredictor , RoleAwareEmbeddingPredictor
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS

class Controller:
    def __init__(self):
        self.riot = Riot()
        self.data_manager = DataManager()
        self.preprocessor = Preprocessor()
        self.stat_model = StatisticalModel(self.data_manager)

    def fetch_and_store_match(self, match_id: str, region: str):
        match_data = self.riot.get_match_data(match_id)
        if match_data:
            self.data_manager.save_match(match_id, region, match_data)
            self.preprocessor.clear_cache()
            return True
        return False

    def fetch_from_division(self, queue_type="RANKED_SOLO_5x5", tier="DIAMOND", division="I", players_limit=10, match_count=5):
        players = self.riot.get_ranked_players(queue_type, tier, division)
        for p in players[:players_limit]:
            puuid = p.get("puuid")
            if not puuid:
                continue
            self.fetch_matches_from_puuid(puuid, match_count, tier)

    def fetch_matches_from_puuid(self, puuid, match_limit, tier=None):
        matches = self.riot.get_matches_by_id(puuid, match_limit, tier)
        for match in matches:
            match_id = match.get("matchId")
            if not match_id:
                continue

            if self.data_manager.get_match(match_id):
                print(f"Already have match {match_id}")
                continue

            self.data_manager.save_match(match_id, self.riot.region, match)
            print(f"Saved match {match_id}")
            time.sleep(0.2)
        
        #self.preprocessor.clear_cache() doont clear cache here to allow batch fetching

    def fetch_and_store_matches(self, game_name: str, tag_line: str, match_count=5):
        account = self.riot.get_account_by_riot_id(game_name, tag_line)
        if not account:
            print("Summoner not found.")
            return

        puuid = account.get("puuid")
        if not puuid:
            print("PUUID not found.")
            return

        matches = self.riot.get_matches_by_id(puuid, match_count)
        if not matches:
            print("No matches found.")
            return

        for match in matches:
            match_id = match["matchId"]
            self.data_manager.save_match(match_id, self.riot.region, match)
            print(f"Saved match {match_id}")
        

        #self.preprocessor.clear_cache()  # don't clear cache here to allow batch fetching

    def preprocess_all_training_data(self, use_cache=True):
        print("=== Preprocessing All Training Data ===")
        print(f"Using cache for loading: {use_cache}")
        print("Note: Processed data will always be saved to cache for future use")
        

        print("\n1. Preprocessing RoleWeighted data...")
        X_rw, y_rw, divisions_rw = self.preprocessor.preprocess_all_matches(
            use_cache=use_cache, 
            always_save_cache=True 
        )
        print(f"   Result: {len(X_rw)} samples")
        
        print("\n2. Preprocessing RoleAware data...")
        X_ra, y_ra, divisions_ra = self.preprocessor.preprocess_all_matches_roleaware(
            use_cache=use_cache,
            always_save_cache=True 
        )
        print(f"   Result: {len(X_ra)} samples")
        

        print("\n3. Building statistical model...")
        self.stat_model.build_from_matches()
        print("   Statistical model built successfully")
        
        cache_status = "loaded from cache" if use_cache and len(X_rw) > 0 else "processed from scratch"
        print(f"\n✅ All training data preprocessing completed ({cache_status})!")
        print(f"   - RoleWeighted: {len(X_rw)} samples")
        print(f"   - RoleAware: {len(X_ra)} samples")

    def train_specific_model(self, model_type, division, epochs=None, batch_size=None, lr=None, use_cache=True):
        """Train a specific model type for a specific division with cached data"""
        print(f"=== Training {model_type} for division {division} ===")
        
        self.preprocess_all_training_data(use_cache=use_cache)
        
        if epochs is None:
            epochs = 100 if model_type == "4" else 50 
        if batch_size is None:
            batch_size = 64 if model_type == "4" else 32
        if lr is None:
            lr = 0.0003 if model_type == "4" else 0.001
        
        if model_type == "1":  
            X, y, divisions = self.preprocessor.preprocess_all_matches(use_cache=use_cache)
            model_name = f"{division}_roleweighted"
            input_size = CHAMPION_COUNT * 2
            train_single_model(X, y, divisions, model_name, input_size, epochs, batch_size, lr, "standard")
        
        elif model_type == "2": 
            X, y, divisions = self.preprocessor.preprocess_all_matches_roleaware(use_cache=use_cache)
            model_name = f"{division}_roleaware"
            input_size = CHAMPION_COUNT * 10
            train_single_model(X, y, divisions, model_name, input_size, epochs, batch_size, lr, "roleaware")
        
        elif model_type == "3": 
            print("Statistical model is built during preprocessing. No training needed.")
            return
        
        elif model_type == "4":  
            self.train_gnn_model_specific(division, epochs, batch_size, lr)
        
        else:
            print("Invalid model type")
            return

    def train_gnn_model_specific(self, division, epochs=100, batch_size=64, lr=0.0003):
        """Train GNN model for specific division"""
        print("=== Training GNN Model ===")
    
        try:
            from model.gnn_predictor import LeagueGNN, preprocess_matches_for_gnn, create_match_graph
            import torch
            import torch.optim as optim
            from torch_geometric.loader import DataLoader
        
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
        
            print("Preprocessing match data into graphs...")
            graphs, divisions = preprocess_matches_for_gnn(self.data_manager)
        
            if len(graphs) == 0:
                print("No valid match data found for GNN training!")
                return
            
            if division != "MIXED":
                filtered_graphs = []
                for graph, graph_division in zip(graphs, divisions):
                    if graph_division == division:
                        filtered_graphs.append(graph)
                graphs = filtered_graphs
                print(f"Filtered to {len(graphs)} graphs for division {division}")
            
            if len(graphs) == 0:
                print(f"No graphs found for division {division}!")
                return
        
            if graphs:
                print(f"First graph - x dtype: {graphs[0].x.dtype}, shape: {graphs[0].x.shape}")
                print(f"First graph - edge_index dtype: {graphs[0].edge_index.dtype}, shape: {graphs[0].edge_index.shape}")
                if hasattr(graphs[0], 'y'):
                    print(f"First graph - y dtype: {graphs[0].y.dtype}, shape: {graphs[0].y.shape}")
        
            dataset = graphs
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
            model = LeagueGNN().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.BCELoss()
        
            print(f"Training on {len(graphs)} graphs for division {division}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                batch_count = 0
            
                for batch in dataloader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                
                    if epoch == 0 and batch_count == 0:
                        print(f"Batch - x dtype: {batch.x.dtype}, shape: {batch.x.shape}")
                        print(f"Batch - edge_index dtype: {batch.edge_index.dtype}, shape: {batch.edge_index.shape}")
                        print(f"Batch - y dtype: {batch.y.dtype}, shape: {batch.y.shape}")
                        print(f"Batch - batch dtype: {batch.batch.dtype}, shape: {batch.batch.shape}")
                
                    out = model(batch.x, batch.edge_index, batch.batch)
                
                    if epoch == 0 and batch_count == 0:
                        print(f"Model output dtype: {out.dtype}, shape: {out.shape}")
                        print(f"Model output range: {out.min().item():.4f} to {out.max().item():.4f}")
                
                    loss = criterion(out, batch.y)
                
                    loss.backward()
                    optimizer.step()
                
                    total_loss += loss.item()
                    batch_count += 1
            
                avg_loss = total_loss / len(dataloader)
                if epoch % 10 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
            import os
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{division}_gnn.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ GNN model for {division} saved to {model_path}")
        
        except Exception as e:
            print(f"❌ GNN training failed: {e}")
            import traceback
            traceback.print_exc()

    def clear_preprocessed_cache(self):
        """Clear all cached preprocessed data"""
        self.preprocessor.clear_cache()

    def train_model(self, epochs=50, batch_size=32, lr=0.001, epochs_b=50, batch_size_b=32, lr_b=0.001):
        X_data, y_label, divisions = self.preprocessor.preprocess_all_matches()
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_label, dtype=torch.float32).unsqueeze(1)
        train_model(self.preprocessor, epochs, batch_size, lr, epochs_b, batch_size_b, lr_b)

    def predict_match(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("=== Predict Match Outcome ===")

        division = input("Enter division (e.g. IRON, GOLD, DIAMOND, or MIXED): ").strip().upper()


        all_predictions = []
        all_weights = []
        

        model_weights = {
            "gnn": 0.6,
            "roleweighted": 0.1,
            "roleaware": 0.1,
            "statistical": 0.1
        }

        print("Enter champion IDs for blue team (5 champs):")
        blue_team = [int(input(f"Blue champ {i+1}: ")) for i in range(5)]

        print("\nEnter champion IDs for red team (5 champs):")
        red_team = [int(input(f"Red champ {i+1}: ")) for i in range(5)]


        try:
            from model.gnn_predictor import LeagueGNN, predict_match_gnn
            gnn_model_path = f"models/{division}_gnn.pth"
            if not os.path.exists(gnn_model_path):
                gnn_model_path = "models/gnn_model.pth"
                print(f"No division-specific GNN found for {division}, using general GNN model")
            
            if os.path.exists(gnn_model_path):
                model = LeagueGNN()
                model.load_state_dict(torch.load(gnn_model_path, map_location=device))
                model.to(device)
                model.eval()
                
                gnn_pred = predict_match_gnn(model, blue_team, red_team, device)
                all_predictions.append(gnn_pred)
                all_weights.append(model_weights["gnn"])
                print(f"GNN prediction: {gnn_pred:.4f}")
            else:
                print("No GNN model found")
        except Exception as e:
            print(f"GNN prediction failed: {e}")


        try:
            model_path = os.path.join(MODELS_DIR, f"{division}_roleweighted.pth")
            if not os.path.exists(model_path):
                print(f"No roleweighted model found for {division}")
            else:
                model = ChampionPredictor(input_size=CHAMPION_COUNT * 2).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                champions_rw = [0.0] * (CHAMPION_COUNT * 2)
                vex = list(ROLE_WEIGHTS.values())
                for i, champ_id in enumerate(blue_team):
                    champions_rw[champ_id] = vex[i]  
                for i, champ_id in enumerate(red_team):
                    champions_rw[champ_id + CHAMPION_COUNT] = vex[i]  
                
                with torch.no_grad():
                    x_rw = torch.tensor([champions_rw], dtype=torch.float32).to(device)
                    pred = model(x_rw).item()
                    all_predictions.append(pred)
                    all_weights.append(model_weights["roleweighted"])
                    print(f"RoleWeighted prediction: {pred:.4f}")
        except Exception as e:
            print(f"RoleWeighted prediction failed: {e}")

  
        try:
            model_path = os.path.join(MODELS_DIR, f"{division}_roleaware.pth")
            if not os.path.exists(model_path):
                print(f"No roleaware model found for {division}")
            else:
                model = RoleAwareEmbeddingPredictor().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                champions_ra = [0.0] * (CHAMPION_COUNT * 10)
                for i, champ_id in enumerate(blue_team):
                    champions_ra[i * CHAMPION_COUNT + champ_id] = 1.0
                for i, champ_id in enumerate(red_team):
                    champions_ra[(i + 5) * CHAMPION_COUNT + champ_id] = 1.0
                
                with torch.no_grad():
                    x_ra = torch.tensor([champions_ra], dtype=torch.float32).to(device)
                    pred = model(x_ra).item()
                    all_predictions.append(pred)
                    all_weights.append(model_weights["roleaware"])
                    print(f"RoleAware prediction: {pred:.4f}")
        except Exception as e:
            print(f"RoleAware prediction failed: {e}")

        try:
            stat_pred = self.stat_model.predict(division, blue_team, red_team)
            if stat_pred is not None:
                all_predictions.append(stat_pred)
                all_weights.append(model_weights["statistical"])
                print(f"Statistical prediction: {stat_pred:.4f}")
        except Exception as e:
            print(f"Statistical prediction failed: {e}")

        if not all_predictions:
            print("No predictions were made.")
            return

        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        
        avg_pred = sum(pred * weight for pred, weight in zip(all_predictions, normalized_weights))

        print(f"\n=== Final Prediction for {division} ===")
        print(f"Blue win probability: {avg_pred * 100:.2f}%")
        print(f"Red win probability: {(1 - avg_pred) * 100:.2f}%")
        print(f"Individual predictions: {[f'{p:.4f}' for p in all_predictions]}")
        print(f"Model weights: {[f'{w:.2f}' for w in normalized_weights]}")


    def train_gnn_model(self, epochs=100, batch_size=64, lr=0.0003):
        """Legacy method - trains general GNN model"""
        self.train_gnn_model_specific("MIXED", epochs, batch_size, lr)