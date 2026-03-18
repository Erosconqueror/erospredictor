import os
import torch
from model.statistical import StatisticalModel
from model.data_manager import DataManager
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS

class Controller:
    def __init__(self, view=None):
        self.data_manager = DataManager()
        self.stat_model = StatisticalModel(self.data_manager)
        self.stat_model.build_from_matches()
        self.view = view
        self.id_to_name = self.data_manager.get_champion_names()
        self.name_to_id = {name: cid for cid, name in self.id_to_name.items()}
        
        if self.view:
            self.view.predict_btn.clicked.connect(self.handle_prediction)

    def _convert_names_to_ids(self, team_names):
        """Átalakítja a felületről jövő hős neveket ID-kká. Ha nincs kiválasztva, 0-t ad."""
        team_ids = []
        for name in team_names:
            if name in self.name_to_id:
                team_ids.append(self.name_to_id[name])
            else:
                team_ids.append(0) 
        return team_ids

    def handle_prediction(self):
        """Ezt hívja meg a gombnyomás a felületen."""
        blue_names, red_names = self.view.get_selected_champions()
        
        if len(blue_names) != 5 or len(red_names) != 5:
            self.view.update_result("Kérlek válassz ki pontosan 5-5 hőst mindkét csapatba!")
            return

        self.view.update_result("Predikció folyamatban... Kérlek várj!")
        
        blue_team = self._convert_names_to_ids(blue_names)
        red_team = self._convert_names_to_ids(red_names)
        
        division = "MIXED" 
        result_text = self.predict_match_for_ui(division, blue_team, red_team)
        
        self.view.update_result(result_text)

    def predict_match_for_ui(self, division, blue_team, red_team):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_predictions = []
        all_weights = []
        
        model_weights = {
            "gnn": 0.6,
            "roleweighted": 0.1,
            "roleaware": 0.1,
            "statistical": 0.1
        }

        try:
            from model.gnn_predictor import LeagueGNN, predict_match_gnn
            gnn_model_path = f"models/{division}_gnn.pth"
            if not os.path.exists(gnn_model_path):
                gnn_model_path = "models/MIXED_gnn.pth" # Fallback
            
            if os.path.exists(gnn_model_path):
                model = LeagueGNN()
                model.load_state_dict(torch.load(gnn_model_path, map_location=device))
                model.to(device)
                model.eval()
                
                gnn_pred = predict_match_gnn(model, blue_team, red_team, device)
                all_predictions.append(gnn_pred)
                all_weights.append(model_weights["gnn"])
        except Exception as e:
            print(f"GNN prediction failed: {e}")

        try:
            model_path = os.path.join(MODELS_DIR, f"{division}_roleweighted.pth")
            if os.path.exists(model_path):
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
        except Exception as e:
            print(f"RoleWeighted prediction failed: {e}")

        try:
            model_path = os.path.join(MODELS_DIR, f"{division}_roleaware.pth")
            if os.path.exists(model_path):
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
        except Exception as e:
            print(f"RoleAware prediction failed: {e}")


        try:
            stat_pred = self.stat_model.predict(division, blue_team, red_team)
            if stat_pred is not None:
                all_predictions.append(stat_pred)
                all_weights.append(model_weights["statistical"])
        except Exception as e:
            print(f"Statistical prediction failed: {e}")

 
        if not all_predictions:
            return "Hiba: Egyik modell sem tudott predikciót végezni. (Hiányzó .pth fájlok?)"

        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        avg_pred = sum(pred * weight for pred, weight in zip(all_predictions, normalized_weights))

        blue_win_pct = avg_pred * 100
        red_win_pct = (1 - avg_pred) * 100

        result_str = (
            f"Kék Csapat Győzelmi Esély: {blue_win_pct:.2f}%\n"
            f"Piros Csapat Győzelmi Esély: {red_win_pct:.2f}%\n\n"
            f"(GNN: {all_predictions[0]*100:.1f}% | Többi modell: {[f'{p*100:.1f}%' for p in all_predictions[1:]]})"
        )
        return result_str