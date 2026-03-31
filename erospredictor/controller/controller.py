import os
import time
from xml.parsers.expat import model
import torch
import torch.nn as nn
from model.statistical import StatisticalModel
from model.riot import Riot
from model.data_manager import DataManager
from model.preprocessor import Preprocessor
from model.train_model import train_single_model
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS

class Controller:
    def __init__(self, view=None):
        self.riot = Riot()
        self.data_manager = DataManager()
        self.preprocessor = Preprocessor()
        self.stat_model = StatisticalModel(self.data_manager)
        
        # --- ÚJ: Cache a betöltött modelleknek ---
        # Így a GUI használatakor nem kell másodpercenként újra betölteni a fájlokat
        self.loaded_models = {}
        self.current_division = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def preprocess_all_training_data(self, use_cache=True):
        print("=== Preprocessing All Training Data ===")
        print(f"Using cache for loading: {use_cache}")
        
        print("\n1. Preprocessing RoleWeighted data...")
        X_rw, y_rw, divisions_rw = self.preprocessor.preprocess_all_matches(
            use_cache=use_cache, always_save_cache=True 
        )
        print(f"   Result: {len(X_rw)} samples")
        
        print("\n2. Preprocessing RoleAware data...")
        X_ra, y_ra, divisions_ra = self.preprocessor.preprocess_all_matches_roleaware(
            use_cache=use_cache, always_save_cache=True 
        )
        print(f"   Result: {len(X_ra)} samples")
        
        print("\n3. Building statistical model...")
        self.stat_model.build_from_matches()
        print("   Statistical model built successfully")

    def train_specific_model(self, model_type, division, epochs=None, batch_size=None, lr=None, use_cache=True):
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
        elif model_type == "4":  
            self.train_gnn_model_specific(division, epochs, batch_size, lr)
        else:
            print("Invalid model type")

    def train_gnn_model_specific(self, division, epochs=100, batch_size=64, lr=0.0003):
        # A GNN betanítása marad a régiben (rövidítve a válaszban, de a tiédből ne töröld ki!)
        # ... Ide képzeld oda a te eredeti train_gnn_model_specific függvényed kódját ...
        pass # Kérlek hagyd itt a te eredeti kódodat!

    def clear_preprocessed_cache(self):
        self.preprocessor.clear_cache()

    def train_gnn_model(self, epochs=100, batch_size=64, lr=0.0003):
        self.train_gnn_model_specific("MIXED", epochs, batch_size, lr)


    # =========================================================================
    # ÚJ, GRAFIKUS FELÜLETHEZ (GUI) OPTIMALIZÁLT METÓDUSOK
    # =========================================================================

    def load_models_for_division(self, division):
        """Betölti a modelleket a memóriába. Ha már be vannak töltve, nem olvassa újra a lemezről."""
        if self.current_division == division and self.loaded_models:
            return # Már betöltve az adott divízió
            
        print(f"Modellek betöltése a(z) {division} divízióhoz a memóriába...")
        self.loaded_models = {}
        self.current_division = division

        # GNN betöltése
        try:
            from model.gnn_predictor import LeagueGNN
            gnn_model_path = f"models/{division}_gnn.pth"
            if not os.path.exists(gnn_model_path):
                gnn_model_path = "models/gnn_model.pth"
            
            if os.path.exists(gnn_model_path):
                gnn_model = LeagueGNN()
                gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=self.device))
                gnn_model.to(self.device)
                gnn_model.eval()
                self.loaded_models["gnn"] = gnn_model
        except Exception as e:
            print(f"GNN load failed: {e}")

        # RoleWeighted betöltése
        try:
            rw_path = os.path.join(MODELS_DIR, f"{division}_roleweighted.pth")
            if os.path.exists(rw_path):
                rw_model = ChampionPredictor(input_size=CHAMPION_COUNT * 2).to(self.device)
                rw_model.load_state_dict(torch.load(rw_path, map_location=self.device))
                rw_model.eval()
                self.loaded_models["roleweighted"] = rw_model
        except Exception as e:
            pass

        # RoleAware betöltése
        try:
            ra_path = os.path.join(MODELS_DIR, f"{division}_roleaware.pth")
            if os.path.exists(ra_path):
                ra_model = RoleAwareEmbeddingPredictor().to(self.device)
                ra_model.load_state_dict(torch.load(ra_path, map_location=self.device))
                ra_model.eval()
                self.loaded_models["roleaware"] = ra_model
        except Exception as e:
            pass

    def get_win_probability(self, division, blue_team, red_team):
        """Belső metódus, ami visszaadja a kék csapat győzelmi esélyét 0.0 és 1.0 között."""
        self.load_models_for_division(division)
        
        all_predictions = []
        all_weights = []
        model_weights = {"gnn": 0.6, "roleweighted": 0.1, "roleaware": 0.1, "statistical": 0.1}

        # 1. GNN
        if "gnn" in self.loaded_models:
            from model.gnn_predictor import predict_match_gnn
            try:
                gnn_pred = predict_match_gnn(self.loaded_models["gnn"], blue_team, red_team, self.device)
                all_predictions.append(gnn_pred)
                all_weights.append(model_weights["gnn"])
            except: pass

        # 2. RoleWeighted
        if "roleweighted" in self.loaded_models:
            champions_rw = [0.0] * (CHAMPION_COUNT * 2)
            vex = list(ROLE_WEIGHTS.values())
            for i, champ_id in enumerate(blue_team):
                if champ_id > 0:
                    champions_rw[champ_id] = vex[i]  
            for i, champ_id in enumerate(red_team):
                if champ_id > 0:
                    champions_rw[champ_id + CHAMPION_COUNT] = vex[i]  
            
            with torch.no_grad():
                x_rw = torch.tensor([champions_rw], dtype=torch.float32).to(self.device)
                pred = self.loaded_models["roleweighted"](x_rw).item()
                all_predictions.append(pred)
                all_weights.append(model_weights["roleweighted"])

        # 3. RoleAware
        if "roleaware" in self.loaded_models:
            champions_ra = [0.0] * (CHAMPION_COUNT * 10)
            for i, champ_id in enumerate(blue_team):
                if champ_id > 0:
                    champions_ra[i * CHAMPION_COUNT + champ_id] = 1.0
            for i, champ_id in enumerate(red_team):
                if champ_id > 0:
                    champions_ra[(i + 5) * CHAMPION_COUNT + champ_id] = 1.0
            
            with torch.no_grad():
                x_ra = torch.tensor([champions_ra], dtype=torch.float32).to(self.device)
                pred = self.loaded_models["roleaware"](x_ra).item()
                all_predictions.append(pred)
                all_weights.append(model_weights["roleaware"])

        # 4. Statistical
        try:
            stat_pred = self.stat_model.predict(division, blue_team, red_team)
            if stat_pred is not None:
                all_predictions.append(stat_pred)
                all_weights.append(model_weights["statistical"])
        except: pass

        if not all_predictions:
            return 0.5 

        total_weight = sum(all_weights)
        normalized_weights = [w / total_weight for w in all_weights]
        avg_pred = sum(p * w for p, w in zip(all_predictions, normalized_weights))
        
        return avg_pred

    def predict_match(self, division, blue_team, red_team):
        """
        GUI ÁLTAL HASZNÁLT: Kiszámolja egy adott mérkőzés esélyeit 60/40 súlyozással.
        blue_team, red_team: 5 elemű listák (ID-k), ahol az üres hely = 0
        """
        blue_prob_original = self.get_win_probability(division, blue_team, red_team)
        
        red_prob_swapped = self.get_win_probability(division, red_team, blue_team)
        
        blue_prob_if_red = 1.0 - red_prob_swapped
        
        final_blue_prob = (blue_prob_original * 0.6) + (blue_prob_if_red * 0.4)
        final_red_prob = 1.0 - final_blue_prob
        
        return {
            "blue_win_prob": final_blue_prob * 100,
            "red_win_prob": final_red_prob * 100
        }

    def recommend_champions(self, division, blue_team, red_team, bans, picking_team, picking_index, top_k=3, allowed_champs=None):
        """
        GUI ÁLTAL HASZNÁLT: Ajánló rendszer a kiválasztott pozícióhoz meta szűréssel.
        """
        unavailable = set(bans)
        unavailable.update([c for c in blue_team if c and c > 0])
        unavailable.update([c for c in red_team if c and c > 0])

        results = []
        
        for champ_id in range(1, CHAMPION_COUNT):
            # 1. Ha a hős már ki van választva, vagy tiltva van
            if champ_id in unavailable:
                continue
                
            # 2. ÚJ: Ha a checkbox be van kapcsolva, és a hős nincs a meta listában, átugorjuk!
            if allowed_champs is not None and champ_id not in allowed_champs:
                continue

            temp_blue = list(blue_team)
            temp_red = list(red_team)

            if picking_team == "blue":
                temp_blue[picking_index] = champ_id
            else:
                temp_red[picking_index] = champ_id

            # 3. Kiszámoljuk az esélyeket az új 60/40-es metódussal!
            preds = self.predict_match(division, temp_blue, temp_red)
            score = preds["blue_win_prob"] if picking_team == "blue" else preds["red_win_prob"]
            
            results.append((champ_id, score))

        # Rendezés a legnagyobb győzelmi esély alapján
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Mivel a predict_match már eleve %-ot (pl. 55.4) ad vissza, nem kell szorozni 100-zal
        return [{"champion_id": cid, "expected_winrate": prob} for cid, prob in results[:top_k]]