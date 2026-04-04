import os
import time
import torch
from model.statistical import StatisticalModel
from model.riot import Riot
from model.data_manager import DataManager
from model.preprocessor import Preprocessor
from model.train_model import train_single_model
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS

class Controller:
    """Mediates between View and underlying ML/Data models."""

    def __init__(self, view=None):
        self.view = view
        self.riot = Riot()
        self.db = DataManager()
        self.prep = Preprocessor()
        self.stat_model = StatisticalModel(self.db)
        
        self.models = {}
        self.active_div = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self, div: str):
        """Loads PyTorch models for the specified division into memory."""
        if self.active_div == div and self.models:
            return 
            
        self.models = {}
        self.active_div = div

        try:
            from model.gnn_predictor import LeagueGNN
            gnn_path = f"models/{div}_gnn.pth"
            if not os.path.exists(gnn_path):
                gnn_path = "models/gnn_model.pth"
            
            if os.path.exists(gnn_path):
                gnn = LeagueGNN()
                gnn.load_state_dict(torch.load(gnn_path, map_location=self.device))
                gnn.to(self.device)
                gnn.eval()
                self.models["gnn"] = gnn
        except Exception: pass

        try:
            rw_path = os.path.join(MODELS_DIR, f"{div}_roleweighted.pth")
            if os.path.exists(rw_path):
                rw = ChampionPredictor(input_size=CHAMPION_COUNT * 2).to(self.device)
                rw.load_state_dict(torch.load(rw_path, map_location=self.device))
                rw.eval()
                self.models["roleweighted"] = rw
        except Exception: pass

        try:
            ra_path = os.path.join(MODELS_DIR, f"{div}_roleaware.pth")
            if os.path.exists(ra_path):
                ra = RoleAwareEmbeddingPredictor().to(self.device)
                ra.load_state_dict(torch.load(ra_path, map_location=self.device))
                ra.eval()
                self.models["roleaware"] = ra
        except Exception: pass

    def calc_win_prob(self, div: str, blue: list, red: list) -> float:
        """Calculates win probability based on ensemble model results."""
        self.load_models(div)
        
        preds = []
        weights = []
        w_map = {"gnn": 0.6, "roleweighted": 0.1, "roleaware": 0.1, "statistical": 0.1}

        if "gnn" in self.models:
            from model.gnn_predictor import predict_gnn
            try:
                p = predict_gnn(self.models["gnn"], blue, red, self.device)
                preds.append(p)
                weights.append(w_map["gnn"])
            except Exception: pass

        if "roleweighted" in self.models:
            c_rw = [0.0] * (CHAMPION_COUNT * 2)
            vw = list(ROLE_WEIGHTS.values())
            for i, c_id in enumerate(blue):
                if c_id > 0: c_rw[c_id] = vw[i]  
            for i, c_id in enumerate(red):
                if c_id > 0: c_rw[c_id + CHAMPION_COUNT] = vw[i]  
            
            with torch.no_grad():
                x_rw = torch.tensor([c_rw], dtype=torch.float32).to(self.device)
                p = self.models["roleweighted"](x_rw).item()
                preds.append(p)
                weights.append(w_map["roleweighted"])

        if "roleaware" in self.models:
            c_ra = [0.0] * (CHAMPION_COUNT * 10)
            for i, c_id in enumerate(blue):
                if c_id > 0: c_ra[i * CHAMPION_COUNT + c_id] = 1.0
            for i, c_id in enumerate(red):
                if c_id > 0: c_ra[(i + 5) * CHAMPION_COUNT + c_id] = 1.0
            
            with torch.no_grad():
                x_ra = torch.tensor([c_ra], dtype=torch.float32).to(self.device)
                p = self.models["roleaware"](x_ra).item()
                preds.append(p)
                weights.append(w_map["roleaware"])

        try:
            p_stat = self.stat_model.predict(div, blue, red)
            if p_stat is not None:
                preds.append(p_stat)
                weights.append(w_map["statistical"])
        except Exception: pass

        if not preds:
            return 0.5 

        tot_w = sum(weights)
        norm_w = [w / tot_w for w in weights]
        return sum(p * w for p, w in zip(preds, norm_w))

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        """Runs predictions from both sides and averages the result."""
        prob_orig = self.calc_win_prob(div, blue, red)
        prob_swap = self.calc_win_prob(div, red, blue)
        prob_if_red = 1.0 - prob_swap
        
        blue_final = (prob_orig * 0.6) + (prob_if_red * 0.4)
        
        return {
            "blue_win_prob": blue_final * 100,
            "red_win_prob": (1.0 - blue_final) * 100
        }

    def recommend_champs(self, div: str, blue: list, red: list, bans: list, team: str, r_idx: int, top_k: int = 5, allowed: set = None) -> list:
        """Simulates matches for available champions to find the highest win rate."""
        unavail = set(bans).union(set([c for c in blue + red if c > 0]))
        res = []
        
        for c_id in range(1, CHAMPION_COUNT):
            if c_id in unavail or (allowed is not None and c_id not in allowed):
                continue

            t_blue, t_red = list(blue), list(red)
            if team == "blue": t_blue[r_idx] = c_id
            else: t_red[r_idx] = c_id

            preds = self.predict_match(div, t_blue, t_red)
            score = preds["blue_win_prob"] if team == "blue" else preds["red_win_prob"]
            res.append((c_id, score))

        res.sort(key=lambda x: x[1], reverse=True)
        return [{"id": cid, "wr": prob} for cid, prob in res[:top_k]]

    def fetch_match(self, match_id: str, region: str) -> bool:
        """Fetches and stores a single match from Riot API."""
        data = self.riot.get_match_data(match_id)
        if data:
            self.db.save_match(match_id, region, data)
            self.prep.clear_cache()
            return True
        return False

    def fetch_div(self, q_type="RANKED_SOLO_5x5", tier="DIAMOND", div="I", p_limit=10, m_limit=5):
        """Fetches players from a specific division to gather their match data."""
        players = self.riot.get_ranked_players(q_type, tier, div)
        for p in players[:p_limit]:
            if puuid := p.get("puuid"):
                self.fetch_puuid(puuid, m_limit, tier)

    def fetch_puuid(self, puuid: str, limit: int, tier: str = None):
        """Fetches matches for a specific player PUUID."""
        for match in self.riot.get_matches_by_id(puuid, limit, tier):
            m_id = match.get("matchId")
            if m_id and not self.db.get_match(m_id):
                self.db.save_match(m_id, self.riot.region, match)
                time.sleep(0.2)
        
    def prep_training_data(self, cache=True):
        """Preprocesses all match data for training."""
        self.prep.preprocess_all_matches(use_cache=cache, always_save_cache=True)
        self.prep.preprocess_all_matches_roleaware(use_cache=cache, always_save_cache=True)
        self.stat_model.build_from_matches()

    def train_model(self, m_type: str, div: str, epochs=None, batch=None, lr=None, cache=True):
        """Trains a specified machine learning model type."""
        self.prep_training_data(cache=cache)
        epochs = epochs or (100 if m_type == "4" else 50)
        batch = batch or (64 if m_type == "4" else 32)
        lr = lr or (0.0003 if m_type == "4" else 0.001)
        
        if m_type == "1":  
            X, y, divs = self.prep.preprocess_all_matches(use_cache=cache)
            train_single_model(X, y, divs, f"{div}_roleweighted", CHAMPION_COUNT * 2, epochs, batch, lr, "standard")
        elif m_type == "2": 
            X, y, divs = self.prep.preprocess_all_matches_roleaware(use_cache=cache)
            train_single_model(X, y, divs, f"{div}_roleaware", CHAMPION_COUNT * 10, epochs, batch, lr, "roleaware")
        elif m_type == "4":  
            self.train_gnn_spec(div, epochs, batch, lr)

    def train_gnn_spec(self, div: str, epochs=100, batch=64, lr=0.0003):
        """Placeholder for GNN specific training routine."""
        pass 

    def train_gnn(self, epochs=100, batch=64, lr=0.0003):
        """Placeholder for general GNN training routine."""
        self.train_gnn_spec("MIXED", epochs, batch, lr)