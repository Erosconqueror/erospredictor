import os
import torch
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor

class CoreModel:
    """This class serves as the main interface for making predictions and recommendations, vie using multiple underlying models"""
    def __init__(self, stat_model):
        self.stat_model = stat_model
        self.models = {}
        self.active_div = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_data = {}

    def load_meta_data(self, data):
        """Loads champion meta information from disk, which can be used to filter out non-meta champions in recommendations."""
        self.meta_data = data

    def load_models(self, div: str):
        """Loads the appropriate machine learning models for the given division, ensuring that only relevant models are kept in memory to optimize performance."""
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
        """Calculates the win probability for the blue team by aggregating predictions from multiple models, including GNN, role-weighted, role-aware, and statistical models.
            Each model's prediction is weighted according to a predefined scheme to produce a final probability score.
            
            GNN models are given nodes that represent champions, and the graphs edges are counter/synergy values with all other champions in the given lobby
            
            Rolawere models were given a  champion count long vector filled with 0s with the sole exception of the champions id,
            where it was a 1 instead X 10 times for each role at both teams - but was quite ineffective so the concept remained but now includes embedding
            
            Roleweighted models are given a champion count long vector filled with 0s with the sole exception
            of all the champions one team had where it was the role weight for that position X 2 for both teams
            """
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
        norm_w = [w / tot_w for w in weights] #this saved me quite hard, cuz weights were not summing to 1....
        return sum(p * w for p, w in zip(preds, norm_w))

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        """Predicts the win probabilities for both teams in a given match setup"""
        prob_orig = self.calc_win_prob(div, blue, red)
        prob_swap = self.calc_win_prob(div, red, blue)
        prob_if_red = 1.0 - prob_swap
        
        blue_final = (prob_orig * 0.6) + (prob_if_red * 0.4)
        
        return {
            "blue_win_prob": blue_final * 100,
            "red_win_prob": (1.0 - blue_final) * 100
        }

    def recommend_champs(self, div: str, blue: list, red: list, bans: list, team: str, r_idx: int, top_k: int = 5, filter_off_meta: bool = True) -> list:
        """Recommends the top K champions for a given role and team, based on predicted win probabilities while respecting bans and optionally filtering by meta relevance."""
        unavail = set(bans).union(set([c for c in blue + red if c > 0]))
        res = []
        
        allowed = set()
        if self.meta_data:
            lookup_div = div if div in self.meta_data else "MIXED"
            mode = "strict" if filter_off_meta else "loose"
            try:
                allowed = set(self.meta_data[lookup_div][mode][str(r_idx)])
            except KeyError:
                pass
        
        for c_id in range(1, CHAMPION_COUNT):
            if c_id in unavail:
                continue
            if allowed and c_id not in allowed:
                continue

            t_blue, t_red = list(blue), list(red)
            if team == "blue": t_blue[r_idx] = c_id
            else: t_red[r_idx] = c_id

            preds = self.predict_match(div, t_blue, t_red)
            score = preds["blue_win_prob"] if team == "blue" else preds["red_win_prob"]
            res.append((c_id, score))

        res.sort(key=lambda x: (x[1] if x[1] != 50.0 else -1.0), reverse=True)
        return [{"id": cid, "wr": prob} for cid, prob in res[:top_k]]