import os
import json
import torch
import numpy as np
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor

class CoreModel:
    """Main interface for fast, deterministic predictions with meta-learned ensemble weights."""
    
    def __init__(self, stat_model):
        self.stat_model = stat_model
        self.models = {}
        self.active_div = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_data = {}
        self.ensemble_weights = self._default_weights()

    def _default_weights(self) -> dict:
        """Default ensemble weights."""
        return {
            "gnn": 0.58,
            "roleweighted": 0.11,
            "roleaware": 0.12,
            "statistical": 0.19
        }

    def load_meta_data(self, data):
        """Load champion meta information."""
        self.meta_data = data

    def load_ensemble_weights(self, div: str = "MIXED"):
        """Load learned ensemble weights from meta-calibrator."""
        try:
            from model.meta_calibrator import MetaLearningCalibrator
            calibrator = MetaLearningCalibrator()
            path = f"models/{div}_meta_calibrator.pkl"
            
            if os.path.exists(path):
                calibrator.load(path)
                self.ensemble_weights = calibrator.get_weights()
                return True
        except Exception:
            pass
        
        self.ensemble_weights = self._default_weights()
        return False

    def load_models(self, div: str):
        """Load appropriate models for given division."""
        if self.active_div == div and self.models:
            return 
            
        self.models = {}
        self.active_div = div
        self.load_ensemble_weights(div)

        try:
            from model.gnn_predictor import LeagueGNN
            gnn_path = f"models/{div}_gnn.pth"
            if not os.path.exists(gnn_path):
                gnn_path = "models/gnn_model.pth"
            
            if os.path.exists(gnn_path):
                gnn = LeagueGNN()
                gnn.load_state_dict(torch.load(gnn_path, map_location=self.device, weights_only=True))
                gnn.to(self.device)
                gnn.eval() 
                self.models["gnn"] = gnn
        except Exception: pass

        try:
            rw_path = os.path.join(MODELS_DIR, f"{div}_roleweighted.pth")
            if os.path.exists(rw_path):
                rw = ChampionPredictor(input_size=CHAMPION_COUNT * 2).to(self.device)
                rw.load_state_dict(torch.load(rw_path, map_location=self.device, weights_only=True))
                rw.eval() 
                self.models["roleweighted"] = rw
        except Exception: pass

        try:
            ra_path = os.path.join(MODELS_DIR, f"{div}_roleaware.pth")
            if os.path.exists(ra_path):
                ra = RoleAwareEmbeddingPredictor().to(self.device)
                ra.load_state_dict(torch.load(ra_path, map_location=self.device, weights_only=True))
                ra.eval()
                self.models["roleaware"] = ra
        except Exception: pass

    def calc_win_prob_fast(self, div: str, blue: list, red: list) -> float:
        """Calculate deterministic win probability via meta-learned weights (Lightning Fast)."""
        self.load_models(div)
        
        preds = []
        weights = []

        if "gnn" in self.models:
            from model.gnn_predictor import predict_gnn
            try:
                p = predict_gnn(self.models["gnn"], blue, red, self.device)
                preds.append(p)
                weights.append(self.ensemble_weights["gnn"])
            except Exception: pass

        if "roleweighted" in self.models:
            c_rw = [0.0] * (CHAMPION_COUNT * 2)
            vw = list(ROLE_WEIGHTS.values())
            for i, c_id in enumerate(blue):
                if c_id > 0: c_rw[c_id] = vw[i]
            for i, c_id in enumerate(red):
                if c_id > 0: c_rw[c_id + CHAMPION_COUNT] = vw[i]
            
            x_rw = torch.tensor([c_rw], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.models["roleweighted"](x_rw)
            preds.append(out.item())
            weights.append(self.ensemble_weights["roleweighted"])

        if "roleaware" in self.models:
            c_ra = [0.0] * (CHAMPION_COUNT * 10)
            for i, c_id in enumerate(blue):
                if c_id > 0: c_ra[i * CHAMPION_COUNT + c_id] = 1.0
            for i, c_id in enumerate(red):
                if c_id > 0: c_ra[(i + 5) * CHAMPION_COUNT + c_id] = 1.0
            
            x_ra = torch.tensor([c_ra], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.models["roleaware"](x_ra)
            preds.append(out.item())
            weights.append(self.ensemble_weights["roleaware"])

        try:
            p_stat = self.stat_model.predict(div, blue, red)
            if p_stat is not None:
                preds.append(p_stat)
                weights.append(self.ensemble_weights["statistical"])
        except Exception: pass

        if not preds:
            return 0.50
        tot_w = sum(weights)
        norm_w = [w / tot_w for w in weights]
        
        blue_prob = sum(p * w for p, w in zip(preds, norm_w))
        return blue_prob

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        """Predict match outcome deterministically."""
        blue_prob_orig = self.calc_win_prob_fast(div, blue, red)
        blue_prob_swap = self.calc_win_prob_fast(div, red, blue)
        
        red_prob_from_swap = 1.0 - blue_prob_swap
        
        blue_final = (blue_prob_orig * 0.58) + (red_prob_from_swap * 0.42)
        
        return {
            "blue_win_prob": blue_final * 100,
            "red_win_prob": (1.0 - blue_final) * 100,
            "uncertainty": 0.0,    
            "confidence": 100.0   
        }

    def recommend_champs(self, div: str, blue: list, red: list, bans: list, 
                        team: str, r_idx: int, top_k: int = 5, 
                        filter_off_meta: bool = True) -> list:
        """Recommend champions fast and deterministically."""
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

            pred = self.predict_match(div, t_blue, t_red)
            score = pred["blue_win_prob"] if team == "blue" else pred["red_win_prob"]
            
            res.append((c_id, score, 0.0, 100.0)) 

        res.sort(key=lambda x: (x[1] if x[1] != 50.0 else -1.0), reverse=True)
        return [{"id": cid, "wr": prob, "uncertainty": unc, "confidence": conf} 
                for cid, prob, unc, conf in res[:top_k]]