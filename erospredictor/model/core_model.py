import os
import json
import torch
import numpy as np
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor

class CoreModel:
    """Main interface for predictions with uncertainty and meta-learned ensemble weights."""
    
    def __init__(self, stat_model):
        self.stat_model = stat_model
        self.models = {}
        self.active_div = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_data = {}
        self.uncertainty_stats = {}
        self.ensemble_weights = self._default_weights()
        self.meta_calibrator = None

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

    def load_uncertainty_stats(self, path: str = "data/uncertainty_calibration.json"):
        """Load uncertainty calibration stats per rank."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.uncertainty_stats = json.load(f)

    def load_ensemble_weights(self, div: str = "MIXED"):
        """Load learned ensemble weights from meta-calibrator.
        
        Falls back to default weights if not available.
        """
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
                self.models["gnn"] = gnn
        except Exception: pass

        try:
            rw_path = os.path.join(MODELS_DIR, f"{div}_roleweighted.pth")
            if os.path.exists(rw_path):
                rw = ChampionPredictor(input_size=CHAMPION_COUNT * 2).to(self.device)
                rw.load_state_dict(torch.load(rw_path, map_location=self.device, weights_only=True))
                self.models["roleweighted"] = rw
        except Exception: pass

        try:
            ra_path = os.path.join(MODELS_DIR, f"{div}_roleaware.pth")
            if os.path.exists(ra_path):
                ra = RoleAwareEmbeddingPredictor().to(self.device)
                ra.load_state_dict(torch.load(ra_path, map_location=self.device, weights_only=True))
                self.models["roleaware"] = ra
        except Exception: pass

    def _mc_dropout_sample(self, model, x: torch.Tensor, samples: int = 20) -> np.ndarray:
        """Run model with dropout enabled for uncertainty estimation."""
        model.train()
        predictions = []
        
        with torch.no_grad():
            for _ in range(samples):
                out = model(x)
                predictions.append(out.cpu().numpy())
        
        model.eval()
        return np.concatenate(predictions, axis=0)

    def calc_win_prob_with_uncertainty(self, div: str, blue: list, red: list, 
                                       mc_samples: int = 20) -> dict:
        """Calculate win probability with uncertainty via MC Dropout and meta-learned weights.
        
        Returns dict with probability, uncertainty, and confidence.
        """
        self.load_models(div)
        
        preds = []
        weights = []
        uncertainties = []

        if "gnn" in self.models:
            from model.gnn_predictor import predict_gnn
            try:
                samples = []
                for _ in range(mc_samples):
                    p = predict_gnn(self.models["gnn"], blue, red, self.device)
                    samples.append(p)
                mean_p = np.mean(samples)
                std_p = np.std(samples)
                preds.append(mean_p)
                uncertainties.append(std_p)
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
            samples = self._mc_dropout_sample(self.models["roleweighted"], x_rw, mc_samples)
            mean_p = np.mean(samples)
            std_p = np.std(samples)
            preds.append(mean_p)
            uncertainties.append(std_p)
            weights.append(self.ensemble_weights["roleweighted"])

        if "roleaware" in self.models:
            c_ra = [0.0] * (CHAMPION_COUNT * 10)
            for i, c_id in enumerate(blue):
                if c_id > 0: c_ra[i * CHAMPION_COUNT + c_id] = 1.0
            for i, c_id in enumerate(red):
                if c_id > 0: c_ra[(i + 5) * CHAMPION_COUNT + c_id] = 1.0
            
            x_ra = torch.tensor([c_ra], dtype=torch.float32).to(self.device)
            samples = self._mc_dropout_sample(self.models["roleaware"], x_ra, mc_samples)
            mean_p = np.mean(samples)
            std_p = np.std(samples)
            preds.append(mean_p)
            uncertainties.append(std_p)
            weights.append(self.ensemble_weights["roleaware"])

        try:
            p_stat = self.stat_model.predict(div, blue, red)
            if p_stat is not None:
                preds.append(p_stat)
                uncertainties.append(0.02)
                weights.append(self.ensemble_weights["statistical"])
        except Exception: pass

        if not preds:
            return {"blue_prob": 0.5, "red_prob": 0.5, "uncertainty": 0.0, "confidence": 0.0}

        tot_w = sum(weights)
        norm_w = [w / tot_w for w in weights]
        
        blue_prob = sum(p * w for p, w in zip(preds, norm_w))
        ensemble_uncertainty = np.sqrt(sum((unc * w) ** 2 for unc, w in zip(uncertainties, norm_w)))
        confidence = max(0.0, 1.0 - ensemble_uncertainty)
        
        return {
            "blue_prob": blue_prob,
            "red_prob": 1.0 - blue_prob,
            "uncertainty": float(ensemble_uncertainty),
            "confidence": float(confidence)
        }

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        """Predict match outcome with uncertainty and meta-learned weights."""
        prob_orig = self.calc_win_prob_with_uncertainty(div, blue, red)
        prob_swap = self.calc_win_prob_with_uncertainty(div, red, blue)
        
        blue_prob = prob_orig["blue_prob"]
        red_prob_from_swap = 1.0 - prob_swap["blue_prob"]
        
        blue_final = (blue_prob * 0.58) + (red_prob_from_swap * 0.42)
        
        avg_uncertainty = (prob_orig["uncertainty"] + prob_swap["uncertainty"]) / 2
        confidence = max(0.0, 1.0 - avg_uncertainty)
        
        return {
            "blue_win_prob": blue_final * 100,
            "red_win_prob": (1.0 - blue_final) * 100,
            "uncertainty": avg_uncertainty * 100,
            "confidence": confidence
        }

    def recommend_champs(self, div: str, blue: list, red: list, bans: list, 
                        team: str, r_idx: int, top_k: int = 5, 
                        filter_off_meta: bool = True) -> list:
        """Recommend champions with uncertainty estimates."""
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
            uncertainty = pred["uncertainty"]
            confidence = pred["confidence"]
            
            res.append((c_id, score, uncertainty, confidence))

        res.sort(key=lambda x: (x[1] if x[1] != 50.0 else -1.0), reverse=True)
        return [{"id": cid, "wr": prob, "uncertainty": unc, "confidence": conf} 
                for cid, prob, unc, conf in res[:top_k]]
