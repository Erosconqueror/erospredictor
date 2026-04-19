import os
import pickle
import torch
from typing import Dict, Any
from configs import CHAMPION_COUNT, ROLE_WEIGHTS

class MetaLearningCalibrator:
    """Evaluates individual models against the Golden Dataset and calculates optimal ensemble weights."""

    def __init__(self, dataset=None):
        self.dataset = dataset
        self.weights = {
            "gnn": 0.25,
            "roleweighted": 0.25,
            "roleaware": 0.25,
            "statistical": 0.25
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _wrap_model_prediction(self, model_name: str, model_obj: Any) -> callable:
        """Creates a standardized prediction function for a specific model."""
        def predict_fn(div: str, blue: list, red: list) -> dict:
            prob = 0.5
            if model_name == "gnn":
                from model.gnn_predictor import predict_gnn
                try:
                    prob = predict_gnn(model_obj, blue, red, self.device)
                except Exception:
                    pass
            elif model_name == "roleweighted":
                c_rw = [0.0] * (CHAMPION_COUNT * 2)
                vw = list(ROLE_WEIGHTS.values())
                for i, c_id in enumerate(blue):
                    if c_id > 0: c_rw[c_id] = vw[i]
                for i, c_id in enumerate(red):
                    if c_id > 0: c_rw[c_id + CHAMPION_COUNT] = vw[i]
                x_rw = torch.tensor([c_rw], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    prob = model_obj(x_rw).item()
            elif model_name == "roleaware":
                c_ra = [0.0] * (CHAMPION_COUNT * 10)
                for i, c_id in enumerate(blue):
                    if c_id > 0: c_ra[i * CHAMPION_COUNT + c_id] = 1.0
                for i, c_id in enumerate(red):
                    if c_id > 0: c_ra[(i + 5) * CHAMPION_COUNT + c_id] = 1.0
                x_ra = torch.tensor([c_ra], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    prob = model_obj(x_ra).item()
            elif model_name == "statistical":
                try:
                    res = model_obj.predict(div, blue, red)
                    if res is not None:
                        prob = res
                except Exception:
                    pass
            return {"blue_win_prob": prob * 100}
        return predict_fn

    def calibrate(self, models: Dict[str, Any], div: str) -> dict:
        """Runs predictions for all models and calculates normalized weights based on accuracy."""
        if not self.dataset:
            return self.weights

        scores = {}
        total_score = 0.0

        for name, model in models.items():
            if not model:
                continue
                
            pred_fn = self._wrap_model_prediction(name, model)
            res = self.dataset.validate_predictions(pred_fn, div)
            
            score = 0.1 + res.get('passed', 0)
            scores[name] = score
            total_score += score

        if total_score > 0:
            self.weights = {k: round(v / total_score, 3) for k, v in scores.items()}
            
        return self.weights

    def save(self, path: str):
        """Saves calculated weights to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load(self, path: str):
        """Loads weights from a file."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.weights = pickle.load(f)
        return self.weights
    
    def get_weights(self) -> dict:
        """Returns the current weights."""
        return self.weights