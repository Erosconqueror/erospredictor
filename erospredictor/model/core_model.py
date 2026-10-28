import math
import os
import pickle

try:
    import torch
except ImportError:
    torch = None

from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS


EMPTY_CHAMPION = -1


class CoreModel:
    """Match inference, recommendations, and explicitly requested diagnostics."""

    def __init__(self, stat_model):
        self.stat_model = stat_model
        self.models = {}
        self.active_div = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch else "cpu"
        self.meta_data = {}
        self.ensemble_weights = self._default_weights()
        self.load_errors = {}

    def _default_weights(self) -> dict:
        return {"gnn": 0.58, "roleweighted": 0.11, "roleaware": 0.12, "statistical": 0.19}

    def load_meta_data(self, data):
        self.meta_data = data

    def load_ensemble_weights(self, div: str = "MIXED") -> bool:
        """Load a calibrated ensemble, falling back to the packaged defaults."""
        try:
            path = os.path.join(MODELS_DIR, f"{div}_meta_calibrator.pkl")
            if os.path.exists(path):
                with open(path, "rb") as stream:
                    saved = pickle.load(stream)
                weights = saved.get("weights", saved) if isinstance(saved, dict) else saved
                if (isinstance(weights, dict) and weights
                        and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                                for v in weights.values()) and sum(weights.values()) > 0):
                    self.ensemble_weights = weights
                    return True
        except Exception as exc:
            self.load_errors["calibration"] = str(exc)
        self.ensemble_weights = self._default_weights()
        return False

    def load_models(self, div: str):
        """Load once per division; record unavailable components for diagnostics."""
        if self.active_div == div:
            return
        self.models = {}
        self.active_div = div
        self.load_errors = {}
        self.load_ensemble_weights(div)
        if torch is None:
            self.load_errors["neural_models"] = "A PyTorch nincs telepítve; csak a statisztikai modell érhető el."
            return

        for name in ("gnn", "roleweighted", "roleaware"):
            path = os.path.join(MODELS_DIR, f"{div}_{name}.pth")
            if name == "gnn" and not os.path.exists(path):
                path = os.path.join(MODELS_DIR, "gnn_model.pth")
            if not os.path.exists(path):
                self.load_errors[name] = "A modell súlyfájlja nem található."
                continue
            try:
                if name == "gnn":
                    from model.gnn_predictor import LeagueGNN
                    model = LeagueGNN()
                elif name == "roleweighted":
                    from model.predictor import ChampionPredictor
                    model = ChampionPredictor(input_size=CHAMPION_COUNT * 2)
                else:
                    from model.predictor import RoleAwareEmbeddingPredictor
                    model = RoleAwareEmbeddingPredictor()
                model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                model.to(self.device)
                model.eval()
                self.models[name] = model
            except Exception as exc:
                self.load_errors[name] = str(exc)

    @staticmethod
    def _validate_draft_shape(blue, red):
        if len(blue) != 5 or len(red) != 5:
            raise ValueError("Mindkét csapathoz pontosan öt pozíció szükséges.")
        if any(not isinstance(cid, int) or isinstance(cid, bool)
               or not EMPTY_CHAMPION <= cid < CHAMPION_COUNT for cid in list(blue) + list(red)):
            raise ValueError("Érvénytelen hősazonosító a draftban.")

    @classmethod
    def _validate_recommendation_request(cls, blue, red, team, r_idx):
        cls._validate_draft_shape(blue, red)
        if team not in ("blue", "red") or not isinstance(r_idx, int) or not 0 <= r_idx < 5:
            raise ValueError("Érvénytelen csapat vagy pozíció.")
        if (blue if team == "blue" else red)[r_idx] != EMPTY_CHAMPION:
            raise ValueError("Erre a pozícióra már választottál hőst!")

    def _component_predictions(self, div, blue, red):
        """Return successful model probabilities and their effective normalized weights."""
        self.load_models(div)
        predictions = {}
        for name, model in self.models.items():
            try:
                if name == "gnn":
                    from model.gnn_predictor import predict_gnn
                    probability = predict_gnn(model, blue, red, self.device)
                else:
                    if name == "roleweighted":
                        features = [0.0] * (CHAMPION_COUNT * 2)
                        role_weights = list(ROLE_WEIGHTS.values())
                        for side, champs in enumerate((blue, red)):
                            for role, cid in enumerate(champs):
                                if cid >= 0:
                                    features[side * CHAMPION_COUNT + cid] = role_weights[role]
                    else:
                        features = [0.0] * (CHAMPION_COUNT * 10)
                        for side, champs in enumerate((blue, red)):
                            for role, cid in enumerate(champs):
                                if cid >= 0:
                                    features[(side * 5 + role) * CHAMPION_COUNT + cid] = 1.0
                    inputs = torch.tensor([features], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        probability = model(inputs).item()
                if math.isfinite(probability) and 0 <= probability <= 1:
                    predictions[name] = probability
            except Exception as exc:
                self.load_errors[name] = str(exc)
        try:
            probability = self.stat_model.predict(div, blue, red)
            if probability is not None and math.isfinite(probability) and 0 <= probability <= 1:
                predictions["statistical"] = probability
        except Exception as exc:
            self.load_errors["statistical"] = str(exc)

        weights = {name: self.ensemble_weights.get(name, 0.0) for name in predictions}
        total = sum(weights.values())
        if predictions and total <= 0:
            weights = {name: 1.0 for name in predictions}
            total = len(predictions)
        return [{"model": name, "probability": probability, "weight": weights[name] / total}
                for name, probability in predictions.items() if weights[name] > 0]

    @staticmethod
    def _component_mean(components):
        return sum(item["probability"] * item["weight"] for item in components) if components else 0.5

    def calc_win_prob_fast(self, div: str, blue: list, red: list) -> float:
        return self._component_mean(self._component_predictions(div, blue, red))

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        """Evaluate both orientations with the same side balance as recommendations."""
        self._validate_draft_shape(blue, red)
        blue_prob = self.calc_win_prob_fast(div, blue, red)
        swapped_prob = self.calc_win_prob_fast(div, red, blue)
        final = blue_prob * 0.58 + (1.0 - swapped_prob) * 0.42
        return {"blue_win_prob": final * 100, "red_win_prob": (1.0 - final) * 100}

    def recommend_champs(self, div: str, blue: list, red: list, bans: list,
                        team: str, r_idx: int, top_k: int = 5,
                        filter_off_meta: bool = True) -> list:
        """Rank candidates only. Explanation inference is a separate optional call."""
        self._validate_recommendation_request(blue, red, team, r_idx)
        unavailable = {cid for cid in list(bans) + list(blue) + list(red) if cid >= 0}
        allowed = None
        if self.meta_data:
            lookup_div = div if div in self.meta_data else "MIXED"
            mode = "strict" if filter_off_meta else "loose"
            role_list = self.meta_data.get(lookup_div, {}).get(mode, {}).get(str(r_idx))
            if role_list is not None:
                allowed = set(role_list)
        results = []
        if top_k <= 0:
            return results
        for cid in range(CHAMPION_COUNT):
            if cid in unavailable or (allowed is not None and cid not in allowed):
                continue
            candidate_blue, candidate_red = list(blue), list(red)
            (candidate_blue if team == "blue" else candidate_red)[r_idx] = cid
            score = self.predict_match(div, candidate_blue, candidate_red)[f"{team}_win_prob"]
            results.append({"id": cid, "wr": score, "uncertainty": 0.0, "confidence": 100.0})
        results.sort(key=lambda item: item["wr"], reverse=True)
        return results[:top_k]

    def _prediction_components(self, div, blue, red, team):
        """Explain the actual side-balanced ensemble, including asymmetric failures."""
        direct = self._component_predictions(div, blue, red)
        swapped = self._component_predictions(div, red, blue)
        direct = direct or [{"model": "fallback", "probability": 0.5, "weight": 1.0}]
        swapped = swapped or [{"model": "fallback", "probability": 0.5, "weight": 1.0}]
        weights, contributions = {}, {}
        for orientation, balance in ((direct, 0.58), (swapped, 0.42)):
            for component in orientation:
                name = component["model"]
                weight = balance * component["weight"]
                probability = component["probability"] if orientation is direct else 1 - component["probability"]
                weights[name] = weights.get(name, 0.0) + weight
                contributions[name] = contributions.get(name, 0.0) + weight * probability
        details = []
        for name, weight in weights.items():
            probability = contributions[name] / weight
            if team == "red":
                probability = 1 - probability
            details.append({"model": name, "win_prob": probability * 100,
                            "weight": weight, "weighted_win_prob": weight * probability * 100})
        return details

    def explain_champion_effects(self, div: str, blue: list, red: list, team: str,
                                 r_idx: int, champion_id=None) -> dict:
        """Calculate only the visible draft's sensitivity to one selected champion.

        This explicit lightweight operation supports an already picked focus or a
        candidate for an empty role. Effects measure the change in the focus's
        gain after removing each other known champion, in percentage points from
        the selected team's perspective. They are not additive causal effects.
        """
        self._validate_draft_shape(blue, red)
        if (team not in ("blue", "red") or not isinstance(r_idx, int)
                or isinstance(r_idx, bool) or not 0 <= r_idx < 5):
            raise ValueError("Érvénytelen csapat vagy pozíció.")
        candidate_blue, candidate_red = list(blue), list(red)
        own = candidate_blue if team == "blue" else candidate_red
        if champion_id is None:
            champion_id = own[r_idx]
            if champion_id == EMPTY_CHAMPION:
                raise ValueError("Válassz hőst a kijelölt csapat pozíciójára, vagy válassz egy ajánlott hőst!")
        elif (not isinstance(champion_id, int) or isinstance(champion_id, bool)
              or not 0 <= champion_id < CHAMPION_COUNT):
            raise ValueError("Érvénytelen hősazonosító az elemzéshez.")
        elif own[r_idx] not in (EMPTY_CHAMPION, champion_id):
            raise ValueError("A kijelölt pozíció hőse megváltozott. Válaszd ki újra az elemzendő hőst!")
        own[r_idx] = champion_id
        known = [cid for cid in candidate_blue + candidate_red if cid >= 0]
        if len(known) != len(set(known)):
            raise ValueError("Egy hős csak egyszer lehet kiválasztva!")

        cache = {}

        def score(b, r):
            key = (tuple(b), tuple(r))
            if key not in cache:
                cache[key] = self.predict_match(div, b, r)[f"{team}_win_prob"]
            return cache[key]

        base_blue, base_red = list(candidate_blue), list(candidate_red)
        (base_blue if team == "blue" else base_red)[r_idx] = EMPTY_CHAMPION
        current, baseline = score(candidate_blue, candidate_red), score(base_blue, base_red)
        gain = current - baseline
        interactions = []
        for side, champions in (("blue", candidate_blue), ("red", candidate_red)):
            for role, champion in enumerate(champions):
                if champion < 0 or (side == team and role == r_idx):
                    continue
                without_blue, without_red = list(candidate_blue), list(candidate_red)
                empty_blue, empty_red = list(base_blue), list(base_red)
                (without_blue if side == "blue" else without_red)[role] = EMPTY_CHAMPION
                (empty_blue if side == "blue" else empty_red)[role] = EMPTY_CHAMPION
                without_score = score(without_blue, without_red)
                gain_without = without_score - score(empty_blue, empty_red)
                interactions.append({
                    "champion_id": champion, "team": side,
                    "side": "ally" if side == team else "opponent", "role_index": role,
                    "delta_pp": gain - gain_without, "score_delta_pp": current - without_score,
                    "is_lane_opponent": side != team and role == r_idx,
                })
        interactions.sort(key=lambda item: abs(item["delta_pp"]), reverse=True)
        return {
            "id": champion_id, "team": team, "role_index": r_idx,
            "wr": current, "baseline_win_prob": baseline, "candidate_delta_pp": gain,
            "interactions": interactions, "explanation_kind": "conditional_leave_one_out",
            "note": "A nyilak a kijelölt hős feltételes modellhatását jelzik százalékpontban. "
                    "Az egyes hatások nem adhatók össze, és nem bizonyítanak ok-okozatot.",
        }

    def explain_recommendations(self, div: str, blue: list, red: list, team: str,
                                r_idx: int, recommendations: list, max_items: int = 5,
                                selected_id=None) -> list:
        """Diagnose the selection and up to five ranked candidates, without repeats.

        Conditional effects compare the candidate's gain before/after omitting one
        known champion. They are sensitivity measurements, not causal or additive
        attribution, and their baseline contains a deliberately unknown position.
        """
        self._validate_recommendation_request(blue, red, team, r_idx)
        limit = max(0, min(int(max_items), 5))
        if not limit or (not recommendations and selected_id is None):
            return []
        ranked = recommendations[:limit]
        targets = ([{"id": selected_id}] if selected_id is not None else []) + list(ranked)
        for recommendation in targets:
            cid = recommendation["id"]
            if (not isinstance(cid, int) or isinstance(cid, bool) or not 0 <= cid < CHAMPION_COUNT
                    or cid in list(blue) + list(red)):
                raise ValueError("Az ajánlás már nem illik az aktuális drafthoz. Kérj új ajánlást!")
        ranks = {}
        for position, recommendation in enumerate(ranked, 1):
            ranks.setdefault(recommendation["id"], position)
        cache = {}

        def score(b, r):
            key = (tuple(b), tuple(r))
            if key not in cache:
                cache[key] = self.predict_match(div, b, r)[f"{team}_win_prob"]
            return cache[key]

        base_blue, base_red = list(blue), list(red)
        baseline = score(base_blue, base_red)
        result, seen = [], set()
        for recommendation in targets:
            cid = recommendation["id"]
            if cid in seen:
                continue
            seen.add(cid)
            candidate_blue, candidate_red = list(blue), list(red)
            (candidate_blue if team == "blue" else candidate_red)[r_idx] = cid
            components = self._prediction_components(div, candidate_blue, candidate_red, team)
            current = sum(item["weighted_win_prob"] for item in components)
            cache[(tuple(candidate_blue), tuple(candidate_red))] = current
            gain = current - baseline
            interactions = []
            for side, champions in (("blue", blue), ("red", red)):
                for role, champion in enumerate(champions):
                    if champion < 0:
                        continue
                    without_blue, without_red = list(candidate_blue), list(candidate_red)
                    empty_blue, empty_red = list(base_blue), list(base_red)
                    (without_blue if side == "blue" else without_red)[role] = EMPTY_CHAMPION
                    (empty_blue if side == "blue" else empty_red)[role] = EMPTY_CHAMPION
                    without_score = score(without_blue, without_red)
                    gain_without = without_score - score(empty_blue, empty_red)
                    interactions.append({
                        "champion_id": champion, "side": "ally" if side == team else "opponent",
                        "role_index": role, "delta_pp": gain - gain_without,
                        "score_delta_pp": current - without_score,
                        "is_lane_opponent": side != team and role == r_idx,
                    })
            interactions.sort(key=lambda item: abs(item["delta_pp"]), reverse=True)
            own = candidate_blue if team == "blue" else candidate_red
            enemy = candidate_red if team == "blue" else candidate_blue
            lane = self.stat_model.get_lane_matchup(div, cid, enemy[r_idx], r_idx) if enemy[r_idx] >= 0 else None
            synergy = self.stat_model.get_bot_synergy(div, own[3], own[4]) if r_idx in (3, 4) else None
            result.append({
                "id": cid, "wr": current, "baseline_win_prob": baseline,
                "is_selected": cid == selected_id, "recommendation_rank": ranks.get(cid),
                "candidate_delta_pp": gain, "components": components,
                "interactions": interactions, "lane_matchup": lane, "bot_synergy": synergy,
                "unavailable_models": dict(self.load_errors),
                "explanation_kind": "conditional_leave_one_out",
                "note": "Feltételes modellérzékenység; a hatások nem összeadhatók, és nem bizonyítanak ok-okozatot. "
                        "A hiányos draft becslése bizonytalanabb. A lane-adat külön, megfigyelt statisztika.",
            })
        return result
