from configs import DIVISION_WEIGHTS
import math
from pathlib import Path

class StatisticalModel:
    """Predicts match outcomes purely based on historical champion matchup winrates and bot lane synergies."""
    
    def __init__(self, db):
        self.db = db
        self.matchups = {}
        self.interactions = None
        
        self.c_map = self.db.get_champion_mapping()

    def build_stats(self): 
        """Builds lookup tables for champion matchup statistics and bot lane synergies based on historical data."""
        self.matchups = {}
        self.interactions = None
        matches = self.db.get_all_matches()
        if not matches: return
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY", "BOT_SYNERGY"]
        
        for m in matches:
            div = m.get("tier", "UNKNOWN")
            b_raw, r_raw = m.get("blue_team", []), m.get("red_team", [])
            
            if isinstance(b_raw, str): b_raw = b_raw.strip("{}").split(",")
            if isinstance(r_raw, str): r_raw = r_raw.strip("{}").split(",")

            b_team = [str(self.c_map.get(str(c), "-1")) for c in b_raw]
            r_team = [str(self.c_map.get(str(c), "-1")) for c in r_raw]
            
            if len(b_team) != 5 or len(r_team) != 5 or "-1" in b_team or "-1" in r_team:
                continue
            
            if div not in self.matchups:
                self.matchups[div] = {r: {} for r in roles}

            for i, r in enumerate(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]):
                bc, rc = b_team[i], r_team[i]
                if bc not in self.matchups[div][r]: self.matchups[div][r][bc] = {}
                if rc not in self.matchups[div][r][bc]: self.matchups[div][r][bc][rc] = {"w": 0, "m": 0}

                self.matchups[div][r][bc][rc]["m"] += 1
                if m.get("blue_win", False): self.matchups[div][r][bc][rc]["w"] += 1

            b_adc, b_sup = b_team[3], b_team[4]
            if b_adc not in self.matchups[div]["BOT_SYNERGY"]: self.matchups[div]["BOT_SYNERGY"][b_adc] = {}
            if b_sup not in self.matchups[div]["BOT_SYNERGY"][b_adc]: self.matchups[div]["BOT_SYNERGY"][b_adc][b_sup] = {"w": 0, "m": 0}
            self.matchups[div]["BOT_SYNERGY"][b_adc][b_sup]["m"] += 1
            if m.get("blue_win", False): self.matchups[div]["BOT_SYNERGY"][b_adc][b_sup]["w"] += 1

            r_adc, r_sup = r_team[3], r_team[4]
            if r_adc not in self.matchups[div]["BOT_SYNERGY"]: self.matchups[div]["BOT_SYNERGY"][r_adc] = {}
            if r_sup not in self.matchups[div]["BOT_SYNERGY"][r_adc]: self.matchups[div]["BOT_SYNERGY"][r_adc][r_sup] = {"w": 0, "m": 0}
            self.matchups[div]["BOT_SYNERGY"][r_adc][r_sup]["m"] += 1
            if not m.get("blue_win", False): self.matchups[div]["BOT_SYNERGY"][r_adc][r_sup]["w"] += 1

    def save_cache(self, path: str = "data/stats_cache.json"):
        """Saves generated matchups and bot lane synergies to disk via DataManager."""
        self.db.save_stats_cache(self.matchups, path)

    def load_cache(self, path: str = "data/stats_cache.json") -> bool:
        """Loads matchups and bot lane synergies from disk into memory via DataManager."""
        cached_data = self.db.load_stats_cache(path)
        if cached_data is not None:
            self.matchups = cached_data
            from services.interaction_statistics import InteractionStatistics, load_interaction_statistics
            document = load_interaction_statistics(Path(path).with_name("interaction_stats.json"))
            self.interactions = InteractionStatistics(document, self.c_map)
            return True
        return False

    def predict(self, div: str, blue: list, red: list) -> float:
        """Predicts the probability of the blue team winning based on champion matchups and bot lane synergies for the specified division."""
        if self.uses_interactions(div, blue, red):
            return self._predict_interactions(div, blue, red)
        return self._predict_legacy(div, blue, red)

    def has_interactions(self, div):
        return self.interactions is not None and self.interactions.available(div)

    def has_strong_evidence(self, div, blue, red):
        """Every known main edge needs 100 games, other applicable edges need 20.

        Empty slots are not edges. Missing eligible pair evidence fails the gate;
        off-meta edges follow the actual legacy lane/bot fallback, not new edges.
        At least one concrete lane-opponent pairing is required.
        """
        from services.ensemble_weights import MAIN_EDGE_MIN_MATCHES, OTHER_EDGE_MIN_MATCHES
        enhanced = self.uses_interactions(div, blue, red)
        data = self.interactions
        main_edges = 0
        for role, champion in enumerate(blue):
            if champion < 0:
                continue
            for other_role, other in enumerate(red):
                if other < 0:
                    continue
                if enhanced and data.eligible(div, champion, role) and data.eligible(div, other, other_role):
                    row = data._edge("counter", div, champion, role, other, other_role)
                    count = row[0] if row is not None else 0
                elif role == other_role:
                    summary = self.get_lane_matchup(div, champion, other, role)
                    count = summary["matches"] if summary else 0
                else:
                    continue
                required = MAIN_EDGE_MIN_MATCHES if role == other_role else OTHER_EDGE_MIN_MATCHES
                if count < required:
                    return False
                main_edges += role == other_role
        if not main_edges:
            return False
        for team in (blue, red):
            for first in range(5):
                for second in range(first + 1, 5):
                    if team[first] < 0 or team[second] < 0:
                        continue
                    if enhanced and data.eligible(div, team[first], first) and data.eligible(div, team[second], second):
                        row = data._edge("synergy", div, team[first], first, team[second], second)
                        count = row[0] if row is not None else 0
                    elif first == 3 and second == 4:
                        summary = self.get_bot_synergy(div, team[first], team[second])
                        count = summary["matches"] if summary else 0
                    else:
                        continue
                    if count < OTHER_EDGE_MIN_MATCHES:
                        return False
        return True

    def uses_interactions(self, div, blue, red):
        if not self.has_interactions(div):
            return False
        data = self.interactions
        for role, champion in enumerate(blue):
            if champion < 0 or not data.eligible(div, champion, role):
                continue
            for other_role, other in enumerate(red):
                if other >= 0 and data.eligible(div, other, other_role) and data._edge("counter", div, champion, role, other, other_role):
                    return True
        for team in (blue, red):
            for first in range(5):
                for second in range(first + 1, 5):
                    if (team[first] >= 0 and team[second] >= 0
                            and data.eligible(div, team[first], first) and data.eligible(div, team[second], second)
                            and data._edge("synergy", div, team[first], first, team[second], second)):
                        return True
        return False

    def _predict_interactions(self, div, blue, red):
        """At most 25 counter + 20 unique same-team synergy cached lookups.

        New interactions exclude off-meta endpoints. Existing lane/bot evidence
        remains available for those endpoints, adjusted against the same champion
        baselines, while a completely unsupported draft keeps the old predictor.
        """
        data = self.interactions
        bucket = data._bucket(div)
        matrices = bucket["matrices"]
        def base(champion, role):
            summary = data.baseline(div, champion, role) if champion >= 0 else None
            return summary["winrate"] / 100 if summary else .5
        def logit(value):
            value = max(.001, min(.999, value))
            return math.log(value / (1 - value))
        def sigmoid(value):
            return 1 / (1 + math.exp(-max(-30, min(30, value))))
        blue_base, red_base = [base(cid, role) for role, cid in enumerate(blue)], [base(cid, role) for role, cid in enumerate(red)]
        blue_prior = bucket.get("blue_winrate", .5)
        side = logit(blue_prior)
        known_blue = [blue_base[role] for role, cid in enumerate(blue) if cid >= 0]
        known_red = [red_base[role] for role, cid in enumerate(red) if cid >= 0]
        strength = (sum(known_blue) / len(known_blue) - .5 if known_blue else 0)
        strength -= sum(known_red) / len(known_red) - .5 if known_red else 0
        counters, synergies = [], []
        for role, champion in enumerate(blue):
            if champion < 0:
                continue
            for other_role, other in enumerate(red):
                if other < 0:
                    continue
                row = data._edge("counter", div, champion, role, other, other_role)
                if row is not None:
                    counters.append((row[4] / 100, matrices["counter_weights"][role][other_role]))
                elif role == other_role and not (data.eligible(div, champion, role) and data.eligible(div, other, role)):
                    old = self.get_lane_matchup(div, champion, other, role)
                    if old:
                        expected = sigmoid(logit(blue_base[role]) - logit(red_base[role]) + side)
                        delta = (old["weighted_winrate"] / 100 - expected) * old["matches"] / (old["matches"] + 20)
                        counters.append((delta, 1.0))
        for team, sign, bases in ((blue, 1, blue_base), (red, -1, red_base)):
            for first in range(5):
                for second in range(first + 1, 5):
                    if team[first] < 0 or team[second] < 0:
                        continue
                    row = data._edge("synergy", div, team[first], first, team[second], second)
                    if row is not None:
                        weight = (matrices["synergy_weights"][first][second] + matrices["synergy_weights"][second][first]) / 2
                        synergies.append((sign * row[4] / 100, weight))
                    elif first == 3 and second == 4 and not (data.eligible(div, team[first], first) and data.eligible(div, team[second], second)):
                        old = self.get_bot_synergy(div, team[first], team[second])
                        if old:
                            expected = sigmoid(logit(bases[first]) + logit(bases[second]))
                            delta = (old["weighted_winrate"] / 100 - expected) * old["matches"] / (old["matches"] + 20)
                            synergies.append((sign * delta, 1.0))
        def mean(rows):
            return sum(value * weight for value, weight in rows) / sum(weight for _, weight in rows) if rows else 0
        return max(.01, min(.99, blue_prior + strength + mean(counters) + mean(synergies)))

    def _predict_legacy(self, div: str, blue: list, red: list) -> float:
        """Original six-edge fallback for old artifacts and unsupported drafts."""
        if div == "MIXED":
            total_weighted_prob = 0.0
            total_weight_sum = 0.0
            
            for tier_name in self._tiers(div):
                tier_weight = DIVISION_WEIGHTS[tier_name]
                prob, m_count = self._predict_tier(tier_name, blue, red)
                
                if m_count > 0:
                    combined_weight = tier_weight * m_count
                    total_weighted_prob += prob * combined_weight
                    total_weight_sum += combined_weight
                    
            if total_weight_sum > 0:
                return total_weighted_prob / total_weight_sum
            return 0.50
        else:
            prob, _ = self._predict_tier(div, blue, red)
            return prob

    def _tiers(self, div: str):
        """Respect a partial update's MIXED scope while retaining other tier caches."""
        if div != "MIXED":
            return (div,)
        scope = self.matchups.get("__scope__", {})
        selected = scope.get("mixed_tiers") if isinstance(scope, dict) else None
        if selected is None:
            return tuple(DIVISION_WEIGHTS)
        if not isinstance(selected, list):
            return ()
        return tuple(dict.fromkeys(tier for tier in selected if tier in DIVISION_WEIGHTS))

    def _observed_summary(self, div: str, records: list, kind: str):
        """Pool actual observations; expose both raw counts and tier weighting."""
        matches, wins, weighted_matches, weighted_wins = 0, 0, 0.0, 0.0
        tiers = set()
        for tier, record, reverse in records:
            if not record or record.get("m", 0) <= 0:
                continue
            count, won = record["m"], record["w"]
            if reverse:
                won = count - won
            weight = DIVISION_WEIGHTS.get(tier, 1.0) if div == "MIXED" else 1.0
            matches += count
            wins += won
            weighted_matches += count * weight
            weighted_wins += won * weight
            tiers.add(tier)
        if not matches:
            return None
        return {"kind": kind, "matches": matches, "wins": wins,
                "winrate": wins / matches * 100,
                "weighted_winrate": weighted_wins / weighted_matches * 100,
                "tiers": sorted(tiers)}

    def get_lane_matchup(self, div: str, champion_id: int, opponent_id: int, role_index: int):
        """Historical champion-vs-champion results in a role, pooling both sides.

        This describes whole-match outcomes for a lane pairing; it is not a
        measurement of who won the laning phase, nor a neural attribution.
        """
        if champion_id < 0 or opponent_id < 0 or not 0 <= role_index < 5:
            return None
        role = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")[role_index]
        records = []
        for tier in self._tiers(div):
            matchups = self.matchups.get(tier, {}).get(role, {})
            direct = matchups.get(str(champion_id), {}).get(str(opponent_id))
            reverse = matchups.get(str(opponent_id), {}).get(str(champion_id))
            records.extend(((tier, direct, False), (tier, reverse, True)))
        summary = self._observed_summary(div, records, "historical_lane_matchup")
        if summary:
            summary.update({"champion_id": champion_id, "opponent_id": opponent_id,
                            "role_index": role_index})
        return summary

    def get_bot_synergy(self, div: str, adc_id: int, support_id: int):
        """Observed whole-match results for a concrete ADC/support pair."""
        if adc_id < 0 or support_id < 0:
            return None
        records = [(tier, self.matchups.get(tier, {}).get("BOT_SYNERGY", {})
                    .get(str(adc_id), {}).get(str(support_id)), False) for tier in self._tiers(div)]
        summary = self._observed_summary(div, records, "historical_bot_synergy")
        if summary:
            summary.update({"adc_id": adc_id, "support_id": support_id})
        return summary

    def _predict_tier(self, div: str, blue: list, red: list):
        """Helper function to predict win probability for a specific division based on champion matchups and bot lane synergies."""
        probs = []
        total_w = 0.0
        total_m = 0.0
        
        for i, r in enumerate(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]):
            if blue[i] < 0 or red[i] < 0:
                continue
            bc, rc = str(blue[i]), str(red[i])
            c_stats = self.matchups.get(div, {}).get(r, {}).get(bc, {}).get(rc)
            
            if c_stats and c_stats["m"] > 0:
                probs.append(c_stats["w"] / c_stats["m"])
                total_w += c_stats["w"]
                total_m += c_stats["m"]

        b_adc, b_sup = str(blue[3]), str(blue[4])
        b_syn_stats = self.matchups.get(div, {}).get("BOT_SYNERGY", {}).get(b_adc, {}).get(b_sup)
        
        r_adc, r_sup = str(red[3]), str(red[4])
        r_syn_stats = self.matchups.get(div, {}).get("BOT_SYNERGY", {}).get(r_adc, {}).get(r_sup)
        
        syn_probs = []
        
        if b_syn_stats and b_syn_stats["m"] > 0:
            b_w, b_m = float(b_syn_stats["w"]), float(b_syn_stats["m"])
            syn_probs.append(b_w / b_m)
            total_w += b_w / 2.0
            total_m += b_m / 2.0
            
        if r_syn_stats and r_syn_stats["m"] > 0:
            r_w, r_m = float(r_syn_stats["w"]), float(r_syn_stats["m"])
            syn_probs.append(1.0 - (r_w / r_m))
            total_w += (r_m - r_w) / 2.0
            total_m += r_m / 2.0
            
        if syn_probs:
            probs.append(sum(syn_probs) / len(syn_probs))

        if not probs or total_m == 0:
            return 0.50, 0

        global_avg = total_w / total_m
        matchup_avg = sum(probs) / len(probs)

        return (global_avg + matchup_avg) / 2.0, total_m
