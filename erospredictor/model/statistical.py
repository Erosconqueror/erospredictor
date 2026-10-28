from configs import DIVISION_WEIGHTS

class StatisticalModel:
    """Predicts match outcomes purely based on historical champion matchup winrates and bot lane synergies."""
    
    def __init__(self, db):
        self.db = db
        self.matchups = {}
        
        self.c_map = self.db.get_champion_mapping()

    def build_stats(self): 
        """Builds lookup tables for champion matchup statistics and bot lane synergies based on historical data."""
        self.matchups = {}
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
            return True
        return False

    def predict(self, div: str, blue: list, red: list) -> float:
        """Predicts the probability of the blue team winning based on champion matchups and bot lane synergies for the specified division."""
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
