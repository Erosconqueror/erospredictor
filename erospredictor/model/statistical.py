import json
import os
from configs import CHAMPION_DATA_PATH

class StatisticalModel:
    """Predicts match outcomes purely based on historical champion matchup winrates."""
    
    def __init__(self, db):
        self.db = db
        self.matchups = {}
        with open(CHAMPION_DATA_PATH, 'r', encoding='utf-8') as f:
            self.c_map = json.load(f)

    def build_stats(self):
        """Builds lookup tables for champion matchup statistics based on historical data."""
        matches = self.db.get_all_matches()
        if not matches: return

        self.matchups = {}
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        
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

            for i, r in enumerate(roles):
                bc, rc = b_team[i], r_team[i]
                if bc not in self.matchups[div][r]: self.matchups[div][r][bc] = {}
                if rc not in self.matchups[div][r][bc]: self.matchups[div][r][bc][rc] = {"w": 0, "m": 0}

                self.matchups[div][r][bc][rc]["m"] += 1
                if m.get("blue_win", False): self.matchups[div][r][bc][rc]["w"] += 1

    def save_cache(self, path: str = "data/stats_cache.json"):
        """Saves generated matchups to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.matchups, f, indent=4)

    def load_cache(self, path: str = "data/stats_cache.json") -> bool:
        """Loads matchups from disk into memory."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.matchups = json.load(f)
            return True
        return False

    def predict(self, div: str, blue: list, red: list) -> float:
        """Predicts probability of blue team win using aggregated statistics."""
        probs = []
        for i, r in enumerate(["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]):
            bc, rc = str(blue[i]), str(red[i])
            
            if div == "MIXED":
                c_stats = {"w": 0, "m": 0}
                for d_stat in self.matchups.values():
                    s = d_stat.get(r, {}).get(bc, {}).get(rc)
                    if s:
                        c_stats["w"] += s["w"]
                        c_stats["m"] += s["m"]
            else:
                c_stats = self.matchups.get(div, {}).get(r, {}).get(bc, {}).get(rc)
            
            if c_stats and c_stats["m"] > 0:
                probs.append(c_stats["w"] / c_stats["m"])

        return sum(probs) / len(probs) if probs else 0.50