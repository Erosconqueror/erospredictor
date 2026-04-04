import json
from configs import CHAMPION_COUNT, ALLOWED_PATCHES, ROLE_WEIGHTS, CHAMPION_DATA_PATH
from model.data_manager import DataManager

class Preprocessor:
    """Prepares and transforms match data for ML model training."""
    
    def __init__(self):
        self.db = DataManager()
        self.c_rw = None
        self.c_ra = None
        self.weights = self._calc_weights()
        
        with open(CHAMPION_DATA_PATH, 'r', encoding='utf-8') as f:
            self.c_map = json.load(f)

    def _calc_weights(self) -> dict:
        w_map = {}
        for i, p in enumerate(reversed(ALLOWED_PATCHES)):
            w_map[p] = max(0.1, 1.0 - (i * 0.2))
        return w_map

    def clear_cache(self):
        """Clears memory variables for preprocessed data."""
        self.c_rw = None
        self.c_ra = None

    def process_matches(self, use_cache: bool = True) -> tuple:
        """Preprocesses matches using role-based weighting arrays."""
        if use_cache and self.c_rw: return self.c_rw
        
        matches = self.db.get_all_matches()
        if not matches: return [], [], [], []
            
        x_lst, y_lst, d_lst, w_lst = [], [], [], []
        v_weights = list(ROLE_WEIGHTS.values())
        
        for m in matches:
            b_raw, r_raw = m.get("blue_team", []), m.get("red_team", [])
            b_team = [int(self.c_map[str(c)]) for c in b_raw if str(c) in self.c_map]
            r_team = [int(self.c_map[str(c)]) for c in r_raw if str(c) in self.c_map]
            
            if len(b_team) != 5 or len(r_team) != 5: continue
                
            champs = [0.0] * (CHAMPION_COUNT * 2)
            for i, cid in enumerate(b_team): champs[cid] = v_weights[i]
            for i, cid in enumerate(r_team): champs[cid + CHAMPION_COUNT] = v_weights[i]
                
            x_lst.append(champs)
            y_lst.append(1.0 if m.get("blue_win") else 0.0)
            d_lst.append(m.get("tier", "UNKNOWN"))
            w_lst.append(self.weights.get(m.get("patch", "UNKNOWN"), 0.5))
            
        self.c_rw = (x_lst, y_lst, d_lst, w_lst)
        return self.c_rw

    def process_matches_ra(self, use_cache: bool = True) -> tuple:
        """Preprocesses matches as flattened positional one-hot arrays."""
        if use_cache and self.c_ra: return self.c_ra
            
        matches = self.db.get_all_matches()
        if not matches: return [], [], [], []
            
        x_lst, y_lst, d_lst, w_lst = [], [], [], []
        for m in matches:
            b_raw, r_raw = m.get("blue_team", []), m.get("red_team", [])
            b_team = [int(self.c_map[str(c)]) for c in b_raw if str(c) in self.c_map]
            r_team = [int(self.c_map[str(c)]) for c in r_raw if str(c) in self.c_map]
                    
            if len(b_team) != 5 or len(r_team) != 5: continue
                
            champs = [0.0] * (CHAMPION_COUNT * 10)
            for i, cid in enumerate(b_team): champs[i * CHAMPION_COUNT + cid] = 1.0
            for i, cid in enumerate(r_team): champs[(i + 5) * CHAMPION_COUNT + cid] = 1.0
                
            x_lst.append(champs)
            y_lst.append(1.0 if m.get("blue_win") else 0.0)
            d_lst.append(m.get("tier", "UNKNOWN"))
            w_lst.append(self.weights.get(m.get("patch", "UNKNOWN"), 0.5))
            
        self.c_ra = (x_lst, y_lst, d_lst, w_lst)
        return self.c_ra
    
    def gen_meta_champs(self):
        """Generates a JSON configuration of frequently played meta champions."""
        matches = self.db.get_all_matches()
        if not matches: return

        totals, stats = {}, {}
        for m in matches:
            tier = m.get("tier", "UNKNOWN")
            b_team = [int(self.c_map[str(c)]) for c in m.get("blue_team", []) if str(c) in self.c_map]
            r_team = [int(self.c_map[str(c)]) for c in m.get("red_team", []) if str(c) in self.c_map]

            if len(b_team) != 5 or len(r_team) != 5: continue

            for div in [tier, "MIXED"]:
                if div not in totals:
                    totals[div] = 0
                    stats[div] = {i: {} for i in range(5)} 
                
                totals[div] += 1
                for i, cid in enumerate(b_team): stats[div][i][cid] = stats[div][i].get(cid, 0) + 1
                for i, cid in enumerate(r_team): stats[div][i][cid] = stats[div][i].get(cid, 0) + 1

        meta = {}
        for div, t_match in totals.items():
            meta[div] = {}
            t_picks = t_match * 2 
            for r_idx in range(5):
                val_c = [c for c, count in stats[div][r_idx].items() if count >= 500 or (count / t_picks) >= 0.01]
                meta[div][str(r_idx)] = val_c

        with open("data/meta_champs.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)