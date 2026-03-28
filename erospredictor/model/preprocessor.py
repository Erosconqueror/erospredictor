import numpy as np
import json
from pathlib import Path
from configs import CHAMPION_COUNT, ALLOWED_PATCHES, ROLE_WEIGHTS, CHAMPION_DATA_PATH
from model.data_manager import DataManager

class Preprocessor:
    def __init__(self):
        self.data_manager = DataManager()
        self.cached_rw = None
        self.cached_ra = None
        self.patch_weights = self._calculate_patch_weights()
        self.champ_mapping = self._load_champ_mapping() # Beolvassuk a RAM-ba egyszer, hogy gyors legyen!

    def _calculate_patch_weights(self):
        weights = {}
        decay_factor = 0.2
        
        for i, patch in enumerate(reversed(ALLOWED_PATCHES)):
            weight = max(0.1, 1.0 - (i * decay_factor))
            weights[patch] = weight
            
        return weights

    def _load_champ_mapping(self):
        """Betölti a champion ID -> index (0-170) leképezést egyszer a memóriába."""
        path = Path(CHAMPION_DATA_PATH)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def clear_cache(self):
        self.cached_rw = None
        self.cached_ra = None

    def preprocess_all_matches(self, use_cache=True, always_save_cache=True):
        if use_cache and self.cached_rw is not None:
            return self.cached_rw
            
        # Egyetlen villámgyors lekérdezés az összes meccsre!
        matches = self.data_manager.get_all_matches()
        if not matches:
            print("Nincs meccs az adatbázisban!")
            return [], [], [], []
            
        X, y, divisions, weights = [], [], [], []
        vex = list(ROLE_WEIGHTS.values())
        
        for match_data in matches:
            division = match_data.get("tier", "UNKNOWN")
            patch = match_data.get("patch", "UNKNOWN")
            weight = self.patch_weights.get(patch, 0.5) 
            
            blue_team_raw = match_data.get("blue_team", [])
            red_team_raw = match_data.get("red_team", [])
            blue_win = match_data.get("blue_win", False)
            
            # Riot ID-k átváltása a mi belső indexeinkre (a memóriából)
            blue_team = [int(self.champ_mapping[str(cid)]) for cid in blue_team_raw if str(cid) in self.champ_mapping]
            red_team = [int(self.champ_mapping[str(cid)]) for cid in red_team_raw if str(cid) in self.champ_mapping]
            
            # Ha egy hős hiányzik az adatbázisból (pl. új karakter jött be), kihagyjuk a meccset
            if len(blue_team) != 5 or len(red_team) != 5:
                continue
                
            champions = [0.0] * (CHAMPION_COUNT * 2)
            for i, champ_id in enumerate(blue_team):
                champions[champ_id] = vex[i]
            for i, champ_id in enumerate(red_team):
                champions[champ_id + CHAMPION_COUNT] = vex[i]
                
            X.append(champions)
            y.append(1.0 if blue_win else 0.0)
            divisions.append(division)
            weights.append(weight)
            
        result = (X, y, divisions, weights)
        if always_save_cache:
            self.cached_rw = result
        return result

    def preprocess_all_matches_roleaware(self, use_cache=True, always_save_cache=True):
        if use_cache and self.cached_ra is not None:
            return self.cached_ra
            
        matches = self.data_manager.get_all_matches()
        if not matches:
            return [], [], [], []
            
        X, y, divisions, weights = [], [], [], []
        
        for match_data in matches:
            division = match_data.get("tier", "UNKNOWN")
            patch = match_data.get("patch", "UNKNOWN")
            weight = self.patch_weights.get(patch, 0.5)
            
            blue_team_raw = match_data.get("blue_team", [])
            red_team_raw = match_data.get("red_team", [])
            blue_win = match_data.get("blue_win", False)
            
            blue_team = [int(self.champ_mapping[str(cid)]) for cid in blue_team_raw if str(cid) in self.champ_mapping]
            red_team = [int(self.champ_mapping[str(cid)]) for cid in red_team_raw if str(cid) in self.champ_mapping]
                    
            if len(blue_team) != 5 or len(red_team) != 5:
                continue
                
            champions = [0.0] * (CHAMPION_COUNT * 10)
            for i, champ_id in enumerate(blue_team):
                champions[i * CHAMPION_COUNT + champ_id] = 1.0
            for i, champ_id in enumerate(red_team):
                champions[(i + 5) * CHAMPION_COUNT + champ_id] = 1.0
                
            X.append(champions)
            y.append(1.0 if blue_win else 0.0)
            divisions.append(division)
            weights.append(weight)
            
        result = (X, y, divisions, weights)
        if always_save_cache:
            self.cached_ra = result
        return result