import numpy as np
from configs import CHAMPION_COUNT, ALLOWED_PATCHES, ROLE_WEIGHTS
from model.data_manager import DataManager

class Preprocessor:
    def __init__(self):
        self.data_manager = DataManager()
        self.cached_rw = None
        self.cached_ra = None
        self.patch_weights = self._calculate_patch_weights()

    def _calculate_patch_weights(self):
        weights = {}
        decay_factor = 0.2
        
        for i, patch in enumerate(reversed(ALLOWED_PATCHES)):
            weight = max(0.1, 1.0 - (i * decay_factor))
            weights[patch] = weight
            
        return weights

    def clear_cache(self):
        self.cached_rw = None
        self.cached_ra = None

    def preprocess_all_matches(self, use_cache=True, always_save_cache=True):
        if use_cache and self.cached_rw is not None:
            return self.cached_rw
            
        match_ids = self.data_manager.get_all_match_ids()
        X, y, divisions, weights = [], [], [], []
        
        vex = list(ROLE_WEIGHTS.values())
        
        for match_id in match_ids:
            match_data = self.data_manager.get_match(match_id)
            if not match_data:
                continue
                
            data = match_data["data"]
            division = data.get("tier", "UNKNOWN")
            patch = self.data_manager.extract_patch_version(data)
            weight = self.patch_weights.get(patch, 0.5) 
            
            blue_team = []
            red_team = []
            
            for i, participant in enumerate(data["participants"]):
                champ_index = self.data_manager.get_champindex_by_id(participant["championId"])
                if champ_index is None:
                    continue
                if i < 5:
                    blue_team.append(champ_index)
                else:
                    red_team.append(champ_index)
            
            if len(blue_team) != 5 or len(red_team) != 5:
                continue
                
            blue_win = data["teams"][0]["win"]
            
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
            
        match_ids = self.data_manager.get_all_match_ids()
        X, y, divisions, weights = [], [], [], []
        
        for match_id in match_ids:
            match_data = self.data_manager.get_match(match_id)
            if not match_data:
                continue
                
            data = match_data["data"]
            division = data.get("tier", "UNKNOWN")
            patch = self.data_manager.extract_patch_version(data)
            weight = self.patch_weights.get(patch, 0.5)
            
            blue_team = []
            red_team = []
            
            for i, participant in enumerate(data["participants"]):
                champ_index = self.data_manager.get_champindex_by_id(participant["championId"])
                if champ_index is None:
                    continue
                if i < 5:
                    blue_team.append(champ_index)
                else:
                    red_team.append(champ_index)
                    
            if len(blue_team) != 5 or len(red_team) != 5:
                continue
                
            blue_win = data["teams"][0]["win"]
            
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