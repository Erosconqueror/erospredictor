import json
import os
from pathlib import Path
from configs import CHAMPION_DATA_PATH

class StatisticalModel:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.matchups = {}
        self.champ_mapping = self._load_champ_mapping()
        
    def _load_champ_mapping(self):
        """Betölti a hős indexeket, hogy szinkronban legyen a PyTorch-al."""
        path = Path(CHAMPION_DATA_PATH)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def build_from_matches(self):
        """Végigpörgeti az SQL-t, és kiszámolja a winrate-eket."""
        matches = self.data_manager.get_all_matches()
        if not matches:
            print("Nincs adat az adatbázisban a statisztikákhoz!")
            return

        self.matchups = {}
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        
        feldolgozott = 0
        
        for match in matches:
            division = match.get("tier", "UNKNOWN")
            blue_team_raw = match.get("blue_team", [])
            red_team_raw = match.get("red_team", [])
            blue_win = match.get("blue_win", False)
            
            # Golyóálló konverzió (ha véletlen stringként jönne az SQL-ből)
            if isinstance(blue_team_raw, str):
                blue_team_raw = blue_team_raw.strip("{}").split(",")
            if isinstance(red_team_raw, str):
                red_team_raw = red_team_raw.strip("{}").split(",")

            # Riot ID -> Belső Index
            blue_team = [str(self.champ_mapping.get(str(cid), "-1")) for cid in blue_team_raw]
            red_team = [str(self.champ_mapping.get(str(cid), "-1")) for cid in red_team_raw]
            
            # Ha hiányos a csapat, vagy ismeretlen a hős
            if len(blue_team) != 5 or len(red_team) != 5 or "-1" in blue_team or "-1" in red_team:
                continue

            # Szótár struktúra felépítése
            if division not in self.matchups:
                self.matchups[division] = {r: {} for r in roles}

            for i, role in enumerate(roles):
                bc = blue_team[i]
                rc = red_team[i]

                if bc not in self.matchups[division][role]:
                    self.matchups[division][role][bc] = {}
                if rc not in self.matchups[division][role][bc]:
                    self.matchups[division][role][bc][rc] = {"wins": 0, "matches": 0}

                self.matchups[division][role][bc][rc]["matches"] += 1
                if blue_win:
                    self.matchups[division][role][bc][rc]["wins"] += 1
            
            feldolgozott += 1

        print(f"Statisztika megepitve! Feldolgozott meccsek: {feldolgozott} / {len(matches)}")

    def save_cache(self, filepath="data/stats_cache.json"):
        """Kimenti a kiszámolt statisztikákat egy JSON fájlba."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.matchups, f, indent=4)
        print(f"Statisztikák gyorsítótárazva ide: {filepath}")

    def load_cache(self, filepath="data/stats_cache.json"):
        """Betölti a statisztikákat a JSON-ből."""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.matchups = json.load(f)
            print(f"Statisztikák betöltve a gyorsítótárból ({filepath}).")
            return True
        return False

    def predict(self, division, blue_team, red_team):
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        probabilities = []
        
        for i, role in enumerate(roles):
            blue_champ = str(blue_team[i])
            red_champ = str(red_team[i])
            
            if division == "MIXED":
                champ_stats = {"wins": 0, "matches": 0}
                for div_stats in self.matchups.values():
                    div_role_stats = div_stats.get(role, {})
                    stats = div_role_stats.get(blue_champ, {}).get(red_champ)
                    if stats:
                        champ_stats["wins"] += stats["wins"]
                        champ_stats["matches"] += stats["matches"]
            else:
                role_stats = self.matchups.get(division, {}).get(role, {})
                champ_stats = role_stats.get(blue_champ, {}).get(red_champ)
            
            if champ_stats and champ_stats["matches"] > 0:
                probabilities.append(champ_stats["wins"] / champ_stats["matches"])

        if not probabilities:
            return 0.50
        
        return sum(probabilities) / len(probabilities)