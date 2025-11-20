from collections import defaultdict


class StatisticalModel:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.matchups = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"wins":0,"matches":0}))))
    
    def build_from_matches(self):
        match_ids = self.data_manager.get_all_match_ids()
        for match_id in match_ids:
            match_data = self.data_manager.get_match(match_id)
            if not match_data:
                continue
            
            division = match_data.get("data", {}).get("tier", "UNKNOWN")
            participants = match_data["data"]["participants"]

            blue_team = participants[:5]
            red_team = participants[5:]
            
            blue_win = match_data["data"]["teams"][0]["win"]
            
            roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
            for i, role in enumerate(roles):
                blue_champ = blue_team[i]["championId"]
                red_champ = red_team[i]["championId"]

                self.matchups[division][role][blue_champ][red_champ]["matches"] += 1
                if blue_win:
                    self.matchups[division][role][blue_champ][red_champ]["wins"] += 1

    def predict(self, division, blue_team, red_team):
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        probabilities = []
        
        for i, role in enumerate(roles):
            blue_champ = blue_team[i]
            red_champ = red_team[i]
            
            
            if division == "MIXED":
                role_stats = {}
                for div_stats in self.matchups.values():
                    div_role_stats = div_stats.get(role, {})
                    for blue_champ, red_champs in div_role_stats.items():
                        if blue_champ not in role_stats:
                            role_stats[blue_champ] = {}
                    for red_champ, stats in red_champs.items():
                        if red_champ not in role_stats[blue_champ]:
                            role_stats[blue_champ][red_champ] = {"wins": 0, "matches": 0}
                        role_stats[blue_champ][red_champ]["wins"] += stats["wins"]
                        role_stats[blue_champ][red_champ]["matches"] += stats["matches"]
            else:
                role_stats = self.matchups.get(division, {}).get(role, {})

            champ_stats = role_stats.get(blue_champ, {}).get(red_champ)
            
            if champ_stats and champ_stats["matches"] > 0:
                probabilities.append(champ_stats["wins"] / champ_stats["matches"])

        if not probabilities:
            return 0.5
        
        return sum(probabilities) / len(probabilities)