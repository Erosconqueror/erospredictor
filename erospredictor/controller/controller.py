import json
from model.statistical import StatisticalModel
from model.data_manager import DataManager
from model.core_model import CoreModel

class Controller:
    """The controller class manages the interaction between the data, models, and the view. It handles draft validation, match prediction, and champion recommendations."""
    def __init__(self, view=None, dev_mode=False):
        self.view = view
        self.db = DataManager(dev_mode=dev_mode)
        self.stat_model = StatisticalModel(self.db)
        self.core_model = CoreModel(self.stat_model)
        
        meta_data = self.db.load_meta_champs()
        if meta_data:
            self.core_model.load_meta_data(meta_data)

    def validate_draft(self, blue_picks: list, red_picks: list, blue_bans: list, red_bans: list) -> tuple:
        """Validates the draft by checking for duplicate picks/bans and ensuring picks are not banned."""
        b_bans = [i for i in blue_bans if i > 0]
        r_bans = [i for i in red_bans if i > 0]
        b_picks = [i for i in blue_picks if i > 0]
        r_picks = [i for i in red_picks if i > 0]

        if len(b_bans) != len(set(b_bans)): 
            return False, "A Kék csapat nem bannolhatja ugyanazt kétszer!"
        if len(r_bans) != len(set(r_bans)): 
            return False, "A Piros csapat nem bannolhatja ugyanazt kétszer!"

        picks = b_picks + r_picks
        if len(picks) != len(set(picks)): 
            return False, "Egy hős csak egyszer lehet kiválasztva!"

        all_bans = set(b_bans + r_bans)
        for pick in picks:
            if pick in all_bans: 
                return False, "Kiválasztott hős nem lehet a tiltólistán!"

        return True, ""

    def get_winrate_color(self, wr: float) -> str:
        """Returns a color code based on the win rate for UI display."""
        if wr >= 57.0: return "#00E676"      
        elif wr >= 54.0: return "#66BB6A"    
        elif wr >= 51.5: return "#A5D6A7"    
        elif wr >= 48.5: return "#E0E0E0"    
        elif wr >= 46.0: return "#EF9A9A"    
        elif wr >= 43.0: return "#EF5350"    
        return "#D32F2F"

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        """Takes the division and team compositions, runs the prediction model, and formats the results for UI display."""
        res = self.core_model.predict_match(div, blue, red)
        b_wr = res['blue_win_prob']
        r_wr = res['red_win_prob']
        
        return {
            'blue_text': f"Kék: {b_wr:.1f}%",
            'red_text': f"Piros: {r_wr:.1f}%",
            'bar_val': int(b_wr),
            'blue_win_prob': b_wr,
            'red_win_prob': r_wr
        }

    def recommend_champs(self, div: str, blue: list, red: list, bans: list, team: str, r_idx: int, top_k: int = 5, filter_off_meta: bool = True) -> list:
        """Returns a list of recommended champions based on the current draft state, including win rates and display colors for UI."""
        recs = self.core_model.recommend_champs(div, blue, red, bans, team, r_idx, top_k, filter_off_meta)
        for rec in recs:
            rec['display_color'] = self.get_winrate_color(rec['wr'])
        return recs