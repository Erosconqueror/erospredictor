import json
from model.statistical import StatisticalModel
from model.data_manager import DataManager
from model.core_model import CoreModel

class Controller:
    """The Controller coordinates the prediction process, validates inputs, and formats results for the view."""
    def __init__(self, view=None, dev_mode=False):
        self.view = view
        self.db = DataManager(dev_mode=dev_mode)
        self.stat_model = StatisticalModel(self.db)
        self.core_model = CoreModel(self.stat_model)
        
        try:
            with open("data/meta_champs.json", "r", encoding="utf-8") as f:
                self.core_model.load_meta_data(json.load(f))
        except Exception:
            pass

    def validate_draft(self, blue_picks: list, red_picks: list, blue_bans: list, red_bans: list) -> tuple:
        """Validates the current draft state and returns a (is_valid, error_message) tuple."""
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
        """Returns a hex color string based on winrate percentage thresholds."""
        if wr >= 57.0: return "#00E676"      
        elif wr >= 54.0: return "#66BB6A"    
        elif wr >= 51.5: return "#A5D6A7"    
        elif wr >= 48.5: return "#E0E0E0"    
        elif wr >= 46.0: return "#EF9A9A"    
        elif wr >= 43.0: return "#EF5350"    
        return "#D32F2F"

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        """Predicts match outcome and returns formatted UI data including colors and progress bar values."""
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
        """Fetches champion recommendations and pre-calculates the UI display color for each."""
        recs = self.core_model.recommend_champs(div, blue, red, bans, team, r_idx, top_k, filter_off_meta)
        for rec in recs:
            rec['display_color'] = self.get_winrate_color(rec['wr'])
        return recs