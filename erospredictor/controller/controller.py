from configs import CHAMPION_COUNT
from model.statistical import StatisticalModel
from model.data_manager import DataManager
from model.core_model import CoreModel
from services import feature_flags

class Controller:
    """The controller class manages the interaction between the data, models, and the view. It handles draft validation, match prediction, and champion recommendations."""
    def __init__(self, view=None, dev_mode=False):
        self.view = view
        self.db = DataManager(dev_mode=dev_mode)
        self.stat_model = StatisticalModel(self.db)
        self.cache_error = None
        try:
            self.stat_model.load_cache()
        except (OSError, ValueError, TypeError) as exc:
            self.cache_error = str(exc)
        self.core_model = CoreModel(self.stat_model)
        
        meta_data = self.db.load_meta_champs()
        if meta_data:
            self.core_model.load_meta_data(meta_data)

    def validate_draft(self, blue_picks: list, red_picks: list, blue_bans: list, red_bans: list) -> tuple:
        """Validates the draft by checking for duplicate picks/bans and ensuring picks are not banned."""
        try:
            CoreModel._validate_draft_shape(blue_picks, red_picks)
        except ValueError as exc:
            return False, str(exc)
        if any(not isinstance(cid, int) or isinstance(cid, bool) or not -1 <= cid < CHAMPION_COUNT
               for cid in list(blue_bans) + list(red_bans)):
            return False, "Érvénytelen hősazonosító a tiltások között."
        b_bans = [i for i in blue_bans if i >= 0]
        r_bans = [i for i in red_bans if i >= 0]
        b_picks = [i for i in blue_picks if i >= 0]
        r_picks = [i for i in red_picks if i >= 0]

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

    def recommend_champs(self, div: str, blue: list, red: list, bans: list, team: str, r_idx: int, top_k: int = 5, filter_off_meta: bool = True, history=None) -> list:
        """Returns a list of recommended champions based on the current draft state, including win rates and display colors for UI."""
        recs = self.core_model.recommend_champs(div, blue, red, bans, team, r_idx, top_k, filter_off_meta)
        for rec in recs:
            rec['display_color'] = self.get_winrate_color(rec['wr'])
        if history is not None and feature_flags.PERSONAL_HISTORY_ENABLED:
            from services.personal_history import annotate_recommendations
            recs = annotate_recommendations(recs, history, self.db.get_champion_mapping(), blue, red, team, r_idx)
        return recs

    def explain_recommendations(self, div: str, blue: list, red: list, team: str,
                                r_idx: int, recommendations: list, max_items: int = 5,
                                selected_id=None, history=None) -> list:
        """Run optional diagnostics only after the user requests a second stage."""
        records = self.core_model.explain_recommendations(
            div, blue, red, team, r_idx, recommendations, max_items=max_items, selected_id=selected_id)
        if history is not None and feature_flags.PERSONAL_HISTORY_ENABLED:
            from services.personal_history import annotate_recommendations
            records = annotate_recommendations(records, history, self.db.get_champion_mapping(), blue, red, team, r_idx)
        return records

    def explain_champion_effects(self, div: str, blue: list, red: list, team: str,
                                 r_idx: int, champion_id=None) -> dict:
        """Return optional main-screen effects for a selected pick or candidate."""
        return self.core_model.explain_champion_effects(
            div, blue, red, team, r_idx, champion_id=champion_id)
