import json
from model.statistical import StatisticalModel
from model.data_manager import DataManager
from model.core_model import CoreModel

class Controller:
    """The Controller class only serves as the intermediary between the view and the model components of the application."""
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

    def predict_match(self, div: str, blue: list, red: list) -> dict:
        return self.core_model.predict_match(div, blue, red)

    def recommend_champs(self, div: str, blue: list, red: list, bans: list, team: str, r_idx: int, top_k: int = 5, filter_off_meta: bool = True) -> list:
        return self.core_model.recommend_champs(div, blue, red, bans, team, r_idx, top_k, filter_off_meta)