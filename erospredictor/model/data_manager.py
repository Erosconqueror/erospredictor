import json
import configs as cfg
from pathlib import Path


class DataManager:

    def __init__(self):
        self.file_path = Path(cfg.FILE_PATH)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_data_file()

    def initialize_data_file(self):
        if not self.file_path.exists():
            with open(self.file_path, 'w') as f:
                json.dump({}, f, indent=2)
    
    def load_all(self):
        self.initialize_data_file()
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_all(self, data):
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_match(self, match_id: str, region: str, data: dict):
        all_data = self.load_all()
        all_data[match_id] = {"region": region, "data": data}
        self.save_all(all_data)

    def get_match(self, match_id: str):
        all_data = self.load_all()
        if isinstance(all_data, dict):
            return all_data.get(match_id)
        return None
    
    def get_all_match_ids(self):
        all_data = self.load_all()
        if isinstance(all_data, dict):
            return list(all_data.keys())
        return []

    def get_champindex_by_id(self, champ_id: int):
        champ_data_path = Path(cfg.CHAMPION_DATA_PATH)
        if not champ_data_path.exists():
            return None
        with open(champ_data_path, 'r') as f:
            champ_data = json.load(f)
            return int(champ_data.get(str(champ_id)))