import sys
import json
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from view.main_window import MainWindow
from controller.controller import Controller
from model.data_manager import DataManager

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_predictor():
    app = QApplication(sys.argv)
    
    dm = DataManager()
    champ_dict = dm.get_champion_names()
    champ_names = sorted(list(champ_dict.values()))
    
    name_to_id = {name: int(champ_id) for champ_id, name in champ_dict.items()}
    
    # ÚJ: Beolvassuk a Meta hősöket, ha létezik a fájl
    meta_champs = {}
    meta_path = Path("data/meta_champs.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_champs = json.load(f)
    
    view = MainWindow(champ_names)
    controller = Controller(view=view)
    
    view.set_controller_and_mapping(controller, name_to_id)
    
    # ÚJ: Átadjuk a Meta JSON-t a felületnek
    view.set_meta_data(meta_champs)
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_predictor()