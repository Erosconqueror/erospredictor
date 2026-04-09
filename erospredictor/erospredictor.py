import sys
import json
import os
import ctypes
from PyQt6.QtWidgets import QApplication
from view.main_window import MainWindow
from controller.controller import Controller
from model.data_manager import DataManager

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_predictor():
    """Starts the main predictor application and builds the MVC architecture."""
    try:
        appid = 'erosconqueror.erospredictor.app.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
    except AttributeError:
        pass

    app = QApplication(sys.argv)
    
    db = DataManager(False)
    names_dict = db.get_champion_names()
    names_list = sorted(list(names_dict.values()))
    name_map = {name: int(cid) for cid, name in names_dict.items()}
    
    with open("data/meta_champs.json", "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    view = MainWindow(names_list, name_map)
    ctrl = Controller(view=view)
    
    view.set_controller_and_mapping(ctrl, name_map)
    view.set_meta_data(meta_data)
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_predictor()