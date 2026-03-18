import sys
from PyQt6.QtWidgets import QApplication
from view.main_window import MainWindow
from controller.controller import Controller
from model.data_manager import DataManager
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_predictor():
    app = QApplication(sys.argv)
    
    dm = DataManager()
    champ_dict = dm.get_champion_names()
    champ_names = sorted(list(champ_dict.values()))
    
    view = MainWindow(champ_names)
    controller = Controller(view=view)
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_predictor()