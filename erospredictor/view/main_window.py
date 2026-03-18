from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self, champion_names):
        super().__init__()
        
        self.setWindowTitle("EROS Predictor - League of Legends ML")
        self.setMinimumSize(800, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        title = QLabel("Gyozelem Esely es Hos Ajanlo")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        main_layout.addWidget(title)
        
        teams_layout = QHBoxLayout()
        
        self.blue_combos = []
        blue_layout = QVBoxLayout()
        blue_label = QLabel("Kek Csapat (Blue Team)")
        blue_label.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        blue_layout.addWidget(blue_label)
        
        champs = champion_names
        for _ in range(5):
            combo = QComboBox()
            combo.addItem("--- Valassz host ---")
            combo.addItems(champs)
            self.blue_combos.append(combo)
            blue_layout.addWidget(combo)
            
        teams_layout.addLayout(blue_layout)
        
        self.red_combos = []
        red_layout = QVBoxLayout()
        red_label = QLabel("Piros Csapat (Red Team)")
        red_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        red_layout.addWidget(red_label)
        
        for _ in range(5):
            combo = QComboBox()
            combo.addItem("--- Valassz host ---")
            combo.addItems(champs)
            self.red_combos.append(combo)
            red_layout.addWidget(combo)
            
        teams_layout.addLayout(red_layout)
        main_layout.addLayout(teams_layout)
        
        self.predict_btn = QPushButton("Predikcio Futtatasa")
        self.predict_btn.setMinimumHeight(50)
        self.predict_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(self.predict_btn)
        
        self.result_label = QLabel("Kerlek valaszd ki a hosoket es nyomj a gombra!")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; margin-top: 20px;")
        main_layout.addWidget(self.result_label)

    def get_selected_champions(self):
        blue_team = [c.currentText() for c in self.blue_combos if c.currentText() != "--- Valassz host ---"]
        red_team = [c.currentText() for c in self.red_combos if c.currentText() != "--- Valassz host ---"]
        return blue_team, red_team

    def update_result(self, text):
        self.result_label.setText(text)