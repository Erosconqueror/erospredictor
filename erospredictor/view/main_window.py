from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox)
from PyQt6.QtCore import Qt
from configs import TARGET_TIERS  # <--- JAVÍTVA: TIERS a DIVISIONS helyett!

class MainWindow(QMainWindow):
    def __init__(self, champion_names):
        super().__init__()
        self.setWindowTitle("Eros Predictor - ML E-sport Elemző")
        self.setMinimumSize(800, 600)
        self.champion_names = champion_names
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        title = QLabel("League of Legends Mérkőzés Prediktor és Ajánló")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Rang választó
        rank_layout = QHBoxLayout()
        rank_label = QLabel("Modell / Divízió:")
        rank_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.rank_combo = QComboBox()
        self.rank_combo.addItem("MIXED")
        self.rank_combo.addItems(TARGET_TIERS) # <--- JAVÍTVA!
        self.rank_combo.setStyleSheet("font-size: 14px; padding: 5px;")
        rank_layout.addWidget(rank_label)
        rank_layout.addWidget(self.rank_combo)
        rank_layout.addStretch()
        main_layout.addLayout(rank_layout)

        teams_layout = QHBoxLayout()
        
        # Kék csapat
        blue_layout = QVBoxLayout()
        blue_label = QLabel("Kék Csapat (Blue Team)")
        blue_label.setStyleSheet("color: blue; font-weight: bold; font-size: 16px;")
        blue_layout.addWidget(blue_label)
        
        self.blue_combos = []
        for _ in range(5):
            combo = self.create_searchable_combo()
            self.blue_combos.append(combo)
            blue_layout.addWidget(combo)
            
        # Piros csapat
        red_layout = QVBoxLayout()
        red_label = QLabel("Piros Csapat (Red Team)")
        red_label.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        red_layout.addWidget(red_label)
        
        self.red_combos = []
        for _ in range(5):
            combo = self.create_searchable_combo()
            self.red_combos.append(combo)
            red_layout.addWidget(combo)
            
        teams_layout.addLayout(blue_layout)
        teams_layout.addLayout(red_layout)
        main_layout.addLayout(teams_layout)
        
        # Gombok
        button_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Predikció Indítása")
        self.predict_btn.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; background-color: #4CAF50; color: white;")
        
        self.recommend_btn = QPushButton("Hős Ajánlása (1 üres helyre)")
        self.recommend_btn.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px; background-color: #2196F3; color: white;")
        
        button_layout.addWidget(self.predict_btn)
        button_layout.addWidget(self.recommend_btn)
        main_layout.addLayout(button_layout)
        
        self.result_label = QLabel("Válaszd ki a hősöket és a divíziót a kezdéshez!")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 20px; border: 2px dashed gray;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        main_layout.addWidget(self.result_label)

    def create_searchable_combo(self):
        combo = QComboBox()
        combo.addItems(self.champion_names)
        combo.setStyleSheet("font-size: 14px; padding: 5px;")
        
        combo.setEditable(True)
        combo.lineEdit().setPlaceholderText("Gépelj ide egy hőst...") # <--- EZ A TRÜKK!
        combo.setCurrentIndex(-1) # <--- Alapból üresen indul, csak a placeholder látszik
        
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.completer().setCompletionMode(combo.completer().CompletionMode.PopupCompletion)
        combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        return combo