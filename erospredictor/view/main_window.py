from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QApplication, QCheckBox, QGroupBox)
from PyQt6.QtCore import Qt
from configs import TARGET_TIERS

class MainWindow(QMainWindow):
    def __init__(self, champion_names):
        super().__init__()
        self.setWindowTitle("Eros Predictor - ML E-sport Elemző")
        self.setMinimumSize(850, 750) # Kicsit megnöveltük a magasságot a banok miatt
        self.champion_names = champion_names
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        title = QLabel("League of Legends Mérkőzés Prediktor és Ajánló")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        rank_layout = QHBoxLayout()
        rank_label = QLabel("Modell / Divízió:")
        rank_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.rank_combo = QComboBox()
        self.rank_combo.addItem("MIXED")
        self.rank_combo.addItems(TARGET_TIERS) 
        self.rank_combo.setStyleSheet("font-size: 14px; padding: 5px;")
        rank_layout.addWidget(rank_label)
        rank_layout.addWidget(self.rank_combo)
        rank_layout.addStretch()
        main_layout.addLayout(rank_layout)

        # ==========================================
        # ÚJ RÉSZ: TILTÁSOK (BANS) SZEKCIÓ
        # ==========================================
        bans_group = QGroupBox("🚫 Kitiltott hősök (Bans)")
        bans_group.setStyleSheet("font-weight: bold;")
        bans_layout = QVBoxLayout()
        
        blue_bans_layout = QHBoxLayout()
        blue_bans_label = QLabel("Kék tiltások:")
        blue_bans_label.setStyleSheet("color: blue;")
        blue_bans_layout.addWidget(blue_bans_label)
        self.blue_ban_combos = []
        for _ in range(5):
            combo = self.create_searchable_combo()
            self.blue_ban_combos.append(combo)
            blue_bans_layout.addWidget(combo)
            
        red_bans_layout = QHBoxLayout()
        red_bans_label = QLabel("Piros tiltások:")
        red_bans_label.setStyleSheet("color: red;")
        red_bans_layout.addWidget(red_bans_label)
        self.red_ban_combos = []
        for _ in range(5):
            combo = self.create_searchable_combo()
            self.red_ban_combos.append(combo)
            red_bans_layout.addWidget(combo)
            
        bans_layout.addLayout(blue_bans_layout)
        bans_layout.addLayout(red_bans_layout)
        bans_group.setLayout(bans_layout)
        main_layout.addWidget(bans_group)

        # ==========================================
        # CSAPATOK SZEKCIÓ
        # ==========================================
        teams_layout = QHBoxLayout()
        
        # Kék csapat
        blue_layout = QVBoxLayout()
        blue_label = QLabel("Kék Csapat (Blue Team)")
        blue_label.setStyleSheet("color: blue; font-weight: bold; font-size: 16px;")
        blue_layout.addWidget(blue_label)
        
        self.blue_combos = []
        roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
        for i in range(5):
            row_layout = QHBoxLayout()
            role_lbl = QLabel(roles[i] + ":")
            role_lbl.setFixedWidth(55)
            combo = self.create_searchable_combo()
            self.blue_combos.append(combo)
            row_layout.addWidget(role_lbl)
            row_layout.addWidget(combo)
            blue_layout.addLayout(row_layout)
            
        # Piros csapat
        red_layout = QVBoxLayout()
        red_label = QLabel("Piros Csapat (Red Team)")
        red_label.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        red_layout.addWidget(red_label)
        
        self.red_combos = []
        for i in range(5):
            row_layout = QHBoxLayout()
            role_lbl = QLabel(roles[i] + ":")
            role_lbl.setFixedWidth(55)
            combo = self.create_searchable_combo()
            self.red_combos.append(combo)
            row_layout.addWidget(role_lbl)
            row_layout.addWidget(combo)
            red_layout.addLayout(row_layout)
            
        teams_layout.addLayout(blue_layout)
        teams_layout.addLayout(red_layout)
        main_layout.addLayout(teams_layout)
        
        # ==========================================
        # PREDIKTOR ÉS AJÁNLÓ PANELEK
        # ==========================================
        controls_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("🏆 Predikció Indítása")
        self.predict_btn.setStyleSheet("font-size: 16px; font-weight: bold; padding: 15px; background-color: #4CAF50; color: white; min-width: 200px;")
        controls_layout.addWidget(self.predict_btn)
        
        rec_group = QGroupBox("💡 Hős Ajánló Rendszer")
        rec_group.setStyleSheet("font-weight: bold;")
        rec_layout = QVBoxLayout()
        
        rec_row1 = QHBoxLayout()
        self.rec_team_combo = QComboBox()
        self.rec_team_combo.addItems(["Kék Csapatba (Blue)", "Piros Csapatba (Red)"])
        self.rec_role_combo = QComboBox()
        self.rec_role_combo.addItems(["Top", "Jungle", "Mid", "ADC", "Support"])
        rec_row1.addWidget(QLabel("Hova:"))
        rec_row1.addWidget(self.rec_team_combo)
        rec_row1.addWidget(self.rec_role_combo)
        
        rec_row2 = QHBoxLayout()
        self.meta_checkbox = QCheckBox("Csak 'Meta' hősök (min. 500 pick / 1%)")
        self.meta_checkbox.setChecked(True)
        self.recommend_btn = QPushButton("Top 3 Hős Ajánlása")
        self.recommend_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white; padding: 5px;")
        rec_row2.addWidget(self.meta_checkbox)
        rec_row2.addWidget(self.recommend_btn)
        
        rec_layout.addLayout(rec_row1)
        rec_layout.addLayout(rec_row2)
        rec_group.setLayout(rec_layout)
        
        controls_layout.addWidget(rec_group)
        main_layout.addLayout(controls_layout)
        
        self.result_label = QLabel("Válaszd ki a hősöket és a divíziót a kezdéshez!")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 20px; border: 2px dashed gray;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        main_layout.addWidget(self.result_label)

        # Események
        self.predict_btn.clicked.connect(self.on_predict_clicked)
        self.recommend_btn.clicked.connect(self.on_recommend_clicked)
        
        self.controller = None
        self.name_to_id = {} 
        self.meta_champs = {}

    def set_meta_data(self, meta_data):
        self.meta_champs = meta_data

    def create_searchable_combo(self):
        combo = QComboBox()
        combo.addItems([""] + self.champion_names)
        combo.setStyleSheet("font-size: 14px; padding: 3px;")
        combo.setEditable(True)
        combo.lineEdit().setPlaceholderText("Gépelj ide egy hőst...") 
        combo.setCurrentIndex(0) 
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.completer().setCompletionMode(combo.completer().CompletionMode.PopupCompletion)
        combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        return combo

    def set_controller_and_mapping(self, controller, name_to_id_dict):
        self.controller = controller
        self.name_to_id = name_to_id_dict

    def get_team_ids(self, combos):
        team_ids = []
        for combo in combos:
            name = combo.currentText().strip()
            if name and name in self.name_to_id:
                team_ids.append(self.name_to_id[name])
            else:
                team_ids.append(0)
        return team_ids

    def get_all_unavailable_ids(self):
        """Összegyűjti az összes jelenleg foglalt hős ID-ját (pickek + banok)"""
        blue_ids = self.get_team_ids(self.blue_combos)
        red_ids = self.get_team_ids(self.red_combos)
        blue_ban_ids = self.get_team_ids(self.blue_ban_combos)
        red_ban_ids = self.get_team_ids(self.red_ban_combos)
        
        picked = [i for i in blue_ids + red_ids if i > 0]
        banned = [i for i in blue_ban_ids + red_ban_ids if i > 0]
        return set(picked), set(banned)

    def on_predict_clicked(self):
        if not self.controller: return
        
        # Validáció: Egy hős csak egyszer szerepelhet
        blue_team = self.get_team_ids(self.blue_combos)
        red_team = self.get_team_ids(self.red_combos)
        blue_bans = self.get_team_ids(self.blue_ban_combos)
        red_bans = self.get_team_ids(self.red_ban_combos)
        
        all_ids = [i for i in blue_team + red_team + blue_bans + red_bans if i > 0]
        if len(all_ids) != len(set(all_ids)):
            self.result_label.setText("⚠️ HIBA: Egy hős csak egyszer szerepelhet a pickek és tiltások között!")
            return

        division = self.rank_combo.currentText()
        
        self.result_label.setText("Predikció számítása... Kérlek várj!")
        QApplication.instance().processEvents() 
        
        result = self.controller.predict_match(division, blue_team, red_team)
        
        text = (f"=== 🏆 Predikció Eredmény ({division}) ===\n\n"
                f"🟦 Kék csapat (Blue) győzelmi esély: {result['blue_win_prob']:.2f}%\n"
                f"🟥 Piros csapat (Red) győzelmi esély: {result['red_win_prob']:.2f}%")
        self.result_label.setText(text)

    def on_recommend_clicked(self):
        if not self.controller: return
        
        # Validáció: Egy hős csak egyszer szerepelhet
        blue_team = self.get_team_ids(self.blue_combos)
        red_team = self.get_team_ids(self.red_combos)
        blue_bans = self.get_team_ids(self.blue_ban_combos)
        red_bans = self.get_team_ids(self.red_ban_combos)
        
        all_ids = [i for i in blue_team + red_team + blue_bans + red_bans if i > 0]
        if len(all_ids) != len(set(all_ids)):
            self.result_label.setText("⚠️ HIBA: Kérlek javítsd a duplikált hősöket az ajánlás előtt!")
            return

        division = self.rank_combo.currentText()
        picked, banned = self.get_all_unavailable_ids()
        all_bans = list(banned)
        
        picking_team = "blue" if self.rec_team_combo.currentIndex() == 0 else "red"
        picking_index = self.rec_role_combo.currentIndex()
        
        # Validáció: Foglalt helyre nem ajánlunk
        current_id = blue_team[picking_index] if picking_team == "blue" else red_team[picking_index]
        if current_id > 0:
            self.result_label.setText(f"⚠️ Erre a pozícióra ({self.rec_role_combo.currentText()}) már választottál hőst! Töröld ki az ajánláshoz.")
            return
        
        # Meta szűrő logika
        allowed_champs = None
        if self.meta_checkbox.isChecked() and self.meta_champs:
            lookup_div = division if division in self.meta_champs else "MIXED"
            role_str = str(picking_index)
            
            if lookup_div in self.meta_champs and role_str in self.meta_champs[lookup_div]:
                allowed_champs = set(self.meta_champs[lookup_div][role_str])
        
        self.result_label.setText(f"Szimuláció futtatása a(z) {self.rec_role_combo.currentText()} pozícióra... Várj egy kicsit!")
        QApplication.instance().processEvents()
        
        recommendations = self.controller.recommend_champions(
            division, blue_team, red_team, bans=all_bans, 
            picking_team=picking_team, picking_index=picking_index, 
            top_k=3, allowed_champs=allowed_champs
        )
        
        id_to_name = {v: k for k, v in self.name_to_id.items()}
        csapat_nev = "Kék (Blue)" if picking_team == "blue" else "Piros (Red)"
        role_nev = self.rec_role_combo.currentText()
        
        res_text = f"💡 Ajánlott hősök ide: {csapat_nev} - {role_nev}\n"
        if allowed_champs:
            res_text += f"(Csak a divízióban Meta hősökből szűrve)\n\n"
        else:
            res_text += "\n"
        
        if not recommendations:
            res_text += "Nem található megfelelő hős a feltételek alapján!"
        else:
            for i, rec in enumerate(recommendations):
                champ_name = id_to_name.get(rec['champion_id'], f"Ismeretlen (ID: {rec['champion_id']})")
                res_text += f"{i+1}. {champ_name} ➔ Győzelmi esély a csapattal: {rec['expected_winrate']:.2f}%\n"
            
        self.result_label.setText(res_text)