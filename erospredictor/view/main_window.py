import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QApplication, QCheckBox, QGroupBox)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from configs import TARGET_TIERS

class MainWindow(QMainWindow):
    def __init__(self, champion_names, name_to_id): 
        super().__init__()
        self.setWindowTitle("Eros Predictor - ML E-sport Elemző")
        self.setMinimumSize(950, 850) # Kicsit nagyobbra vettük, hogy minden kényelmesen elférjen
        self.champion_names = champion_names
        self.name_to_id = name_to_id 
        
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E24;
                color: #FFFFFF;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel { color: #E0E0E0; }
            QComboBox {
                background-color: #2D2D36;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 4px;
                color: white;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2D2D36;
                color: white;
                selection-background-color: #4CAF50;
            }
            QGroupBox {
                border: 2px solid #3A3A45;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: bold;
                color: #A0A0B5;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            /* A Kitiltott hősök panel piros kerete */
            QGroupBox#bansGroup {
                border: 2px solid #E57373;
                color: #E57373;
            }
            QCheckBox { color: #E0E0E0; }
        """)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        title = QLabel("League of Legends Mérkőzés Prediktor és Ajánló")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px; color: #4CAF50;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        

        rank_layout = QHBoxLayout()
        rank_label = QLabel("Modell / Divízió:")
        rank_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.rank_combo = QComboBox()
        self.rank_combo.setMinimumWidth(200)
        self.rank_combo.setIconSize(QSize(24, 24))
        self.rank_combo.setStyleSheet("font-size: 15px; padding: 5px; height: 35px;")
        

        all_ranks_icon = os.path.abspath("assets/ranks/All Ranks.png")
        if os.path.exists(all_ranks_icon):
            self.rank_combo.addItem(QIcon(all_ranks_icon), "ALL RANKS")
        else:
            self.rank_combo.addItem("ALL RANKS")
            

        for tier in TARGET_TIERS:
 
            tier_formatted = tier.capitalize() 
            icon_path = os.path.abspath(f"assets/ranks/{tier_formatted}.png")
            
            if os.path.exists(icon_path):
                self.rank_combo.addItem(QIcon(icon_path), tier)
            else:
                self.rank_combo.addItem(tier)
                
        rank_layout.addWidget(rank_label)
        rank_layout.addWidget(self.rank_combo)
        rank_layout.addStretch()
        main_layout.addLayout(rank_layout)

        bans_group = QGroupBox("Kitiltott hősök (Bans)")
        bans_group.setObjectName("bansGroup") 
        bans_layout = QVBoxLayout()
        
        blue_bans_layout = QHBoxLayout()
        blue_bans_label = QLabel("Kék tiltások:")
        blue_bans_label.setStyleSheet("color: #64B5F6; font-weight: bold;") 
        blue_bans_layout.addWidget(blue_bans_label)
        self.blue_ban_combos = []
        for _ in range(5):
            combo = self.create_searchable_combo()
            self.blue_ban_combos.append(combo)
            blue_bans_layout.addWidget(combo)
            
        red_bans_layout = QHBoxLayout()
        red_bans_label = QLabel("Piros tiltások:")
        red_bans_label.setStyleSheet("color: #E57373; font-weight: bold;")
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

   
        teams_layout = QHBoxLayout()
        
        blue_layout = QVBoxLayout()
        blue_label = QLabel("Kék Csapat (Blue Team)")
        blue_label.setStyleSheet("color: #64B5F6; font-weight: bold; font-size: 18px; padding-bottom: 5px;")
        blue_layout.addWidget(blue_label)
        
        self.blue_combos = []
        roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
        for i in range(5):
            row_layout = QHBoxLayout()
            role_lbl = QLabel(roles[i] + ":")
            role_lbl.setFixedWidth(65)
            role_lbl.setMinimumHeight(45) # Magasabb felirat
            role_lbl.setStyleSheet("font-weight: bold; color: #888; font-size: 14px;")
            combo = self.create_searchable_combo()
            self.blue_combos.append(combo)
            row_layout.addWidget(role_lbl)
            row_layout.addWidget(combo)
            blue_layout.addLayout(row_layout)
            
        red_layout = QVBoxLayout()
        red_label = QLabel("Piros Csapat (Red Team)")
        red_label.setStyleSheet("color: #E57373; font-weight: bold; font-size: 18px; padding-bottom: 5px;")
        red_layout.addWidget(red_label)
        
        self.red_combos = []
        for i in range(5):
            row_layout = QHBoxLayout()
            role_lbl = QLabel(roles[i] + ":")
            role_lbl.setFixedWidth(65)
            role_lbl.setMinimumHeight(45) # Magasabb felirat
            role_lbl.setStyleSheet("font-weight: bold; color: #888; font-size: 14px;")
            combo = self.create_searchable_combo()
            self.red_combos.append(combo)
            row_layout.addWidget(role_lbl)
            row_layout.addWidget(combo)
            red_layout.addLayout(row_layout)
            
        teams_layout.addLayout(blue_layout)
        teams_layout.addLayout(red_layout)
        main_layout.addLayout(teams_layout)
        

        controls_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("PREDIKCIÓ INDÍTÁSA")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; font-weight: bold; padding: 15px; 
                background-color: #4CAF50; color: white; border-radius: 6px; min-width: 200px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        controls_layout.addWidget(self.predict_btn)
        
        rec_group = QGroupBox("Okos Ajánló Rendszer")
        rec_layout = QVBoxLayout()
        
        rec_row1 = QHBoxLayout()
        self.rec_team_combo = QComboBox()
        self.rec_team_combo.addItems(["Kék Csapatba", "Piros Csapatba"])
        self.rec_team_combo.setStyleSheet("height: 30px;")
        self.rec_role_combo = QComboBox()
        self.rec_role_combo.addItems(["Top", "Jungle", "Mid", "ADC", "Support"])
        self.rec_role_combo.setStyleSheet("height: 30px;")
        rec_row1.addWidget(QLabel("Hova:"))
        rec_row1.addWidget(self.rec_team_combo)
        rec_row1.addWidget(self.rec_role_combo)
        
        rec_row2 = QHBoxLayout()
        self.meta_checkbox = QCheckBox("Csak 'Meta' hősök (min. 500 pick / 1%)")
        self.meta_checkbox.setChecked(True)
        self.recommend_btn = QPushButton("Top 3 Ajánlása")
        self.recommend_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold; background-color: #2196F3; color: white; padding: 10px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        rec_row2.addWidget(self.meta_checkbox)
        rec_row2.addWidget(self.recommend_btn)
        
        rec_layout.addLayout(rec_row1)
        rec_layout.addLayout(rec_row2)
        rec_group.setLayout(rec_layout)
        
        controls_layout.addWidget(rec_group)
        main_layout.addLayout(controls_layout)
        
        self.result_label = QLabel("Készen áll a draftolásra! Válassz hősöket és divíziót!")
        self.result_label.setStyleSheet("font-size: 16px; padding: 15px; border: 2px dashed #4CAF50; border-radius: 8px; background-color: #25252D;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(240) 
        main_layout.addWidget(self.result_label)

        self.predict_btn.clicked.connect(self.on_predict_clicked)
        self.recommend_btn.clicked.connect(self.on_recommend_clicked)
        
        self.controller = None
        self.meta_champs = {}

    def set_meta_data(self, meta_data):
        self.meta_champs = meta_data

    def create_searchable_combo(self):
        combo = QComboBox()
        combo.setFixedWidth(145)  
        combo.setMinimumHeight(45)
        combo.setIconSize(QSize(36, 36))
        combo.setStyleSheet("font-size: 14px; padding: 3px;")
        
        combo.addItem(QIcon(), "") 
        
        for name in self.champion_names:
            champ_id = self.name_to_id.get(name)
            icon_abs = os.path.abspath(f"assets/icons/{champ_id}.png")
            
            if os.path.exists(icon_abs):
                combo.addItem(QIcon(icon_abs), name)
            else:
                combo.addItem(name) 
                
        combo.setEditable(True)
        combo.lineEdit().setPlaceholderText("Gépelj...") 
        combo.setCurrentIndex(0) 
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.completer().setCompletionMode(combo.completer().CompletionMode.PopupCompletion)
        combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        
        return combo

    def set_controller_and_mapping(self, controller, name_to_id_dict):
        self.controller = controller

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
        blue_ids = self.get_team_ids(self.blue_combos)
        red_ids = self.get_team_ids(self.red_combos)
        blue_ban_ids = self.get_team_ids(self.blue_ban_combos)
        red_ban_ids = self.get_team_ids(self.red_ban_combos)
        
        picked = [i for i in blue_ids + red_ids if i > 0]
        banned = [i for i in blue_ban_ids + red_ban_ids if i > 0]
        return set(picked), set(banned)

    def on_predict_clicked(self):
        if not self.controller: return
        
        blue_team = self.get_team_ids(self.blue_combos)
        red_team = self.get_team_ids(self.red_combos)
        blue_bans = self.get_team_ids(self.blue_ban_combos)
        red_bans = self.get_team_ids(self.red_ban_combos)
        
        all_ids = [i for i in blue_team + red_team + blue_bans + red_bans if i > 0]
        if len(all_ids) != len(set(all_ids)):
            self.result_label.setText("<h3 style='color: #EF5350;'>HIBA: Egy hős csak egyszer szerepelhet!</h3>")
            return

        division_ui = self.rank_combo.currentText()

        division_backend = "MIXED" if division_ui == "ALL RANKS" else division_ui
        
        self.result_label.setText("<h3>Predikció számítása... Kérlek várj!</h3>")
        QApplication.instance().processEvents() 
        
        result = self.controller.predict_match(division_backend, blue_team, red_team)
        
        blue_w = result['blue_win_prob']
        red_w = result['red_win_prob']
        

        text = f"""
        <div style='text-align: center;'>
            <h2 style='color: #E0E0E0; margin-bottom: 15px;'>Predikció Eredmény ({division_ui})</h2>
            
            <table width="100%" height="45" cellspacing="0" cellpadding="0" style="border-radius: 8px;">
                <tr>
                    <td width="{blue_w}%" style="background-color: #64B5F6; text-align: center; font-size: 20px; font-weight: bold; color: #1E1E24;">
                        Kék: {blue_w:.1f}%
                    </td>
                    <td width="{red_w}%" style="background-color: #E57373; text-align: center; font-size: 20px; font-weight: bold; color: #1E1E24;">
                        Piros: {red_w:.1f}%
                    </td>
                </tr>
            </table>
        </div>
        """
        self.result_label.setText(text)

    def get_winrate_color(self, wr):
        """Dinamikusan színezi a winratet. Zöldebb a jó, pirosabb a rossz."""
        if wr >= 54.0: return "#43A047"      
        elif wr >= 51.5: return "#66BB6A"    
        elif wr >= 48.5: return "#E0E0E0"    
        elif wr >= 46.0: return "#E57373"    
        else: return "#E53935"               

    def on_recommend_clicked(self):
        if not self.controller: return
        
        blue_team = self.get_team_ids(self.blue_combos)
        red_team = self.get_team_ids(self.red_combos)
        blue_bans = self.get_team_ids(self.blue_ban_combos)
        red_bans = self.get_team_ids(self.red_ban_combos)
        
        all_ids = [i for i in blue_team + red_team + blue_bans + red_bans if i > 0]
        if len(all_ids) != len(set(all_ids)):
            self.result_label.setText("<h3 style='color: #EF5350;'>HIBA: Kérlek javítsd a duplikált hősöket az ajánlás előtt!</h3>")
            return

        division_ui = self.rank_combo.currentText()
        division_backend = "MIXED" if division_ui == "ALL RANKS" else division_ui
        
        picked, banned = self.get_all_unavailable_ids()
        all_bans = list(banned)
        
        picking_team = "blue" if self.rec_team_combo.currentIndex() == 0 else "red"
        picking_index = self.rec_role_combo.currentIndex()
        
        current_id = blue_team[picking_index] if picking_team == "blue" else red_team[picking_index]
        if current_id > 0:
            self.result_label.setText(f"<h3 style='color: #FFCA28;'>Erre a pozícióra már választottál! Töröld ki az ajánláshoz.</h3>")
            return
        
        allowed_champs = None
        if self.meta_checkbox.isChecked() and self.meta_champs:
            lookup_div = division_backend if division_backend in self.meta_champs else "MIXED"
            role_str = str(picking_index)
            if lookup_div in self.meta_champs and role_str in self.meta_champs[lookup_div]:
                allowed_champs = set(self.meta_champs[lookup_div][role_str])
        
        self.result_label.setText(f"<h3>Szimuláció futtatása... Várj egy kicsit!</h3>")
        QApplication.instance().processEvents()
        
        recommendations = self.controller.recommend_champions(
            division_backend, blue_team, red_team, bans=all_bans, 
            picking_team=picking_team, picking_index=picking_index, 
            top_k=3, allowed_champs=allowed_champs
        )
        
        id_to_name = {v: k for k, v in self.name_to_id.items()}
        csapat_nev = "Kék (Blue)" if picking_team == "blue" else "Piros (Red)"
        role_nev = self.rec_role_combo.currentText()
        
        res_text = f"<h2 style='color: #2196F3; margin-bottom: 5px;'>Ajánlott hősök: {csapat_nev} - {role_nev}</h2>"
        if allowed_champs:
            res_text += f"<p style='color: #A0A0B5; font-style: italic;'>(Csak {division_ui} Meta hősökből szűrve)</p>"
        
        if not recommendations:
            res_text += "<h3 style='color: #EF5350;'>Nem található megfelelő hős a feltételek alapján!</h3>"
        else:
            res_text += "<table cellpadding='8'>"
            for i, rec in enumerate(recommendations):
                cid = rec['champion_id']
                wr = rec['expected_winrate']
                champ_name = id_to_name.get(cid, "Ismeretlen")
                
                wr_color = self.get_winrate_color(wr)
                
                icon_path = os.path.abspath(f"assets/icons/{cid}.png").replace("\\", "/")
                img_tag = f'<img src="file:///{icon_path}" width="42" height="42" style="border-radius: 6px;">' if os.path.exists(os.path.abspath(f"assets/icons/{cid}.png")) else '❓'
                
                res_text += f"<tr>"
                res_text += f"<td style='font-size: 20px; font-weight: bold; color: #888;'>{i+1}.</td>"
                res_text += f"<td>{img_tag}</td>"
                res_text += f"<td style='font-size: 20px; font-weight: bold; width: 160px;'>{champ_name}</td>"
                res_text += f"<td style='font-size: 18px;'>Győzelmi esély: <span style='color: {wr_color}; font-weight: bold;'>{wr:.2f}%</span></td>"
                res_text += f"</tr>"
            res_text += "</table>"
            
        self.result_label.setText(res_text)