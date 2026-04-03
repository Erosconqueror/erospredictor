import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QApplication, QCheckBox, QGroupBox, QProgressBar, QFrame, QButtonGroup, QDialog, QTextEdit)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from configs import TARGET_TIERS

class MainWindow(QMainWindow):
    def __init__(self, champion_names, name_to_id): 
        super().__init__()
        
        #ABLAK ALAPBEÁLLÍTÁSOK
        self.setWindowTitle("Erospredictor")
        self.setFixedSize(1280, 880)

        
        icon_path = os.path.abspath("assets/mainicon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.champion_names = champion_names
        self.name_to_id = name_to_id 
        
        # MODERN SÖTÉT TÉMA
        self.setStyleSheet("""
            * {
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
            }
            QWidget#centralwidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #14141A, stop:1 #262633);
            }
            QLabel { color: #E0E0E0; }
            QComboBox {
                background-color: #2D2D36;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 4px;
                color: white;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1A1A22; 
                color: white;
                selection-background-color: #3949AB;
                border-radius: 4px;
                width: 0px;           
            }
            QComboBox[isBan="true"] {
                border: 2px solid #E53935; 
                background-color: #2A1D20;
                
            }
            QGroupBox {
                border: 2px solid #3A3A45;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                font-weight: bold;
                color: #A0A0B5;
                background-color: transparent;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QGroupBox#bansGroup {
                border: none; 
                color: #A0A0B5;
                background-color: transparent;
            }
            QCheckBox { color: #E0E0E0; font-size: 14px; }
            
            QProgressBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EF5350, stop:1 #C62828);
                border-radius: 12px;
                border: 2px solid #BDBDBD; 
                text-align: center;
                color: transparent; 
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1976D2, stop:1 #42A5F5);
                border-top-left-radius: 10px;
                border-bottom-left-radius: 10px;
                border-top-right-radius: 0px; 
                border-bottom-right-radius: 0px;
                border-right: none; 
            }
            
            QFrame#centerDashboard {
                background-color: rgba(30, 30, 40, 200);
                border: 1px solid #444;
                border-radius: 12px;
            }
            
            QPushButton.toggleBtn {
                background-color: #2D2D36;
                color: #A0A0B5;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton.toggleBtn:checked {
                background-color: #3949AB;
                color: white;
                border: 1px solid #5C6BC0;
            }
            QPushButton.teamBlueBtn:checked { background-color: #1E88E5; border-color: #42A5F5; color: white;}
            QPushButton.teamRedBtn:checked { background-color: #E53935; border-color: #EF5350; color: white;}
            QPushButton.teamBlueBtn { text-align: center; padding-left: 0; }
            QPushButton.teamRedBtn { text-align: center; padding-left: 0; }
        """)
        
        main_widget = QWidget()
        main_widget.setObjectName("centralwidget")
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        space_balancer = QWidget()
        space_balancer.setFixedWidth(270)
        header_layout.addWidget(space_balancer)
        
        title = QLabel("League of Legends Hősválasztási Segédprogram")
        title.setStyleSheet("font-size: 28px; font-weight: 900; letter-spacing: 1px; margin: 10px; color: #FFFFFF;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.help_btn = QPushButton("Súgó")
        self.help_btn.setFixedSize(80, 35)
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A45; color: #E0E0E0; font-weight: bold; border-radius: 6px; font-size: 14px;
            }
            QPushButton:hover { background-color: #4A4A55; }
        """)
        self.help_btn.clicked.connect(self.show_help_dialog)
        header_layout.addWidget(self.help_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        main_layout.addWidget(header_widget)
        
        # RANG VÁLASZTÓ
        rank_layout = QHBoxLayout()
        rank_label = QLabel("Rang Divízió")
        rank_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.rank_combo = QComboBox()
        self.rank_combo.setMinimumWidth(220) 
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
        # TILTÁSOK (BANS) SZEKCIÓ
        bans_group = QGroupBox("Kitiltott hősök (Bans)")
        bans_group.setObjectName("bansGroup")
        bans_layout = QVBoxLayout()
        bans_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        
        # KÉK TILTÁSOK
        blue_bans_layout = QHBoxLayout()
        blue_bans_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        
        blue_bans_label = QLabel("Kék tiltások:")
        blue_bans_label.setFixedWidth(90) 
        blue_bans_label.setStyleSheet("color: #42A5F5; font-weight: bold;") 
        blue_bans_layout.addWidget(blue_bans_label)
        
        self.blue_ban_combos = []
        for _ in range(5):
            combo = self.create_searchable_combo(is_ban=True)
            self.blue_ban_combos.append(combo)
            blue_bans_layout.addWidget(combo)
            
        dummy_blue = QLabel("") 
        dummy_blue.setFixedWidth(90)
        blue_bans_layout.addWidget(dummy_blue)
            
        #  PIROS TILTÁSOK
        red_bans_layout = QHBoxLayout()
        red_bans_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        red_bans_label = QLabel("Piros tiltások:")
        red_bans_label.setFixedWidth(90) 
        red_bans_label.setStyleSheet("color: #EF5350; font-weight: bold;")
        red_bans_layout.addWidget(red_bans_label)
        
        self.red_ban_combos = []
        for _ in range(5):
            combo = self.create_searchable_combo(is_ban=True)
            self.red_ban_combos.append(combo)
            red_bans_layout.addWidget(combo)
            
        dummy_red = QLabel("") 
        dummy_red.setFixedWidth(90)
        red_bans_layout.addWidget(dummy_red)
            
        bans_layout.addLayout(blue_bans_layout)
        bans_layout.addLayout(red_bans_layout)
        bans_group.setLayout(bans_layout)
        main_layout.addWidget(bans_group)

        # 3 OSZLOPOS SZEKCIÓ
        teams_layout = QHBoxLayout()
        
        # BAL OSZLOP: KÉK CSAPAT
        blue_layout = QVBoxLayout()
        blue_label = QLabel("Kék Csapat (Blue Team)")
        blue_label.setStyleSheet("color: #42A5F5; font-weight: bold; font-size: 18px; padding-bottom: 5px;")
        blue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        blue_layout.addWidget(blue_label)
        
        self.blue_combos = []
        self.roles_list = ["Top", "Jungle", "Mid", "ADC", "Support"]
        for i in range(5):
            row_layout = QHBoxLayout()
            role_lbl = QLabel(self.roles_list[i] + ":")
            role_lbl.setFixedWidth(65)
            role_lbl.setMinimumHeight(50) 
            role_lbl.setStyleSheet("font-weight: bold; color: #9E9E9E; font-size: 15px;")
            combo = self.create_searchable_combo(is_ban=False)
            self.blue_combos.append(combo)
            row_layout.addWidget(role_lbl)
            row_layout.addWidget(combo)
            blue_layout.addLayout(row_layout)
            
        teams_layout.addLayout(blue_layout, stretch=1)
        
        # KÖZÉPSŐ OSZLOP: MŰSZERFAL 
        self.center_dashboard = QFrame()
        self.center_dashboard.setObjectName("centerDashboard")
        self.center_dashboard.setFixedWidth(460)
        
        dash_layout = QVBoxLayout(self.center_dashboard)
        dash_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        dash_layout.setContentsMargins(1, 15, 1, 0)
        
        self.dash_title_lbl = QLabel("Válassz hősöket a draftoláshoz!")
        self.dash_title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        self.dash_title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dash_title_lbl.setWordWrap(True)
        dash_layout.addWidget(self.dash_title_lbl)
        
        self.winrate_labels = QWidget()
        wl_layout = QHBoxLayout(self.winrate_labels)
        wl_layout.setContentsMargins(10, 80, 10, 5)
        self.blue_wr_lbl = QLabel("Kék: 50.0%")
        self.blue_wr_lbl.setStyleSheet("color: #64B5F6; font-weight: bold; font-size: 20px;")
        self.red_wr_lbl = QLabel("Piros: 50.0%")
        self.red_wr_lbl.setStyleSheet("color: #E57373; font-weight: bold; font-size: 20px;")
        self.red_wr_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        wl_layout.addWidget(self.blue_wr_lbl)
        wl_layout.addWidget(self.red_wr_lbl)
        dash_layout.addWidget(self.winrate_labels)
        self.winrate_labels.hide()
        
        self.winrate_bar = QProgressBar()
        self.winrate_bar.setFixedHeight(28)
        self.winrate_bar.setRange(0, 100)
        self.winrate_bar.setValue(50)
        dash_layout.addWidget(self.winrate_bar)
        self.winrate_bar.hide()
        
        self.dash_rec_lbl = QLabel("")
        self.dash_rec_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        dash_layout.addWidget(self.dash_rec_lbl)
        self.dash_rec_lbl.hide()
        
        dash_layout.addStretch()
        teams_layout.addWidget(self.center_dashboard)
        
        #JOBB OSZLOP: PIROS CSAPAT
        red_layout = QVBoxLayout()
        red_label = QLabel("Piros Csapat (Red Team)")
        red_label.setStyleSheet("color: #EF5350; font-weight: bold; font-size: 18px; padding-bottom: 5px;")
        red_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        red_layout.addWidget(red_label)
        
        self.red_combos = []
        for i in range(5):
            row_layout = QHBoxLayout()
            role_lbl = QLabel(self.roles_list[i] + ":")
            role_lbl.setFixedWidth(65)
            role_lbl.setMinimumHeight(50) 
            role_lbl.setStyleSheet("font-weight: bold; color: #9E9E9E; font-size: 15px;")
            combo = self.create_searchable_combo(is_ban=False)
            self.red_combos.append(combo)
            row_layout.addWidget(role_lbl)
            row_layout.addWidget(combo)
            red_layout.addLayout(row_layout)
            
        teams_layout.addLayout(red_layout, stretch=1)
        main_layout.addLayout(teams_layout)
        # ALSÓ VEZÉRLŐK
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        bottom_layout.setSpacing(15) 
        
        self.predict_btn = QPushButton("PREDIKCIÓ")
        self.predict_btn.setFixedSize(300, 55) 
        self.predict_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px; font-weight: bold; letter-spacing: 1px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #43A047, stop:1 #2E7D32); 
                color: white; border-radius: 8px;
            }
            QPushButton:hover { background: #388E3C; }
        """)
        bottom_layout.addWidget(self.predict_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        rec_group = QGroupBox("Hős Ajánló")
        rec_group.setFixedWidth(700) 
        rec_layout = QVBoxLayout()
        
        team_switch_layout = QHBoxLayout()
        team_switch_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.team_btn_group = QButtonGroup(self)
        
        self.btn_blue_team = QPushButton("Kék Csapat (Blue)")
        self.btn_blue_team.setCheckable(True)
        self.btn_blue_team.setChecked(True)
        self.btn_blue_team.setFixedSize(160, 35)
        self.btn_blue_team.setProperty("class", "toggleBtn teamBlueBtn")
        
        self.btn_red_team = QPushButton("Piros Csapat (Red)")
        self.btn_red_team.setCheckable(True)
        self.btn_red_team.setFixedSize(160, 35)
        self.btn_red_team.setProperty("class", "toggleBtn teamRedBtn")
        
        self.team_btn_group.addButton(self.btn_blue_team, 0)
        self.team_btn_group.addButton(self.btn_red_team, 1)
        
        team_switch_layout.addWidget(self.btn_blue_team)
        team_switch_layout.addWidget(self.btn_red_team)
        rec_layout.addLayout(team_switch_layout)
        
        role_switch_layout = QHBoxLayout()
        role_switch_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.role_btn_group = QButtonGroup(self)
        
        role_file_map = {"Mid": "Middle", "ADC": "Bottom"}
        
        for i, role in enumerate(self.roles_list):
            btn = QPushButton(f" {role}")
            btn.setCheckable(True)
            btn.setFixedSize(115, 38)
            btn.setProperty("class", "toggleBtn")
            
            role_icon_name = role_file_map.get(role, role)
            icon_path = os.path.abspath(f"assets/roles/{role_icon_name}_icon.png")
            if os.path.exists(icon_path):
                btn.setIcon(QIcon(icon_path))
                btn.setIconSize(QSize(20, 20))
                
            if i == 0: btn.setChecked(True) 
            self.role_btn_group.addButton(btn, i)
            role_switch_layout.addWidget(btn)
            
        rec_layout.addLayout(role_switch_layout)
        
        rec_row3 = QHBoxLayout()
        rec_row3.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.meta_checkbox = QCheckBox("Off-Meta hősök szűrése")
        self.meta_checkbox.setChecked(True)
        self.recommend_btn = QPushButton("AJÁNLÁS")
        self.recommend_btn.setFixedSize(250, 40)
        self.recommend_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E88E5, stop:1 #1565C0); 
                color: white; border-radius: 6px;
            }
            QPushButton:hover { background: #1976D2; }
        """)
        rec_row3.addWidget(self.meta_checkbox)
        rec_row3.addWidget(self.recommend_btn)
        
        rec_layout.addLayout(rec_row3)
        rec_group.setLayout(rec_layout)
        
        bottom_layout.addWidget(rec_group, alignment=Qt.AlignmentFlag.AlignHCenter)
        main_layout.addWidget(bottom_container)

        self.predict_btn.clicked.connect(self.on_predict_clicked)
        self.recommend_btn.clicked.connect(self.on_recommend_clicked)
        
        self.controller = None
        self.meta_champs = {}

    def set_meta_data(self, meta_data):
        self.meta_champs = meta_data

    def create_searchable_combo(self, is_ban=False):
        combo = QComboBox()
        if is_ban:
            combo.setFixedWidth(130)   
            combo.setMinimumHeight(35) 
            combo.setIconSize(QSize(24, 24)) 
            combo.setProperty("isBan", True)
        else:
            combo.setFixedWidth(150)   
            combo.setMinimumHeight(50) 
            combo.setIconSize(QSize(42, 42)) 
            
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
        combo.editTextChanged.connect(lambda text, cb=combo: self.handle_combo_text_change(cb, text))
        
        return combo

    def handle_combo_text_change(self, combo, text):
        if text.strip() == "":
            combo.setCurrentIndex(0)

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

    def reset_dashboard(self):
        self.dash_title_lbl.show()
        self.winrate_labels.hide()
        self.winrate_bar.hide()
        self.dash_rec_lbl.hide()

    def validate_draft(self):
        blue_picks = [i for i in self.get_team_ids(self.blue_combos) if i > 0]
        red_picks = [i for i in self.get_team_ids(self.red_combos) if i > 0]
        blue_bans = [i for i in self.get_team_ids(self.blue_ban_combos) if i > 0]
        red_bans = [i for i in self.get_team_ids(self.red_ban_combos) if i > 0]

        if len(blue_bans) != len(set(blue_bans)): 
            return False, "A Kék csapat nem bannolhatja ugyanazt a hőst kétszer!"
        if len(red_bans) != len(set(red_bans)): 
            return False, "A Piros csapat nem bannolhatja ugyanazt a hőst kétszer!"

        all_picks = blue_picks + red_picks
        if len(all_picks) != len(set(all_picks)): 
            return False, "Egy hős csak egyszer lehet kiválasztva (pick)!"

        all_bans = set(blue_bans + red_bans)
        for pick in all_picks:
            if pick in all_bans:
                return False, "Kiválasztott hős nem lehet a tiltólistán!"

        return True, ""

    def on_predict_clicked(self):
        if not self.controller: return
        self.reset_dashboard()
        
        is_valid, error_msg = self.validate_draft()
        if not is_valid:
            self.dash_title_lbl.setText(f"<span style='color: #EF5350;'>HIBA: {error_msg}</span>")
            return

        blue_team = self.get_team_ids(self.blue_combos)
        red_team = self.get_team_ids(self.red_combos)

        division_ui = self.rank_combo.currentText()
        division_backend = "MIXED" if division_ui == "ALL RANKS" else division_ui
        
        self.dash_title_lbl.setText("Számítás folyamatban...")
        QApplication.instance().processEvents() 
        
        result = self.controller.predict_match(division_backend, blue_team, red_team)
        
        blue_w = result['blue_win_prob']
        red_w = result['red_win_prob']
        
        self.dash_title_lbl.setText(f"<span style='color: #FFFFFF; font-size: 20px;'>Prediktált Győzelmi Arány ({division_ui})</span>")
        
        self.blue_wr_lbl.setText(f"Kék: {blue_w:.1f}%")
        self.red_wr_lbl.setText(f"Piros: {red_w:.1f}%")
        self.winrate_bar.setValue(int(blue_w)) 
        
        self.winrate_labels.show()
        self.winrate_bar.show()

    def get_winrate_color(self, wr):
        if wr >= 57.0: return "#00E676"      
        elif wr >= 54.0: return "#66BB6A"    
        elif wr >= 51.5: return "#A5D6A7"    
        elif wr >= 48.5: return "#E0E0E0"    
        elif wr >= 46.0: return "#EF9A9A"    
        elif wr >= 43.0: return "#EF5350"    
        else: return "#D32F2F"               

    def on_recommend_clicked(self):
        if not self.controller: return
        self.reset_dashboard()
        
        is_valid, error_msg = self.validate_draft()
        if not is_valid:
            self.dash_title_lbl.setText(f"<span style='color: #EF5350;'>HIBA: {error_msg}</span>")
            return

        division_ui = self.rank_combo.currentText()
        division_backend = "MIXED" if division_ui == "ALL RANKS" else division_ui
        
        blue_team = self.get_team_ids(self.blue_combos)
        red_team = self.get_team_ids(self.red_combos)
        picked, banned = self.get_all_unavailable_ids()
        all_bans = list(banned)
        
        picking_team = "blue" if self.team_btn_group.checkedId() == 0 else "red"
        picking_index = self.role_btn_group.checkedId()
        
        current_id = blue_team[picking_index] if picking_team == "blue" else red_team[picking_index]
        if current_id > 0:
            self.dash_title_lbl.setText("<span style='color: #FFCA28;'>Erre a pozícióra már választottál hőst!</span>")
            return
        
        allowed_champs = None
        if self.meta_checkbox.isChecked() and self.meta_champs:
            lookup_div = division_backend if division_backend in self.meta_champs else "MIXED"
            role_str = str(picking_index)
            if lookup_div in self.meta_champs and role_str in self.meta_champs[lookup_div]:
                allowed_champs = set(self.meta_champs[lookup_div][role_str])
        
        self.dash_title_lbl.setText("Szimuláció futtatása...")
        QApplication.instance().processEvents()
        
        recommendations = self.controller.recommend_champions(
            division_backend, blue_team, red_team, bans=all_bans, 
            picking_team=picking_team, picking_index=picking_index, 
            top_k=5, allowed_champs=allowed_champs
        )
        
        id_to_name = {v: k for k, v in self.name_to_id.items()}
        csapat_nev = "Kék" if picking_team == "blue" else "Piros"
        role_nev = self.roles_list[picking_index]
        
        role_file_map = {"Mid": "Middle", "ADC": "Bottom"}
        role_icon_name = role_file_map.get(role_nev, role_nev)
        role_icon_path = os.path.abspath(f"assets/roles/{role_icon_name}_icon.png").replace("\\", "/")
        
        role_img_tag = f'<img src="file:///{role_icon_path}" width="28" height="28" style="vertical-align: middle; margin-left: 8px;">' if os.path.exists(os.path.abspath(f"assets/roles/{role_icon_name}_icon.png")) else ''
        
        self.dash_title_lbl.setText(f"<div style='text-align: center;'><span style='color: #FFFFFF; font-size: 20px; font-weight: bold;'>Ajánlott hősök: {csapat_nev} - {role_nev} {role_img_tag}</span></div>")
        
        if not recommendations:
            self.dash_rec_lbl.setText("<h3 style='color: #EF5350;'>Nem található megfelelő hős!</h3>")
        else:
            res_text = "<table width='100%' cellspacing='0' cellpadding='8' align='center' style='margin-top: 10px;'>"
            
            fade_colors = ["#323242", "#2C2C3A", "#262632", "#20202A", "#1A1A22"]
            
            for i, rec in enumerate(recommendations):
                cid = rec['champion_id']
                wr = rec['expected_winrate']
                champ_name = id_to_name.get(cid, "Ismeretlen")
                wr_color = self.get_winrate_color(wr)
                
                icon_path = os.path.abspath(f"assets/icons/{cid}.png").replace("\\", "/")
                img_tag = f'<img src="file:///{icon_path}" width="38" height="38" style="border-radius: 6px;">' if os.path.exists(os.path.abspath(f"assets/icons/{cid}.png")) else ''
                
                bg_color = fade_colors[i]
                
                res_text += f"<tr bgcolor='{bg_color}'>"
                res_text += f"<td style='width: 45px;'>{img_tag}</td>"
                res_text += f"<td style='font-size: 18px; font-weight: bold; color: #FFFFFF; text-align: left;'>{champ_name}</td>"
                res_text += f"<td style='font-size: 18px; color: {wr_color}; font-weight: bold; text-align: right;'>{wr:.2f}%</td>"
                res_text += f"</tr>"
                
            res_text += "</table>"
            
            self.dash_rec_lbl.setText(res_text)
            self.dash_rec_lbl.show()
            
    def show_help_dialog(self):
        dialog = HelpDialog(self)
        dialog.exec()
            
class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Súgó")
        self.setFixedSize(650, 600)
        
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #14141A, stop:1 #262633);
            }
            QTextEdit {
                background-color: #1E1E24;
                color: #E0E0E0;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 15px;
                font-size: 15px;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton.toggleBtn {
                background-color: #2D2D36;
                color: #A0A0B5;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton.toggleBtn:checked {
                background-color: #3949AB;
                color: white;
                border: 1px solid #5C6BC0;
            }
        """)
        
        layout = QVBoxLayout(self)
        

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.btn_group = QButtonGroup(self)
        
        self.btn_lol = QPushButton("Mi is az a LoL?")
        self.btn_lol.setCheckable(True)
        self.btn_lol.setChecked(True)
        self.btn_lol.setFixedSize(200, 40)
        self.btn_lol.setProperty("class", "toggleBtn")
        
        self.btn_prog = QPushButton("E program használata")
        self.btn_prog.setCheckable(True)
        self.btn_prog.setFixedSize(200, 40)
        self.btn_prog.setProperty("class", "toggleBtn")
        
        self.btn_group.addButton(self.btn_lol, 0)
        self.btn_group.addButton(self.btn_prog, 1)
        
        btn_layout.addWidget(self.btn_lol)
        btn_layout.addWidget(self.btn_prog)
        layout.addLayout(btn_layout)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        close_btn = QPushButton("Visszatérés")
        close_btn.setFixedSize(120, 35)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #E53935; color: white; font-weight: bold; border-radius: 6px;
            }
            QPushButton:hover { background-color: #C62828; }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        self.btn_group.idClicked.connect(self.load_text)
        
        self.load_text(0)

    def load_text(self, btn_id):
        file_path = "data/lol_help.txt" if btn_id == 0 else "data/predictor_help.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.text_area.setMarkdown(content)
        except FileNotFoundError:
            self.text_area.setText(f"Hiba: A '{file_path}' fájl nem található a mappában!")