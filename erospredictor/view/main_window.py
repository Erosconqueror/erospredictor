from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QApplication, 
                             QCheckBox, QGroupBox, QProgressBar, QFrame, 
                             QButtonGroup, QDialog, QTextEdit)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from configs import TARGET_TIERS

class MainWindow(QMainWindow):
    """Main graphical user interface for the application."""

    def __init__(self, champ_names: list, name_map: dict): 
        super().__init__()
        
        self.champ_names = champ_names
        self.name_map = name_map 
        self.controller = None
        self.meta_data = {}
        self.roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
        
        self._setup_window()
        self._apply_style()
        self._build_ui()

    def _setup_window(self):
        """Configures basic window properties."""
        self.setWindowTitle("Erospredictor")
        self.setFixedSize(1280, 880)
        self.setWindowIcon(QIcon("assets/mainicon.png"))
        
        self.main_widget = QWidget()
        self.main_widget.setObjectName("centralwidget")
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

    def _apply_style(self):
        """Applies the main CSS stylesheet to the application."""
        self.setStyleSheet("""
            * { font-family: 'Segoe UI', sans-serif; }
            QWidget#centralwidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #14141A, stop:1 #262633); }
            QLabel { color: #E0E0E0; }
            QComboBox { background-color: #2D2D36; border: 1px solid #444; border-radius: 6px; padding: 4px; color: white; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #1A1A22; color: white; selection-background-color: #3949AB; border-radius: 4px; width: 0px; }
            QComboBox[isBan="true"] { border: 2px solid #E53935; background-color: #2A1D20; }
            QGroupBox { border: 2px solid #3A3A45; border-radius: 8px; margin-top: 15px; padding-top: 15px; font-weight: bold; color: #A0A0B5; background-color: transparent; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QGroupBox#bansGroup { border: none; }
            QCheckBox { color: #E0E0E0; font-size: 14px; }
            QProgressBar { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EF5350, stop:1 #C62828); border-radius: 12px; border: 2px solid #BDBDBD; color: transparent; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1976D2, stop:1 #42A5F5); border-top-left-radius: 10px; border-bottom-left-radius: 10px; }
            QFrame#centerDashboard { background-color: rgba(30, 30, 40, 200); border: 1px solid #444; border-radius: 12px; }
            QPushButton.toggleBtn { background-color: #2D2D36; color: #A0A0B5; border: 1px solid #444; border-radius: 6px; padding: 8px; font-weight: bold; text-align: left; padding-left: 10px; }
            QPushButton.toggleBtn:checked { background-color: #3949AB; color: white; border: 1px solid #5C6BC0; }
            QPushButton.teamBlueBtn:checked { background-color: #1E88E5; border-color: #42A5F5; color: white;}
            QPushButton.teamRedBtn:checked { background-color: #E53935; border-color: #EF5350; color: white;}
            QPushButton.teamBlueBtn, QPushButton.teamRedBtn { text-align: center; padding-left: 0; }
        """)

    def _build_ui(self):
        """Constructs the UI components and adds them to the main layout."""
        self.main_layout.addWidget(self._create_header())
        self.main_layout.addLayout(self._create_rank_selector())
        self.main_layout.addWidget(self._create_bans())
        self.main_layout.addLayout(self._create_teams())
        self.main_layout.addWidget(self._create_controls())

    def _create_header(self) -> QWidget:
        """Builds the header widget containing title and help button."""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        
        spacer = QWidget()
        spacer.setFixedWidth(270)
        layout.addWidget(spacer)
        
        title = QLabel("League of Legends Hősválasztási Segédprogram")
        title.setStyleSheet("font-size: 28px; font-weight: 900; letter-spacing: 1px; margin: 10px; color: #FFFFFF;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        btn_help = QPushButton("Súgó")
        btn_help.setFixedSize(80, 35)
        btn_help.setStyleSheet("QPushButton { background-color: #3A3A45; font-weight: bold; border-radius: 6px; font-size: 14px; color: #F0F0F0;} QPushButton:hover { background-color: #4A4A55; }")
        btn_help.clicked.connect(self._show_help)
        layout.addWidget(btn_help, alignment=Qt.AlignmentFlag.AlignRight)
        
        return header

    def _create_rank_selector(self) -> QHBoxLayout:
        """Builds the division selection layout."""
        layout = QHBoxLayout()
        
        lbl_rank = QLabel("Rang Divízió")
        lbl_rank.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.cb_rank = QComboBox()
        self.cb_rank.setMinimumWidth(220) 
        self.cb_rank.setIconSize(QSize(24, 24))
        self.cb_rank.setStyleSheet("font-size: 15px; padding: 5px; height: 35px;")
        
        self.cb_rank.addItem(QIcon("assets/ranks/All Ranks.png"), "ALL RANKS")
        for tier in TARGET_TIERS:
            self.cb_rank.addItem(QIcon(f"assets/ranks/{tier.capitalize()}.png"), tier)
                
        layout.addWidget(lbl_rank)
        layout.addWidget(self.cb_rank)
        layout.addStretch()
        
        return layout

    def _create_bans(self) -> QGroupBox:
        """Builds the ban selection area."""
        group = QGroupBox("Kitiltott hősök (Bans)")
        group.setObjectName("bansGroup")
        layout = QVBoxLayout(group)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        
        self.blue_bans = []
        blue_layout = QHBoxLayout()
        blue_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        
        lbl_blue = QLabel("Kék tiltások:")
        lbl_blue.setFixedWidth(90) 
        lbl_blue.setStyleSheet("color: #42A5F5; font-weight: bold;") 
        blue_layout.addWidget(lbl_blue)
        
        for _ in range(5):
            combo = self._create_combo(is_ban=True)
            self.blue_bans.append(combo)
            blue_layout.addWidget(combo)
            
        dummy_blue = QLabel("") 
        dummy_blue.setFixedWidth(90)
        blue_layout.addWidget(dummy_blue)
            
        self.red_bans = []
        red_layout = QHBoxLayout()
        red_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        lbl_red = QLabel("Piros tiltások:")
        lbl_red.setFixedWidth(90) 
        lbl_red.setStyleSheet("color: #EF5350; font-weight: bold;")
        red_layout.addWidget(lbl_red)
        
        for _ in range(5):
            combo = self._create_combo(is_ban=True)
            self.red_bans.append(combo)
            red_layout.addWidget(combo)
            
        dummy_red = QLabel("") 
        dummy_red.setFixedWidth(90)
        red_layout.addWidget(dummy_red)
            
        layout.addLayout(blue_layout)
        layout.addLayout(red_layout)
        
        return group

    def _create_teams(self) -> QHBoxLayout:
        """Builds the main 3-column layout for teams and dashboard."""
        layout = QHBoxLayout()
        
        blue_col = QVBoxLayout()
        lbl_blue = QLabel("Kék Csapat (Blue Team)")
        lbl_blue.setStyleSheet("color: #42A5F5; font-weight: bold; font-size: 18px; padding-bottom: 5px;")
        blue_col.addWidget(lbl_blue, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.blue_picks = []
        for role in self.roles:
            row = QHBoxLayout()
            lbl = QLabel(f"{role}:")
            lbl.setFixedWidth(65)
            lbl.setMinimumHeight(50) 
            lbl.setStyleSheet("font-weight: bold; color: #9E9E9E; font-size: 15px;")
            combo = self._create_combo()
            self.blue_picks.append(combo)
            row.addWidget(lbl)
            row.addWidget(combo)
            blue_col.addLayout(row)
            
        layout.addLayout(blue_col, stretch=1)
        
        self.dashboard = QFrame()
        self.dashboard.setObjectName("centerDashboard")
        self.dashboard.setFixedWidth(460)
        
        dash_layout = QVBoxLayout(self.dashboard)
        dash_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        dash_layout.setContentsMargins(1, 15, 1, 0)
        
        self.lbl_dash_title = QLabel("Válassz hősöket a draftoláshoz!")
        self.lbl_dash_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        self.lbl_dash_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash_layout.addWidget(self.lbl_dash_title)
        
        self.wr_container = QWidget()
        wr_layout = QHBoxLayout(self.wr_container)
        wr_layout.setContentsMargins(10, 80, 10, 5)
        
        self.lbl_blue_wr = QLabel("Kék: 50.0%")
        self.lbl_blue_wr.setStyleSheet("color: #64B5F6; font-weight: bold; font-size: 20px;")
        
        self.lbl_red_wr = QLabel("Piros: 50.0%")
        self.lbl_red_wr.setStyleSheet("color: #E57373; font-weight: bold; font-size: 20px;")
        self.lbl_red_wr.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        wr_layout.addWidget(self.lbl_blue_wr)
        wr_layout.addWidget(self.lbl_red_wr)
        dash_layout.addWidget(self.wr_container)
        self.wr_container.hide()
        
        self.bar_wr = QProgressBar()
        self.bar_wr.setFixedHeight(28)
        self.bar_wr.setRange(0, 100)
        dash_layout.addWidget(self.bar_wr)
        self.bar_wr.hide()
        
        self.lbl_recommend = QLabel("")
        self.lbl_recommend.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        dash_layout.addWidget(self.lbl_recommend)
        self.lbl_recommend.hide()
        
        dash_layout.addStretch()
        layout.addWidget(self.dashboard)
        
        red_col = QVBoxLayout()
        lbl_red = QLabel("Piros Csapat (Red Team)")
        lbl_red.setStyleSheet("color: #EF5350; font-weight: bold; font-size: 18px; padding-bottom: 5px;")
        red_col.addWidget(lbl_red, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.red_picks = []
        for role in self.roles:
            row = QHBoxLayout()
            lbl = QLabel(f"{role}:")
            lbl.setFixedWidth(65)
            lbl.setMinimumHeight(50) 
            lbl.setStyleSheet("font-weight: bold; color: #9E9E9E; font-size: 15px;")
            combo = self._create_combo()
            self.red_picks.append(combo)
            row.addWidget(lbl)
            row.addWidget(combo)
            red_col.addLayout(row)
            
        layout.addLayout(red_col, stretch=1)
        
        return layout

    def _create_controls(self) -> QWidget:
        """Builds the bottom control panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.setSpacing(15) 
        
        self.btn_predict = QPushButton("PREDIKCIÓ")
        self.btn_predict.setFixedSize(300, 55) 
        self.btn_predict.setStyleSheet("QPushButton { font-size: 18px; font-weight: bold; letter-spacing: 1px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #43A047, stop:1 #2E7D32); color: white; border-radius: 8px; } QPushButton:hover { background: #388E3C; }")
        self.btn_predict.clicked.connect(self._on_predict)
        layout.addWidget(self.btn_predict, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        group_rec = QGroupBox("Hős Ajánló")
        group_rec.setFixedWidth(700) 
        rec_layout = QVBoxLayout(group_rec)
        
        row_team = QHBoxLayout()
        self.grp_team = QButtonGroup(self)
        
        self.btn_blue = QPushButton("Kék Csapat (Blue)")
        self.btn_blue.setCheckable(True)
        self.btn_blue.setChecked(True)
        self.btn_blue.setFixedSize(160, 35)
        self.btn_blue.setProperty("class", "toggleBtn teamBlueBtn")
        
        self.btn_red = QPushButton("Piros Csapat (Red)")
        self.btn_red.setCheckable(True)
        self.btn_red.setFixedSize(160, 35)
        self.btn_red.setProperty("class", "toggleBtn teamRedBtn")
        
        self.grp_team.addButton(self.btn_blue, 0)
        self.grp_team.addButton(self.btn_red, 1)
        row_team.addWidget(self.btn_blue)
        row_team.addWidget(self.btn_red)
        rec_layout.addLayout(row_team)
        
        row_roles = QHBoxLayout()
        self.grp_role = QButtonGroup(self)
        
        role_map = {"Mid": "Middle", "ADC": "Bottom"}
        for i, role in enumerate(self.roles):
            btn = QPushButton(f" {role}")
            btn.setCheckable(True)
            btn.setFixedSize(115, 38)
            btn.setProperty("class", "toggleBtn")
            
            icon_name = role_map.get(role, role)
            btn.setIcon(QIcon(f"assets/roles/{icon_name}_icon.png"))
            btn.setIconSize(QSize(20, 20))
                
            if i == 0: btn.setChecked(True) 
            self.grp_role.addButton(btn, i)
            row_roles.addWidget(btn)
            
        rec_layout.addLayout(row_roles)
        
        row_action = QHBoxLayout()
        self.chk_meta = QCheckBox("Off-Meta hősök szűrése")
        self.chk_meta.setChecked(True)
        
        self.btn_rec = QPushButton("AJÁNLÁS")
        self.btn_rec.setFixedSize(250, 40)
        self.btn_rec.setStyleSheet("QPushButton { font-weight: bold; font-size: 14px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E88E5, stop:1 #1565C0); color: white; border-radius: 6px; } QPushButton:hover { background: #1976D2; }")
        self.btn_rec.clicked.connect(self._on_recommend)
        
        row_action.addWidget(self.chk_meta)
        row_action.addWidget(self.btn_rec)
        rec_layout.addLayout(row_action)
        
        layout.addWidget(group_rec, alignment=Qt.AlignmentFlag.AlignHCenter)
        return container

    def _create_combo(self, is_ban: bool = False) -> QComboBox:
        """Helper to create a stylized champion selection dropdown."""
        combo = QComboBox()
        width = 130 if is_ban else 150
        height = 35 if is_ban else 50
        icon_size = 24 if is_ban else 42
        
        combo.setFixedWidth(width)   
        combo.setMinimumHeight(height) 
        combo.setIconSize(QSize(icon_size, icon_size)) 
        combo.setStyleSheet("font-size: 14px; padding: 3px;")
        
        if is_ban:
            combo.setProperty("isBan", True)
            
        combo.addItem(QIcon(), "") 
        for name in self.champ_names:
            cid = self.name_map.get(name)
            combo.addItem(QIcon(f"assets/icons/{cid}.png"), name)
                
        combo.setEditable(True)
        combo.lineEdit().setPlaceholderText("Gépelj...") 
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.completer().setCompletionMode(combo.completer().CompletionMode.PopupCompletion)
        combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        combo.editTextChanged.connect(lambda t, cb=combo: cb.setCurrentIndex(0) if not t.strip() else None)
        
        return combo

    def set_controller_and_mapping(self, controller, name_map: dict):
        """Injects the controller dependency."""
        self.controller = controller

    def set_meta_data(self, meta_data: dict):
        """Sets the meta champion configuration."""
        self.meta_data = meta_data

    def _get_ids(self, combos: list) -> list:
        """Extracts champion IDs from a list of comboboxes."""
        ids = []
        for cb in combos:
            name = cb.currentText().strip()
            ids.append(self.name_map.get(name, 0))
        return ids

    def _get_unavailable(self) -> set:
        """Returns a set of already picked or banned champion IDs."""
        picks = [i for i in self._get_ids(self.blue_picks) + self._get_ids(self.red_picks) if i > 0]
        bans = [i for i in self._get_ids(self.blue_bans) + self._get_ids(self.red_bans) if i > 0]
        return set(picks).union(set(bans))

    def _reset_dash(self):
        """Hides results and readies dashboard for new calculation."""
        self.lbl_dash_title.show()
        self.wr_container.hide()
        self.bar_wr.hide()
        self.lbl_recommend.hide()

    def _validate(self) -> tuple:
        """Validates the current draft state."""
        b_picks = [i for i in self._get_ids(self.blue_picks) if i > 0]
        r_picks = [i for i in self._get_ids(self.red_picks) if i > 0]
        b_bans = [i for i in self._get_ids(self.blue_bans) if i > 0]
        r_bans = [i for i in self._get_ids(self.red_bans) if i > 0]

        if len(b_bans) != len(set(b_bans)): return False, "A Kék csapat nem bannolhatja ugyanazt kétszer!"
        if len(r_bans) != len(set(r_bans)): return False, "A Piros csapat nem bannolhatja ugyanazt kétszer!"

        picks = b_picks + r_picks
        if len(picks) != len(set(picks)): return False, "Egy hős csak egyszer lehet kiválasztva!"

        all_bans = set(b_bans + r_bans)
        for pick in picks:
            if pick in all_bans: return False, "Kiválasztott hős nem lehet a tiltólistán!"

        return True, ""

    def _color_wr(self, wr: float) -> str:
        """Returns a hex color string based on winrate percentage."""
        if wr >= 57.0: return "#00E676"      
        elif wr >= 54.0: return "#66BB6A"    
        elif wr >= 51.5: return "#A5D6A7"    
        elif wr >= 48.5: return "#E0E0E0"    
        elif wr >= 46.0: return "#EF9A9A"    
        elif wr >= 43.0: return "#EF5350"    
        return "#D32F2F"               

    def _on_predict(self):
        """Handles predict button click."""
        if not self.controller: return
        self._reset_dash()
        
        valid, err = self._validate()
        if not valid:
            self.lbl_dash_title.setText(f"<span style='color: #EF5350;'>HIBA: {err}</span>")
            return

        blue = self._get_ids(self.blue_picks)
        red = self._get_ids(self.red_picks)
        div_ui = self.cb_rank.currentText()
        div = "MIXED" if div_ui == "ALL RANKS" else div_ui
        
        self.lbl_dash_title.setText("Számítás folyamatban...")
        QApplication.instance().processEvents() 
        
        res = self.controller.predict_match(div, blue, red)
        b_wr, r_wr = res['blue_win_prob'], res['red_win_prob']
        
        self.lbl_dash_title.setText(f"<span style='color: #FFFFFF; font-size: 20px;'>Prediktált Győzelmi Arány ({div_ui})</span>")
        self.lbl_blue_wr.setText(f"Kék: {b_wr:.1f}%")
        self.lbl_red_wr.setText(f"Piros: {r_wr:.1f}%")
        self.bar_wr.setValue(int(b_wr)) 
        
        self.wr_container.show()
        self.bar_wr.show()

    def _on_recommend(self):
        """Handles recommend button click."""
        if not self.controller: return
        self._reset_dash()
        
        valid, err = self._validate()
        if not valid:
            self.lbl_dash_title.setText(f"<span style='color: #EF5350;'>HIBA: {err}</span>")
            return

        div_ui = self.cb_rank.currentText()
        div = "MIXED" if div_ui == "ALL RANKS" else div_ui
        
        blue = self._get_ids(self.blue_picks)
        red = self._get_ids(self.red_picks)
        bans = list(self._get_unavailable())
        
        t_id = self.grp_team.checkedId()
        r_id = self.grp_role.checkedId()
        picking = "blue" if t_id == 0 else "red"
        
        if (blue[r_id] if picking == "blue" else red[r_id]) > 0:
            self.lbl_dash_title.setText("<span style='color: #FFCA28;'>Erre a pozícióra már választottál hőst!</span>")
            return
        
        allowed = None
        if self.chk_meta.isChecked() and self.meta_data:
            lookup = div if div in self.meta_data else "MIXED"
            if lookup in self.meta_data and str(r_id) in self.meta_data[lookup]:
                allowed = set(self.meta_data[lookup][str(r_id)])
        
        self.lbl_dash_title.setText("Szimuláció futtatása...")
        QApplication.instance().processEvents()
        
        recs = self.controller.recommend_champs(div, blue, red, bans, picking, r_id, 5, allowed)
        
        id_name = {v: k for k, v in self.name_map.items()}
        t_name = "Kék" if picking == "blue" else "Piros"
        r_name = self.roles[r_id]
        icon_name = {"Mid": "Middle", "ADC": "Bottom"}.get(r_name, r_name)
        
        img = f"<img src='assets/roles/{icon_name}_icon.png' width='28' height='28' style='vertical-align: middle; margin-left: 8px;'>"
        self.lbl_dash_title.setText(f"<div style='text-align: center;'><span style='color: #FFFFFF; font-size: 20px; font-weight: bold;'>Ajánlott hősök: {t_name} - {r_name} {img}</span></div>")
        
        if not recs:
            self.lbl_recommend.setText("<h3 style='color: #EF5350;'>Nem található megfelelő hős!</h3>")
            self.lbl_recommend.show()
            return
            
        html = "<table width='100%' cellspacing='0' cellpadding='8' align='center' style='margin-top: 10px;'>"
        colors = ["#323242", "#2C2C3A", "#262632", "#20202A", "#1A1A22"]
        
        for i, rec in enumerate(recs):
            cid = rec['id']
            wr = rec['wr']
            name = id_name.get(cid, "Unknown")
            c_wr = self._color_wr(wr)
            
            html += f"<tr bgcolor='{colors[i]}'>"
            html += f"<td style='width: 45px;'><img src='assets/icons/{cid}.png' width='38' height='38' style='border-radius: 6px;'></td>"
            html += f"<td style='font-size: 18px; font-weight: bold; color: #FFFFFF; text-align: left;'>{name}</td>"
            html += f"<td style='font-size: 18px; color: {c_wr}; font-weight: bold; text-align: right;'>{wr:.2f}%</td>"
            html += "</tr>"
            
        html += "</table>"
        self.lbl_recommend.setText(html)
        self.lbl_recommend.show()
            
    def _show_help(self):
        """Displays the help dialog."""
        HelpDialog(self).exec()
            

class HelpDialog(QDialog):
    """Dialog window displaying application instructions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Súgó")
        self.setFixedSize(650, 600)
        
        self.setStyleSheet("""
            QDialog { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #14141A, stop:1 #262633); }
            QTextEdit { background-color: #1E1E24; color: #E0E0E0; border: 1px solid #444; border-radius: 8px; padding: 15px; font-size: 15px; }
            QPushButton.toggleBtn { background-color: #2D2D36; color: #A0A0B5; border: 1px solid #444; border-radius: 6px; padding: 8px; font-weight: bold; font-size: 14px; }
            QPushButton.toggleBtn:checked { background-color: #3949AB; color: white; border: 1px solid #5C6BC0; }
        """)
        
        layout = QVBoxLayout(self)
        
        btn_layout = QHBoxLayout()
        self.grp_btn = QButtonGroup(self)
        
        self.btn_lol = QPushButton("Mi is az a LoL?")
        self.btn_lol.setCheckable(True)
        self.btn_lol.setChecked(True)
        self.btn_lol.setFixedSize(200, 40)
        self.btn_lol.setProperty("class", "toggleBtn")
        
        self.btn_app = QPushButton("E program használata")
        self.btn_app.setCheckable(True)
        self.btn_app.setFixedSize(200, 40)
        self.btn_app.setProperty("class", "toggleBtn")
        
        self.grp_btn.addButton(self.btn_lol, 0)
        self.grp_btn.addButton(self.btn_app, 1)
        
        btn_layout.addWidget(self.btn_lol)
        btn_layout.addWidget(self.btn_app)
        layout.addLayout(btn_layout)
        
        self.txt_area = QTextEdit()
        self.txt_area.setReadOnly(True)
        layout.addWidget(self.txt_area)

        btn_close = QPushButton("Visszatérés")
        btn_close.setFixedSize(120, 35)
        btn_close.setStyleSheet("QPushButton { background-color: #E53935; color: white; font-weight: bold; border-radius: 6px; } QPushButton:hover { background-color: #C62828; }")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        self.grp_btn.idClicked.connect(self._load_text)
        self._load_text(0)

    def _load_text(self, btn_id: int):
        """Loads markdown help text from file."""
        path = "data/lol_help.txt" if btn_id == 0 else "data/predictor_help.txt"
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.txt_area.setMarkdown(f.read())
        except FileNotFoundError:
            self.txt_area.setText("Hiba: A súgófájl nem található!")