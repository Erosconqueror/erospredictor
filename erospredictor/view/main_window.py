from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QApplication, 
                             QCheckBox, QGroupBox, QProgressBar, QFrame, 
                             QButtonGroup, QDialog, QTextEdit, QScrollArea, QCompleter,
                             QFileDialog, QMenu)
from PyQt6.QtCore import Qt, QSize, QThread, QTimer
from PyQt6.QtGui import QIcon, QAction, QKeySequence
from configs import TARGET_TIERS, CHAMPION_COUNT
from html import escape
from dataclasses import replace
from pathlib import Path
import json
from view.background_task import BackgroundTask
from view.draft_follower import DraftFollower
from services.client_settings import preferred_lockfile, save_lockfile_path
from services import feature_flags
from services.champion_pools import ALL_MINE, TAGS, load_library, member_ids, internal_tags
from services.ui_preferences import load_ui_preferences, save_ui_preferences

ROLE_SWAP_HINT = "Jobb kattintás a hősön vagy a pozíción: áthelyezés / csere."

class MainWindow(QMainWindow):
    """Main graphical user interface for the application."""

    def __init__(self, champ_names: list, name_map: dict): 
        super().__init__()
        
        self.champ_names = champ_names
        self.name_map = name_map 
        self.controller = None
        self.meta_data = {}
        self.roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
        self._task = None
        self._busy = False
        self._task_callback_active = False
        self._task_finish_pending = False
        self._manual_import_active = False
        self._recommendation_context = None
        self._prediction_context = None
        self._effects_context = None
        self._last_recommendations = []
        self._selected_recommendation_id = None
        self._lockfile_path = None
        self._personal_history = None
        self._history_loading = False
        self._sync_last_snapshot = None
        self._sync_pending = None
        self._sync_session_key = ""
        self._sync_manual_side = None
        self._sync_manual_roles = {}
        self._sync_applying = False
        self._role_document = None
        self._role_stamp = None
        self._sync_latest_snapshot = None
        self._pending_ui_preferences = {}
        self._preferences_suspended = False
        self._pool_library = load_library()
        try:
            self._pool_mapping = {riot: int(index) for riot, index in json.loads((Path(__file__).resolve().parents[1] / "data/champion_id.json").read_text()).items()}
        except (OSError, ValueError, TypeError):
            self._pool_mapping = {}
        
        self._setup_window()
        self._apply_style()
        self._build_ui()
        self._connect_draft_changes()
        self._draft_follower = DraftFollower(self._read_followed_draft, lambda: self._busy, parent=self)
        self._draft_follower.snapshot_ready.connect(self._receive_followed_draft)
        self._draft_follower.status_changed.connect(self._set_sync_status)
        self._draft_follower.draft_unavailable.connect(self._draft_unavailable)
        self._setup_persistent_preferences()

    def _setup_persistent_preferences(self):
        self._preferences_timer = QTimer(self)
        self._preferences_timer.setSingleShot(True)
        self._preferences_timer.setInterval(250)
        self._preferences_timer.timeout.connect(self._flush_ui_preferences)
        values = load_ui_preferences()
        self._ban_role = values.get("ban_role")
        booleans = {"auto_draft": self.chk_follow, "off_meta": self.chk_meta,
            "simple_effects": self.chk_effects, "robust_roles": self.chk_robust,
            "history_markers": self.chk_personal, "history_service": self.chk_history_service,
            "pool_only": self.chk_pool_only}
        combos = {"history_limit": (self.cb_history_limit, True), "sync_side": (self.cb_sync_side, True),
                  "active_pool": (self.cb_pool, False), "tier": (self.cb_rank, False)}
        for key, control in booleans.items():
            if key in values and (key != "history_markers" or feature_flags.PERSONAL_HISTORY_ENABLED):
                previous = control.blockSignals(True)
                control.setChecked(values[key])
                control.blockSignals(previous)
            control.toggled.connect(lambda checked, name=key: self._remember_ui_preference(name, checked))
        for key, (control, use_data) in combos.items():
            if key in values:
                index = control.findData(values[key]) if use_data else control.findText(values[key])
                if index >= 0:
                    previous = control.blockSignals(True)
                    control.setCurrentIndex(index)
                    control.blockSignals(previous)
            control.currentIndexChanged.connect(lambda _index, name=key, widget=control, data=use_data:
                self._remember_ui_preference(name, widget.currentData() if data else widget.currentText()))
        if "preferred_role" in values:
            self.grp_role.button(values["preferred_role"]).setChecked(True)
        self.grp_role.idClicked.connect(lambda role: self._remember_ui_preference("preferred_role", role))
        if self.chk_follow.isChecked():
            QTimer.singleShot(0, self._resume_saved_follow)

    def _resume_saved_follow(self):
        if self.isVisible() and not self._preferences_suspended:
            self._draft_follower.set_enabled(self.chk_follow.isChecked())

    def _remember_ui_preference(self, key, value):
        if self._preferences_suspended:
            return
        self._pending_ui_preferences[key] = value
        self._preferences_timer.start()

    def _flush_ui_preferences(self):
        self._preferences_timer.stop()
        if not self._pending_ui_preferences:
            return True
        if not save_ui_preferences(self._pending_ui_preferences):
            self.options_dialog.preferences_notice.setText("A beállítások most nem menthetők. A következő változtatáskor újrapróbáljuk.")
            self.options_dialog.preferences_notice.show()
            return False
        self._pending_ui_preferences.clear()
        self.options_dialog.preferences_notice.hide()
        return True

    def _setup_window(self):
        """Configures basic window properties."""
        self.setWindowTitle("Erospredictor")
        self.resize(1280, 940)
        self.setMinimumSize(1180, 920)
        self.setWindowIcon(QIcon("assets/mainicon.png"))
        
        self.main_widget = QWidget()
        self.main_widget.setObjectName("centralwidget")
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(4)

    def _apply_style(self):
        """Applies the main CSS stylesheet to the application."""
        self.setStyleSheet("""
            * { font-family: 'Segoe UI', sans-serif; }
            QWidget#centralwidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #14141A, stop:1 #262633); }
            QLabel { color: #E0E0E0; }
            QPushButton { background-color: #343440; color: #E0E0E0; border: 1px solid #555; border-radius: 6px; padding: 7px; }
            QPushButton:disabled { color: #777; border-color: #3A3A45; }
            QComboBox { background-color: #2D2D36; border: 1px solid #444; border-radius: 6px; padding: 4px; color: white; }
            QComboBox:disabled { background-color: #25252D; color: #777; border-color: #3A3A45; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #1A1A22; color: white; selection-background-color: #3949AB; border-radius: 4px; width: 0px; }
            QComboBox[isBan="true"] { border: 2px solid #E53935; background-color: #2A1D20; }
            QGroupBox { border: 2px solid #3A3A45; border-radius: 8px; margin-top: 15px; padding-top: 15px; font-weight: bold; color: #A0A0B5; background-color: transparent; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QGroupBox#bansGroup { border: none; margin-top: 0px; padding-top: 0px; }
            QCheckBox { color: #E0E0E0; font-size: 14px; }
            QCheckBox:disabled { color: #777; }
            QDialog { background-color: #20202A; color: #E0E0E0; }
            QTabWidget::pane { border: 1px solid #444450; background-color: #20202A; }
            QTabBar::tab { background: #2D2D36; color: #BDBDCB; padding: 10px; }
            QTabBar::tab:selected { background: #3949AB; color: white; }
            QMenu { background: #292934; color: #E0E0E0; border: 1px solid #555; }
            QMenu::item { padding: 7px 18px; }
            QMenu::item:selected { background: #3949AB; }
            QLineEdit, QTableWidget, QTextEdit { background-color: #292934; color: #E0E0E0; border: 1px solid #444450; padding: 5px; }
            QHeaderView::section { background-color: #363645; color: #E0E0E0; padding: 6px; }
            QScrollBar:vertical { background: #20202A; width: 9px; margin: 0; }
            QScrollBar::handle:vertical { background: #55556F; min-height: 20px; border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
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
        self.lbl_draft_sync = QLabel("Automatikus draftkövetés kikapcsolva.")
        self.lbl_draft_sync.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_draft_sync.setStyleSheet("font-size: 11px; color: #BDBDCB;")
        self.main_layout.addWidget(self.lbl_draft_sync)
        self.main_layout.addWidget(self._create_bans())
        self.main_layout.addLayout(self._create_teams())
        self.main_layout.addWidget(self._create_controls())

    def _create_header(self) -> QWidget:
        """Builds the header widget containing title and help button."""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 7, 0)
        
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
        lbl_rank = QLabel("Rang (Tier)")
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
        self.chk_effects = QCheckBox("Egyszerű hőshatások")
        self.chk_effects.setToolTip(
            "Opcionális számítás a lent kiválasztott csapat és pozíció hőséhez. "
            "Ajánláskor az első jelöltet vizsgálja; másik hős a nevére kattintva választható. "
            "A számok százalékpontos modellhatások.")
        self.chk_effects.toggled.connect(self._on_effects_toggled)
        self.chk_follow = QCheckBox("Automatikus draft")
        self.chk_follow.setToolTip("1,5 másodpercenként követi a helyi kliens rögzített pickjeit és tiltásait. "
                                   "A becsült szerepkörök a hősök adatbázisbeli eloszlásából származnak.")
        self.chk_follow.toggled.connect(self._on_follow_toggled)
        layout.addWidget(self.chk_follow)
        layout.addStretch()
        self.btn_drafts = QPushButton("Draftok")
        menu = QMenu(self.btn_drafts)
        self.action_save_draft = QAction("Draft mentése…", self)
        self.action_save_draft.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save_draft.triggered.connect(self._on_save_draft)
        self.action_load_draft = QAction("Draft betöltése…", self)
        self.action_load_draft.setShortcut(QKeySequence.StandardKey.Open)
        self.action_load_draft.triggered.connect(self._on_load_draft)
        for action in (self.action_save_draft, self.action_load_draft):
            menu.addAction(action)
            self.addAction(action)
        self.btn_drafts.setMenu(menu)
        layout.addWidget(self.btn_drafts)
        self.btn_import = QPushButton("LoL-kliens beolvasása")
        self.btn_import.setToolTip("A már kiválasztott hősök és tiltások beolvasása a saját kliensből.")
        self.btn_import.clicked.connect(self._on_import_draft)
        layout.addWidget(self.btn_import)
        self.btn_lockfile = QPushButton("LoL lockfile…")
        self.btn_lockfile.setToolTip(f"A mentett vagy alapértelmezett LoL lockfile helye: {preferred_lockfile()}")
        self.btn_lockfile.clicked.connect(self._choose_lockfile)
        return layout

    def _create_bans(self) -> QGroupBox:
        """Builds the ban selection area."""
        group = QGroupBox()
        group.setObjectName("bansGroup")
        layout = QVBoxLayout(group)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter) 
        
        lbl_title = QLabel("Kitiltott hősök (Bans)")
        lbl_title.setStyleSheet("color: #A0A0B5; font-weight: bold; font-size: 15px;")
        layout.addWidget(lbl_title, alignment=Qt.AlignmentFlag.AlignHCenter)
        
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
        self.blue_effects = []
        self.blue_role_labels = []
        for role in self.roles:
            row = QHBoxLayout()
            lbl = QLabel(f"{role}:")
            lbl.setFixedWidth(65)
            lbl.setMinimumHeight(50) 
            lbl.setStyleSheet("font-weight: bold; color: #9E9E9E; font-size: 15px;")
            self.blue_role_labels.append(lbl)
            combo = self._create_combo()
            self.blue_picks.append(combo)
            badge = self._create_effect_badge()
            self.blue_effects.append(badge)
            row.addWidget(badge)
            row.addWidget(lbl)
            row.addWidget(combo)
            blue_col.addLayout(row)
        layout.addLayout(blue_col, stretch=1)
        
        self.dashboard = QFrame()
        self.dashboard.setObjectName("centerDashboard")
        self.dashboard.setFixedWidth(460)
        dash_layout = QVBoxLayout(self.dashboard)
        dash_layout.setContentsMargins(1, 15, 1, 5) 
        
        self.lbl_dash_title = QLabel("Válassz hősöket a draftoláshoz!")
        self.lbl_dash_title.setWordWrap(True)
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
        
        self.scroll_recommend = QScrollArea()
        self.scroll_recommend.setWidgetResizable(True)
        self.scroll_recommend.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_recommend.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_recommend.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_recommend.setStyleSheet("background: transparent;")
        
        self.lbl_recommend = QLabel("")
        self.lbl_recommend.setWordWrap(True)
        self.lbl_recommend.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.lbl_recommend.setStyleSheet("background: transparent;")
        self.lbl_recommend.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.lbl_recommend.setOpenExternalLinks(False)
        self.lbl_recommend.linkActivated.connect(self._on_recommendation_selected)
        self.lbl_recommend.setToolTip("A hős nevére kattintva az egyszerű hőshatások fókusza választható ki.")
        self.scroll_recommend.setWidget(self.lbl_recommend)
        self.scroll_recommend.hide()
        
        dash_layout.addWidget(self.scroll_recommend, 100) 
        dash_layout.addStretch(1) 
        layout.addWidget(self.dashboard)
        
        red_col = QVBoxLayout()
        lbl_red = QLabel("Piros Csapat (Red Team)")
        lbl_red.setStyleSheet("color: #EF5350; font-weight: bold; font-size: 18px; padding-bottom: 5px;")
        red_col.addWidget(lbl_red, alignment=Qt.AlignmentFlag.AlignCenter)
        self.red_picks = []
        self.red_effects = []
        self.red_role_labels = []
        for role in self.roles:
            row = QHBoxLayout()
            lbl = QLabel(f"{role}:")
            lbl.setFixedWidth(65)
            lbl.setMinimumHeight(50) 
            lbl.setStyleSheet("font-weight: bold; color: #9E9E9E; font-size: 15px;")
            self.red_role_labels.append(lbl)
            combo = self._create_combo()
            self.red_picks.append(combo)
            row.addWidget(lbl)
            row.addWidget(combo)
            badge = self._create_effect_badge()
            self.red_effects.append(badge)
            row.addWidget(badge)
            red_col.addLayout(row)
        layout.addLayout(red_col, stretch=1)
        
        return layout

    def _create_effect_badge(self):
        badge = QLabel("")
        badge.setFixedWidth(92)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet("font-size: 13px; font-weight: bold; border: 1px solid transparent;")
        return badge

    def _create_controls(self) -> QWidget:
        """Builds the bottom control panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.setSpacing(12)
        
        self.btn_predict = QPushButton("PREDIKCIÓ")
        self.btn_predict.setFixedSize(300, 55) 
        self.btn_predict.setStyleSheet("QPushButton { font-size: 18px; font-weight: bold; letter-spacing: 1px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #43A047, stop:1 #2E7D32); color: white; border-radius: 8px; } QPushButton:hover { background: #388E3C; }")
        self.btn_predict.clicked.connect(self._on_predict)
        layout.addWidget(self.btn_predict, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        bottom_area = QHBoxLayout()
        bottom_area.addStretch(1)
        
        group_rec = QGroupBox("Hős Ajánló")
        group_rec.setFixedWidth(700) 
        rec_layout = QVBoxLayout(group_rec)
        rec_layout.setSpacing(4)
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

        self.chk_personal = QCheckBox("Játszott hősök jelölése az előzményből")
        self.chk_personal.setChecked(feature_flags.PERSONAL_HISTORY_ENABLED)
        self.chk_personal.setEnabled(False)
        self.chk_personal.setToolTip(
            "Kék keret: legalább 3 saját meccs. Arany keret: legalább 20 meccs és 55% saját győzelmi arány. "
            "A százalék és a sorrend a modell ajánlása marad; saját matchupok és szinergiák a részletes elemzésben.")
        self.chk_personal.toggled.connect(self._on_personal_toggled)
        self.lbl_history = QLabel("Előzmények betöltése indításkor…" if feature_flags.PERSONAL_HISTORY_ENABLED else "Match history szüneteltetve")
        self.lbl_history.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_history.setStyleSheet("font-size: 11px; color: #BDBDCB;")
        self.lbl_history.setMaximumWidth(240)
        self.cb_history_limit = QComboBox()
        for count in (100, 500, 1000):
            self.cb_history_limit.addItem(str(count), count)
        self.cb_history_limit.setToolTip("Ennyi legutóbbi meccset vizsgál. A már mentett részleteket újra felhasználja.")
        self.btn_history = QPushButton("Előzmények frissítése")
        self.btn_history.clicked.connect(self._on_refresh_history)
        self.btn_history.setEnabled(feature_flags.PERSONAL_HISTORY_ENABLED)
        self.cb_history_limit.setEnabled(feature_flags.PERSONAL_HISTORY_ENABLED)
        if not feature_flags.PERSONAL_HISTORY_ENABLED:
            self.btn_history.setToolTip("A személyes meccselőzmény funkció jelenleg le van tiltva.")
        self.chk_history_service = QCheckBox("Régebbi meccsek (Riot)")
        self.chk_history_service.setChecked(False)
        self.chk_history_service.setEnabled(feature_flags.PERSONAL_HISTORY_ENABLED)
        self.chk_history_service.setToolTip(
            "Külön engedélyezhető lekérés a Riot saját előzményszerveréről, a LoL-kliens bejelentkezésével. "
            "A kliens hitelesítési tokenje HTTPS-en a kliens konfigurációjában megadott Riot-szerverhez kerül; "
            "csak memóriában használjuk, nem mentjük el. A fejlesztői Riot API-kulcsot nem használja. "
            "Bekapcsolás után az Előzmények frissítése indítja a lekérést.")
        
        row_action = QHBoxLayout()
        self.chk_meta = QCheckBox("Off-Meta hősök szűrése")
        self.chk_meta.setChecked(True)
        self.chk_robust = QCheckBox("Bizonytalan ellenfélpozíciók vizsgálata (első 5)")
        self.chk_robust.setToolTip("Opcionális extra számítás: legfeljebb három ütközésmentes ellenfél-kiosztás. Az automatikus pozícióhoz legalább 5% gyakoriság kell; teljes ellenfélcsapatnál a kényszerített elhelyezést jelöljük.")
        self.cb_sync_side = QComboBox()
        for title, value in (("Automatikus", None), ("Kék", "blue"), ("Piros", "red")):
            self.cb_sync_side.addItem(title, value)
        self.cb_sync_side.currentIndexChanged.connect(self._on_sync_side_changed)
        pool_row = QHBoxLayout()
        pool_row.addWidget(QLabel("Hőspool:"))
        self.cb_pool = QComboBox()
        self.cb_pool.addItems([ALL_MINE] + sorted(self._pool_library["lists"]))
        pool_row.addWidget(self.cb_pool, 1)
        self.chk_pool_only = QCheckBox("Csak ebből ajánlj")
        pool_row.addWidget(self.chk_pool_only)
        self.btn_pool = QPushButton("Hőseim és listák…")
        self.btn_pool.clicked.connect(self._edit_pool)
        pool_row.addWidget(self.btn_pool)
        rec_layout.addLayout(pool_row)
        self.btn_rec = QPushButton("AJÁNLÁS")
        self.btn_rec.setFixedSize(250, 40)
        self.btn_rec.setStyleSheet("QPushButton { font-weight: bold; font-size: 14px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E88E5, stop:1 #1565C0); color: white; border-radius: 6px; } QPushButton:hover { background: #1976D2; }")
        self.btn_rec.clicked.connect(self._on_recommend)
        row_action.addWidget(self.btn_rec)
        self.btn_bans = QPushButton("Bansegéd")
        self.btn_bans.setToolTip("A választott hőspoolhoz és pozícióhoz ad banötleteket a mentett statisztikákból, neurális számítás nélkül.")
        self.btn_bans.clicked.connect(self._on_ban_advice)
        row_action.addWidget(self.btn_bans)
        self.btn_options = QPushButton("Beállítások / előzmények…")
        self.btn_options.clicked.connect(self._show_options)
        row_action.addWidget(self.btn_options)
        rec_layout.addLayout(row_action)

        self.btn_explain = QPushButton("Miért? – Részletes elemzés")
        self.btn_explain.setEnabled(False)
        self.btn_explain.setToolTip("Predikció után a fókuszhőst, ajánlás után a kijelölt hőst és az első öt jelöltet elemzi.")
        self.btn_explain.clicked.connect(self._on_explain)
        rec_layout.addWidget(self.btn_explain)
        self.lbl_effects_hint = QLabel("")
        self.lbl_effects_hint.setTextFormat(Qt.TextFormat.PlainText)
        self.lbl_effects_hint.setWordWrap(True)
        self.lbl_effects_hint.setStyleSheet("font-size: 12px; color: #BDBDCB;")
        self.lbl_effects_hint.hide()
        rec_layout.addWidget(self.lbl_effects_hint)
        
        bottom_area.addWidget(group_rec)
        right_panel = QVBoxLayout()
        right_panel.addStretch() 
        self.btn_clear = QPushButton("Új Draft")
        self.btn_clear.setFixedSize(80, 35)
        self.btn_clear.setStyleSheet("QPushButton { font-size: 14px; background-color: #1A1A15; font-weight: bold; color: #BBBBBB; border: 1px solid #444; border-radius: 6px; } QPushButton:hover { background-color: #3A3A45; color: white; }")
        self.btn_clear.clicked.connect(self._on_clear_all)
        right_panel.addWidget(self.btn_clear, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        bottom_area.addLayout(right_panel, 1)
        layout.addLayout(bottom_area)
        from view.options_dialog import OptionsDialog
        self.options_dialog = OptionsDialog(self)
        return container

    def _create_combo(self, is_ban: bool = False) -> QComboBox:
        combo = QComboBox()
        width = 130 if is_ban else 150
        height = 35 if is_ban else 50
        icon_size = 24 if is_ban else 42
        
        combo.setFixedWidth(width)   
        combo.setMinimumHeight(height) 
        combo.setIconSize(QSize(icon_size, icon_size)) 
        combo.setStyleSheet("font-size: 14px; padding: 3px;")
        
        if is_ban: combo.setProperty("isBan", True)
        
        combo.addItem(QIcon(), "") 
        for name in self.champ_names:
            cid = self.name_map.get(name)
            combo.addItem(QIcon(f"assets/icons/{cid}.png"), name)
            
        combo.setEditable(True)
        combo.lineEdit().setPlaceholderText("Gépelj...") 
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        

        completer = combo.completer()
        
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

        completer.setFilterMode(Qt.MatchFlag.MatchContains)

        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        
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
        return [self.name_map.get(cb.currentText().strip(), -1) for cb in combos]

    def _get_unavailable(self) -> set:
        """Returns a set of already picked or banned champion IDs."""
        picks = [i for i in self._get_ids(self.blue_picks) + self._get_ids(self.red_picks) if i >= 0]
        bans = [i for i in self._get_ids(self.blue_bans) + self._get_ids(self.red_bans) if i >= 0]
        return set(picks).union(set(bans))

    def _reset_dash(self):
        """Hides results and readies dashboard for new calculation."""
        self.lbl_dash_title.show()
        self.wr_container.hide()
        self.bar_wr.hide()
        self.scroll_recommend.hide()

    def _on_clear_all(self):
        """Clears all champion selections and resets the UI."""
        self.chk_follow.setChecked(False)
        self._draft_unavailable()
        self._clear_role_metadata()
        self._invalidate_recommendations()
        for combo in self.blue_picks + self.red_picks + self.blue_bans + self.red_bans:
            combo.setCurrentIndex(0)
        self._reset_dash()
        self._clear_effect_badges()
        self.lbl_dash_title.setText("Válassz hősöket a draftoláshoz!")

    def _on_save_draft(self):
        if self._busy:
            return
        from services.saved_drafts import DraftFileError, drafts_directory, make_document, save_draft
        groups = {"blue": self.blue_picks, "red": self.red_picks,
                  "blue_bans": self.blue_bans, "red_bans": self.red_bans}
        if any(combo.currentText().strip() and combo.currentText().strip() not in self.name_map
               for group in groups.values() for combo in group):
            self._set_sync_status("Mentés előtt fejezd be a beírt hősnevek kiválasztását.")
            return
        tier = self.cb_rank.currentText()
        try:
            document = make_document({key: self._get_ids(combos) for key, combos in groups.items()}, self._pool_mapping,
                "MIXED" if tier == "ALL RANKS" else tier, "blue" if self.grp_team.checkedId() == 0 else "red",
                self.grp_role.checkedId(), {index: name for name, index in self.name_map.items()})
        except DraftFileError as error:
            self._set_sync_status(str(error))
            return
        directory = drafts_directory()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            directory = Path.home()
        path, _ = QFileDialog.getSaveFileName(self, "Draft mentése", str(directory / "draft.json"), "Erospredictor draft (*.json)")
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".json")
        try:
            save_draft(target, document)
        except DraftFileError as error:
            self._set_sync_status(str(error))
            return
        self._set_sync_status(f"Draft elmentve: {target}")

    def _on_load_draft(self):
        if self._busy:
            return
        from services.saved_drafts import DraftFileError, drafts_directory, load_draft
        directory = drafts_directory()
        path, _ = QFileDialog.getOpenFileName(self, "Draft betöltése", str(directory), "Erospredictor draft (*.json)")
        if not path:
            return
        try:
            state = load_draft(path, self._pool_mapping, available_ids=set(self.name_map.values()))
        except DraftFileError as error:
            self._set_sync_status(str(error))
            return
        # Disable the generation-guarded follower before changing any draft slot.
        self.chk_follow.setChecked(False)
        self._draft_follower.set_enabled(False)
        self._remember_ui_preference("auto_draft", False)
        self._draft_unavailable()
        self._clear_role_metadata()
        self._apply_import(state.values)
        self.cb_rank.setCurrentText("ALL RANKS" if state.tier == "MIXED" else state.tier)
        (self.btn_blue if state.team == "blue" else self.btn_red).setChecked(True)
        self.grp_role.button(state.role).setChecked(True)
        self._remember_ui_preference("preferred_role", state.role)
        self.lbl_dash_title.setText("A mentett draft betöltve.\nKérj ajánlást vagy predikciót.")
        self._set_sync_status(f"Mentett draft: {Path(path).name}. Automatikus követés kikapcsolva.")
        self.lbl_draft_sync.setToolTip(str(path))

    def _on_predict(self):
        """Handles predict button click by coordinating with the controller."""
        if not self.controller or self._busy: return
        self._invalidate_recommendations()
        self._reset_dash()
        
        blue = self._get_ids(self.blue_picks)
        red = self._get_ids(self.red_picks)
        b_bans = self._get_ids(self.blue_bans)
        r_bans = self._get_ids(self.red_bans)
        
        valid, err = self.controller.validate_draft(blue, red, b_bans, r_bans)
        if not valid:
            self.lbl_dash_title.setText(f"<span style='color: #EF5350;'>HIBA: {err}</span>")
            self._effects_context = None
            return

        div_ui = self.cb_rank.currentText()
        div = "MIXED" if div_ui == "ALL RANKS" else div_ui
        self.lbl_dash_title.setText("Számítás folyamatban...")
        context = self._current_effects_context()
        enabled = self.chk_effects.isChecked()
        self._clear_effect_badges()

        def predict():
            result = self.controller.predict_match(div, blue, red)
            effects, error = self._optional_inline_effects(context, enabled)
            return result, effects, error

        def display(result):
            prediction, effects, error = result
            self._display_prediction(prediction, div_ui)
            self._effects_context = context
            self._prediction_context = dict(context) if context is not None else None
            self._update_explain_button()
            if enabled:
                self._display_optional_effects(effects, error)

        self._start_task(predict, display)

    def _display_prediction(self, res, div_ui):
        self.lbl_dash_title.setText(f"<span style='color: #FFFFFF; font-size: 20px;'>Prediktált Győzelmi Arány ({div_ui})</span>")
        self.lbl_blue_wr.setText(res['blue_text'])
        self.lbl_red_wr.setText(res['red_text'])
        self.bar_wr.setValue(res['bar_val']) 
        
        self.wr_container.show()
        self.bar_wr.show()

    def _on_recommend(self):
        """Handles recommend button click by requesting and displaying results from controller."""
        if not self.controller or self._busy: return
        self._invalidate_recommendations()
        self._reset_dash()
        
        blue = self._get_ids(self.blue_picks)
        red = self._get_ids(self.red_picks)
        b_bans = self._get_ids(self.blue_bans)
        r_bans = self._get_ids(self.red_bans)
        
        valid, err = self.controller.validate_draft(blue, red, b_bans, r_bans)
        if not valid:
            self.lbl_dash_title.setText(f"<span style='color: #EF5350;'>HIBA: {err}</span>")
            return

        div_ui = self.cb_rank.currentText()
        div = "MIXED" if div_ui == "ALL RANKS" else div_ui
        bans = list(self._get_unavailable())
        t_id = self.grp_team.checkedId()
        r_id = self.grp_role.checkedId()
        picking = "blue" if t_id == 0 else "red"
        
        if (blue[r_id] if picking == "blue" else red[r_id]) >= 0:
            self.lbl_dash_title.setText("<span style='color: #FFCA28;'>Erre a pozícióra már választottál hőst!</span>")
            return
        
        self.lbl_dash_title.setText("Szimuláció futtatása...")
        context = dict(div=div, blue=blue, red=red, team=picking, r_idx=r_id)
        filter_meta = self.chk_meta.isChecked()
        enabled = self.chk_effects.isChecked()
        history = self._personal_history if feature_flags.PERSONAL_HISTORY_ENABLED and self.chk_personal.isChecked() else None

        candidates = member_ids(self._pool_library, self._pool_mapping, self.cb_pool.currentText()) if self.chk_pool_only.isChecked() else None
        if candidates is not None and not candidates:
            self.lbl_dash_title.setText("A választott hőslista üres. Adj hozzá hősöket, vagy kapcsold ki a listaszűrést.")
            return
        tags = internal_tags(self._pool_library, self._pool_mapping)
        robust = self.chk_robust.isChecked()
        manual_roles = dict(self._sync_manual_roles)

        def recommend():
            if history is not None:
                from services.personal_history import LcuHistoryClient
                LcuHistoryClient(self._lockfile_path).verify_account(history)
            extra = {}
            if candidates is not None:
                extra["candidate_ids"] = candidates
            if tags:
                extra["pool_tags"] = tags
            if robust:
                extra["scenarios"] = self.controller.draft_scenarios(div, blue, red, picking, manual_roles)
            recs = self.controller.recommend_champs(div, blue, red, bans, picking,
                                                     r_id, CHAMPION_COUNT, filter_meta,
                                                     **({"history": history} if history is not None else {}), **extra)
            focus = dict(context, champion_id=recs[0]['id']) if recs else None
            effects, error = self._optional_inline_effects(focus, enabled and bool(recs))
            return recs, effects, error

        def display(result):
            recs, effects, error = result
            self._display_recommendations(recs, context)
            if enabled:
                self._display_optional_effects(effects, error)

        self._start_task(recommend, display)

    def _display_recommendations(self, recs, context):
        picking, r_id = context['team'], context['r_idx']
        self._prediction_context = None
        self._recommendation_context = dict(context, recommendations=recs[:5]) if recs else None
        self._update_explain_button()
        self._last_recommendations = list(recs)
        self._selected_recommendation_id = recs[0]['id'] if recs else None
        self._effects_context = dict(context, champion_id=recs[0]['id']) if recs else None
        t_name = "Kék" if picking == "blue" else "Piros"
        r_name = self.roles[r_id]
        icon_name = {"Mid": "Middle", "ADC": "Bottom"}.get(r_name, r_name)
        img = f"<img src='assets/roles/{icon_name}_icon.png' width='28' height='28' style='vertical-align: middle; margin-left: 8px;'>"
        self.lbl_dash_title.setText(f"<div style='text-align: center;'><span style='color: #FFFFFF; font-size: 20px; font-weight: bold;'>Ajánlott hősök: {t_name} - {r_name} {img}</span></div>")
        if any(rec.get("robust", {}).get("count", 0) >= 2 for rec in recs):
            self.lbl_dash_title.setText(f"Ajánlott hősök: {t_name} – {r_name}<br><span style='font-size: 12px; color: #BDBDCB;'>Első 5: legrosszabb becslés szerint · fő %: aktuális draft</span>")
        elif any("robust" in rec for rec in recs):
            self.lbl_dash_title.setText(f"Ajánlott hősök: {t_name} – {r_name}\nNincs két használható ellenfél-kiosztás; normál sorrend.")
        
        if not recs:
            self.lbl_recommend.setText("<h3 style='color: #EF5350;'>Nem található megfelelő hős!</h3>")
            self.scroll_recommend.show()
            return
        self._render_recommendation_rows()

    def _render_recommendation_rows(self):
        id_name = {v: k for k, v in self.name_map.items()}
        html = "<table width='100%' cellspacing='0' cellpadding='8' align='center' style='margin-top: 21px;'>"
        colors = ["#323242", "#2C2C3A", "#262632", "#20202A", "#1A1A22"]
        for i, rec in enumerate(self._last_recommendations):
            cid, wr, color = rec['id'], rec['wr'], rec['display_color']
            name = escape(id_name.get(cid, "Unknown"))
            row_color = colors[min(i, len(colors) - 1)]
            selected = cid == self._selected_recommendation_id and self.chk_effects.isChecked()
            if selected:
                row_color = "#34475D"
            name_color = "#FFE082" if selected else "#FFFFFF"
            personal = rec.get("personal") if feature_flags.PERSONAL_HISTORY_ENABLED and self.chk_personal.isChecked() else None
            marker = personal.get("marker") if personal else None
            tags = rec.get("pool_tags", ())
            tag_colors = {"played": "#64B5F6", "comfort": "#81C784", "learn": "#B39DDB"}
            frame_color = ("#FFD166" if marker == "good" else
                           tag_colors["comfort"] if "comfort" in tags else
                           tag_colors["learn"] if "learn" in tags else
                           tag_colors["played"] if "played" in tags or marker == "played" else None)
            frame_color = frame_color or "transparent"
            frame = f"border-top: 2px solid {frame_color}; border-bottom: 2px solid {frame_color};"
            left = f"border-left: 2px solid {frame_color};"
            right = f"border-right: 2px solid {frame_color};"
            html += f"<tr bgcolor='{row_color}'>"
            html += f"<td style='width: 45px; {frame} {left}'><img src='assets/icons/{cid}.png' width='38' height='38' style='border-radius: 6px;'></td>"
            html += (f"<td style='font-size: 18px; font-weight: bold; text-align: left; white-space: nowrap; {frame}'>"
                     f"<a href='champion:{cid}' style='color: {name_color}; text-decoration: none;'>{name}</a>")
            details = []
            if tags:
                details.extend(f"<span style='color: {tag_colors[tag]};'>{TAGS[tag]}</span>"
                               for tag in tags if tag in TAGS)
            robust = rec.get("robust")
            if robust and robust.get("count", 0) >= 2:
                warning = " ⚠" if robust.get("forced") else ""
                title = f"{robust['count']} kiosztás: {robust['min']:.1f}–{robust['max']:.1f}%{warning}"
                html += f" <a href='layouts:{cid}' title='{title}' style='font-size: 10px; color: #80CBC4;'>↔{robust['count']}{warning}</a>"
            if personal:
                # One compact metadata line; personal evidence remains in the detailed analysis.
                summary = "Bevált saját hős" if marker == "good" else "Játszott saját hős"
                if not tags:
                    details.append(f"<span style='color: #CDD6E5;'>{summary}</span>")
                elif marker == "good":
                    details.append("<span style='color: #FFD166;'>★</span>")
            html += "<br><span style='font-size: 10px;'>" + (" · ".join(details) or "&nbsp;") + "</span>"
            html += "</td>"
            blind = []
            if rec.get("blind") is not None:
                from services.blind_pick import risk_badges
                blind = [f"<span style='color: {tint};'>{escape(caption)}</span>"
                         for caption, tint in risk_badges(rec["blind"])]
            blind_line = " · ".join(blind) or "&nbsp;"
            html += (f"<td width='220' style='font-size: 18px; color: {color}; font-weight: bold; text-align: right; white-space: nowrap; {frame} {right}'>"
                     f"{wr:.2f}%<br><span style='font-size: 10px; font-weight: normal;'>{blind_line}</span></td>")
            html += "</tr>"
        html += "</table>"
        self.lbl_recommend.setText(html)
        self.scroll_recommend.show()

    def _current_effects_context(self):
        blue, red = self._get_ids(self.blue_picks), self._get_ids(self.red_picks)
        team = "blue" if self.grp_team.checkedId() == 0 else "red"
        role = self.grp_role.checkedId()
        if role < 0 or (blue if team == "blue" else red)[role] < 0:
            return None
        rank = self.cb_rank.currentText()
        return dict(div="MIXED" if rank == "ALL RANKS" else rank,
                    blue=blue, red=red, team=team, r_idx=role)

    def _optional_inline_effects(self, context, enabled):
        """Worker-only optional step: failure must not hide the main prediction."""
        if not enabled:
            return None, None
        if context is None:
            return None, "A kijelölt csapat és pozíció helyén még nincs hős. Válassz hőst, vagy kérj ajánlást."
        try:
            return self.controller.explain_champion_effects(**context), None
        except Exception as exc:
            return None, str(exc) or "A hőshatás most nem számítható ki."

    def _display_optional_effects(self, result, error):
        if not self.chk_effects.isChecked():
            return
        if result is not None:
            self._display_inline_effects(result)
        else:
            self._clear_effect_badges()
            self.lbl_effects_hint.setText(error or "Nincs elemezhető ajánlott hős.")
            self.lbl_effects_hint.show()

    def _clear_effect_badges(self):
        for badge in self.blue_effects + self.red_effects:
            badge.clear()
            badge.setToolTip("")
            badge.setProperty("laneOpponent", False)
            badge.setProperty("focusChampion", False)
            badge.setStyleSheet("font-size: 13px; font-weight: bold; border: 1px solid transparent;")

    @staticmethod
    def _effect_color(delta):
        magnitude = abs(delta)
        if magnitude < 0.005:
            return "#A0A0B5"
        if delta > 0:
            return "#B9F6CA" if magnitude < 0.5 else "#81C784" if magnitude < 2 else "#00E676"
        return "#FFCDD2" if magnitude < 0.5 else "#EF9A9A" if magnitude < 2 else "#FF5252"

    def _display_inline_effects(self, result):
        if not self.chk_effects.isChecked():
            return
        self._clear_effect_badges()
        badges = {"blue": self.blue_effects, "red": self.red_effects}
        for item in result['interactions']:
            badge = badges[item['team']][item['role_index']]
            delta = float(item['delta_pp'])
            arrow = "▲" if delta >= 0.005 else "▼" if delta <= -0.005 else "→"
            text = f"{arrow} {delta:+.2f}" if abs(delta) >= 0.005 else "→ 0.00"
            lane = bool(item.get('is_lane_opponent'))
            badge.setText(text)
            badge.setProperty("laneOpponent", lane)
            border = "#FFFFFF" if lane else "transparent"
            badge.setStyleSheet(f"font-size: 13px; font-weight: bold; color: {self._effect_color(delta)}; "
                               f"border: {'2' if lane else '1'}px solid {border}; "
                               "border-radius: 5px; padding: 3px; background: transparent;")
            badge.setToolTip(("Azonos pozícióban játszó lane-ellenfél. " if lane else "")
                             + f"A fókuszhős becsült többletének változása: {delta:+.2f} százalékpont. "
                             "Pozitív: kedvező; negatív: kedvezőtlen. A hatások külön vizsgálatok, nem összeadhatók.")
        focus = badges[result['team']][result['role_index']]
        focus.setText("● Fókusz")
        focus.setProperty("focusChampion", True)
        focus.setStyleSheet("font-size: 13px; font-weight: bold; color: #90CAF9; "
                           "border: 1px solid #64B5F6; border-radius: 5px; padding: 3px;")
        names = {v: k for k, v in self.name_map.items()}
        name = names.get(result['id'], str(result['id']))
        side = "Kék" if result['team'] == "blue" else "Piros"
        focus.setToolTip(f"Kijelölt hős: {name}. A többi jelzés ehhez a hőshöz és a {side} csapathoz viszonyít.")
        self.lbl_effects_hint.setText(
            f"Fókusz: {side} {self.roles[result['role_index']]} – {name}. "
            "▲ kedvező, ▼ kedvezőtlen hatás (százalékpont). Fehér keret: lane-ellenfél.")
        self.lbl_effects_hint.show()

    def _on_effects_toggled(self, enabled):
        self._clear_effect_badges()
        if self._last_recommendations:
            self._render_recommendation_rows()
        if not enabled:
            self.lbl_effects_hint.hide()
            return
        context = self._effects_context or self._current_effects_context()
        if not context or not self.controller:
            self.lbl_effects_hint.setText("Válassz lent csapatot és pozíciót, majd kérj predikciót vagy ajánlást.")
            self.lbl_effects_hint.show()
            return
        self._run_inline_effects(context)

    def _run_inline_effects(self, context):
        if self._busy or not self.controller or not self.chk_effects.isChecked():
            return
        valid, error = self.controller.validate_draft(
            context['blue'], context['red'], self._get_ids(self.blue_bans), self._get_ids(self.red_bans))
        if not valid:
            self.lbl_effects_hint.setText(error)
            self.lbl_effects_hint.show()
            return
        self._effects_context = context
        self._clear_effect_badges()
        self.lbl_effects_hint.setText("A kijelölt hős egyszerű hatásainak számítása…")
        self.lbl_effects_hint.show()
        self._start_task(lambda: self._optional_inline_effects(context, True),
                         lambda result: self._display_optional_effects(*result))

    def _on_recommendation_selected(self, link):
        if link.startswith("layouts:") and not self._busy and self._recommendation_context:
            try:
                self._show_layouts(int(link.split(":", 1)[1]))
            except ValueError:
                pass
            return
        if self._busy or not self._recommendation_context or not link.startswith("champion:"):
            return
        try:
            champion_id = int(link.split(":", 1)[1])
        except ValueError:
            return
        if champion_id not in {rec['id'] for rec in self._last_recommendations}:
            return
        self._selected_recommendation_id = champion_id
        context = {key: value for key, value in self._recommendation_context.items()
                   if key != 'recommendations'}
        self._effects_context = dict(context, champion_id=champion_id)
        self._render_recommendation_rows()
        self._run_inline_effects(self._effects_context)

    def _connect_draft_changes(self):
        for combo in self.blue_picks + self.red_picks + self.blue_bans + self.red_bans:
            combo.currentTextChanged.connect(self._invalidate_recommendations)
        self.cb_rank.currentTextChanged.connect(self._invalidate_recommendations)
        self.grp_team.idClicked.connect(self._invalidate_recommendations)
        self.grp_role.idClicked.connect(self._invalidate_recommendations)
        self.chk_meta.toggled.connect(self._invalidate_recommendations)
        self.chk_robust.toggled.connect(self._invalidate_recommendations)
        self.chk_pool_only.toggled.connect(self._invalidate_recommendations)
        self.cb_pool.currentTextChanged.connect(self._invalidate_recommendations)
        self.cb_rank.currentTextChanged.connect(self._sync_rank_changed)
        self._role_menu_widgets = []  # Keep wrappers for the combo's C++-created line edits alive.
        for team, combos in (("blue", self.blue_picks), ("red", self.red_picks)):
            labels = self.blue_role_labels if team == "blue" else self.red_role_labels
            for role, combo in enumerate(combos):
                combo.currentTextChanged.connect(lambda _text, t=team, r=role: self._clear_role_slot(t, r))
                combo.activated.connect(lambda _index, t=team, r=role: self._confirm_manual_role(t, r))
                combo.lineEdit().editingFinished.connect(lambda t=team, r=role: self._confirm_manual_role(t, r))
                widgets = (combo, combo.lineEdit(), labels[role])
                self._role_menu_widgets.extend(widgets)
                for widget in widgets:
                    widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                    widget.customContextMenuRequested.connect(
                        lambda point, t=team, r=role, w=widget: self._show_pick_role_menu(t, r, w, point))
                self._clear_role_slot(team, role)

    def _pick_role_menu(self, team, role):
        combos = self.blue_picks if team == "blue" else self.red_picks
        ids = tuple(self._get_ids(combos))
        menu = QMenu(self)
        title = menu.addAction(f"{combos[role].currentText() or 'Üres pozíció'} · pozíciócsere")
        title.setEnabled(False)
        menu.addSeparator()
        for target, role_name in enumerate(self.roles):
            occupant = combos[target].currentText().strip()
            action = menu.addAction(f"{role_name} ↔ {occupant}" if occupant else f"{role_name} · üres")
            action.setData(target)
            reason = self._role_swap_block_reason(team, role, target)
            action.setEnabled(target != role and not reason)
            action.setToolTip(reason)
            action.triggered.connect(lambda _checked=False, t=target:
                                     self._swap_pick_roles(team, role, t, expected_ids=ids))
        return menu

    def _show_pick_role_menu(self, team, role, widget, point):
        menu = self._pick_role_menu(team, role)
        if widget is (self.blue_picks if team == "blue" else self.red_picks)[role].lineEdit():
            text_menu = widget.createStandardContextMenu()
            text_menu.setTitle("Szövegszerkesztés")
            menu.addSeparator()
            menu.addMenu(text_menu)
        try:
            menu.exec(widget.mapToGlobal(point))
        finally:
            menu.deleteLater()

    def _role_swap_block_reason(self, team, source, target):
        if self._busy or self._manual_import_active:
            return "A folyamatban lévő művelet alatt nem cserélhetők a pozíciók."
        combos = self.blue_picks if team == "blue" else self.red_picks
        if self._get_ids([combos[source]])[0] < 0:
            return "Előbb válassz egy hőst az áthelyezéshez."
        if combos[target].currentText().strip() and self._get_ids([combos[target]])[0] < 0:
            return "A célpozíción előbb fejezd be vagy töröld a beírt hősnevet."
        snapshot = self._sync_last_snapshot if self.chk_follow.isChecked() else None
        if snapshot is not None:
            own = snapshot.own_team or self.cb_sync_side.currentData() or self._sync_manual_side
            fixed = {pick.role for pick in snapshot.picks if pick.role is not None
                     and (pick.team == own or pick.team == "own")}
            if team == own and (source in fixed or target in fixed):
                return "A saját csapat kliens által közölt pozícióját az automatikus követés kezeli."
        return ""

    def _swap_pick_roles(self, team, source, target, expected_ids=None):
        """Move or swap two picks, confirming both roles before the next poll."""
        if team not in ("blue", "red") or any(type(role) is not int or not 0 <= role < 5
                                              for role in (source, target)) or source == target:
            return False
        combos = self.blue_picks if team == "blue" else self.red_picks
        ids = self._get_ids(combos)
        reason = self._role_swap_block_reason(team, source, target)
        if expected_ids is not None and tuple(ids) != expected_ids:
            reason = "A draft közben megváltozott; nyisd meg újra a pozíciócsere menüt."
        if reason:
            self._set_sync_status(reason)
            return False
        source_id, target_id = ids[source], ids[target]
        manual = dict(self._sync_manual_roles)
        manual[(team, source_id)] = target
        if target_id >= 0:
            manual[(team, target_id)] = source
        snapshot = self._sync_last_snapshot if self.chk_follow.isChecked() else None
        if snapshot is not None:
            from services.draft_assignment import assign_draft
            document, _ = self._load_roles()
            tier = self.cb_rank.currentText()
            try:
                result = assign_draft(snapshot, document,
                    own_side=snapshot.own_team or self.cb_sync_side.currentData() or self._sync_manual_side,
                    previous_roles=manual, tier="MIXED" if tier == "ALL RANKS" else tier)
                expected = (target_id if target_id >= 0 else None, source_id)
                if (result.values[team][source], result.values[team][target]) != expected:
                    raise ValueError("A helyi draft eltér a kliensétől; kézi szerkesztéshez kapcsold ki az automatikus követést.")
            except ValueError as error:
                self._set_sync_status(f"Pozíciócsere: {error}")
                return False
            self._sync_manual_roles = manual
            self._receive_followed_draft(snapshot, force=True)
            return True
        names = {index: name for name, index in self.name_map.items()}
        blocked = [combos[source].blockSignals(True), combos[target].blockSignals(True)]
        try:
            combos[source].setCurrentIndex(combos[source].findText(names.get(target_id, "")))
            combos[target].setCurrentIndex(combos[target].findText(names[source_id]))
        finally:
            combos[source].blockSignals(blocked[0])
            combos[target].blockSignals(blocked[1])
        self._sync_manual_roles = manual
        for role in (source, target):
            self._clear_role_slot(team, role)
            if self._get_ids([combos[role]])[0] >= 0:
                labels = self.blue_role_labels if team == "blue" else self.red_role_labels
                labels[role].setText(f"{self.roles[role]}:<br><span style='font-size: 10px; color: #BDBDCB;'>kézi</span>")
                labels[role].setToolTip("Kézzel megerősített szerepkör\n" + ROLE_SWAP_HINT)
                combos[role].setToolTip(labels[role].toolTip())
        self._invalidate_recommendations()
        self._reset_dash()
        self.lbl_dash_title.setText("A pozíciók megváltoztak; kérj új predikciót vagy ajánlást.")
        self._set_sync_status("Kézi pozíciócsere elmentve az aktuális draftban.")
        return True

    def _show_options(self):
        self.options_dialog.show()
        self.options_dialog.raise_()
        self.options_dialog.activateWindow()

    def _show_layouts(self, champion_id):
        record = next((item for item in self._last_recommendations if item["id"] == champion_id), None)
        scenarios = record.get("robust", {}).get("scenarios", []) if record else []
        if not scenarios:
            return
        names = {index: name for name, index in self.name_map.items()}
        dialog = QDialog(self)
        dialog.setWindowTitle("Vizsgált ellenfél-kiosztások")
        dialog.resize(650, 580)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        html = "<p>A súlyok a vizsgált legfeljebb három kiosztás közötti relatív statisztikai súlyok. A tartomány ezek eltérő modellbecsléseit mutatja, nem konfidenciaintervallum.</p>"
        for index, scenario in enumerate(scenarios, 1):
            html += f"<h3>{index}. kiosztás · {scenario['wr']:.2f}% · relatív súly: {scenario['weight']:.1%}</h3>"
            if scenario["forced"]:
                html += "<p>⚠ Kényszerített, 5% alatti vagy adat nélküli pozíciót is tartalmaz.</p>"
            html += "<table cellpadding='6'><tr><th>Pozíció</th><th>Kék</th><th>Piros</th></tr>"
            for role, role_name in enumerate(self.roles):
                html += f"<tr><td>{role_name}</td><td>{escape(names.get(scenario['blue'][role], '–'))}</td><td>{escape(names.get(scenario['red'][role], '–'))}</td></tr>"
            html += "</table>"
        text.setHtml(html)
        layout.addWidget(text)
        button = QPushButton("Bezárás")
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)
        dialog.exec()

    def _edit_pool(self):
        if self._busy:
            return
        from view.champion_pool_dialog import ChampionPoolDialog
        dialog = ChampionPoolDialog(self._pool_library, self._pool_mapping,
            {index: name for name, index in self.name_map.items()}, self._personal_history, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = self.cb_pool.currentText()
        self._pool_library = dialog.document
        self.cb_pool.blockSignals(True)
        self.cb_pool.clear()
        self.cb_pool.addItems([ALL_MINE] + sorted(self._pool_library["lists"]))
        self.cb_pool.setCurrentText(selected if selected in [ALL_MINE] + list(self._pool_library["lists"]) else ALL_MINE)
        self.cb_pool.blockSignals(False)
        self._remember_ui_preference("active_pool", self.cb_pool.currentText())
        self._invalidate_recommendations()
        self._reset_dash()
        self.lbl_dash_title.setText("Hőslisták elmentve. Kérj új ajánlást vagy banötleteket.")

    def _on_ban_advice(self):
        if self._busy or self.controller is None:
            return
        pool = member_ids(self._pool_library, self._pool_mapping, self.cb_pool.currentText())
        if not pool:
            self.lbl_dash_title.setText("A bansegédhez előbb adj hősöket a kiválasztott saját listához.")
            return
        blue, red = self._get_ids(self.blue_picks), self._get_ids(self.red_picks)
        valid, error = self.controller.validate_draft(blue, red, self._get_ids(self.blue_bans), self._get_ids(self.red_bans))
        if not valid:
            self.lbl_dash_title.setText(error)
            return
        tier = self.cb_rank.currentText()
        role = self._ban_role if self._ban_role is not None else self.grp_role.checkedId()
        div = "MIXED" if tier == "ALL RANKS" else tier
        unavailable = self._get_unavailable()
        title = f"Bansegéd · {self.cb_pool.currentText()} · {tier}"
        history = self._personal_history if feature_flags.PERSONAL_HISTORY_ENABLED else None
        path = self._lockfile_path
        def prepare():
            personal = history
            note = "Nincs betöltött saját előzmény. Most csak az általános adatok számítanak; frissítés: Beállítások → Saját előzmények."
            if personal is not None:
                from services.personal_history import LcuHistoryClient, DraftImportError
                try:
                    LcuHistoryClient(path).verify_account(personal)
                    note = f"Saját minta: {personal.name} · {len(personal.games)} betöltött rangsorolt meccs. Csak a kiválasztott pool, pozíció és ismert lane-ellenfél számít."
                except DraftImportError:
                    personal = None
                    note = "A betöltött előzmény fiókja most nem ellenőrizhető a helyi kliensben. Csak általános statisztika; frissítsd a saját előzményeket."
            return self.controller.recommend_ban_lists(div, pool, unavailable, **({"history": personal} if personal is not None else {})), note
        self._start_task(prepare, lambda result: self._display_bans(result[0], title, role, result[1]))

    def _display_bans(self, lists, title, role, history_note):
        from view.ban_advisor_dialog import BanAdvisorDialog
        names = {index: name for name, index in self.name_map.items()}
        dialog = BanAdvisorDialog(lists, names, self.roles, role, title, history_note, self)
        dialog.role_changed.connect(self._remember_ban_role)
        dialog.exec()

    def _remember_ban_role(self, role):
        self._ban_role = role
        self._remember_ui_preference("ban_role", role)

    def _invalidate_recommendations(self, *_):
        self.scroll_recommend.hide()
        self.lbl_recommend.clear()
        self._recommendation_context = None
        self._prediction_context = None
        self._effects_context = None
        self._last_recommendations = []
        self._selected_recommendation_id = None
        self._update_explain_button()
        self._clear_effect_badges()
        if self.chk_effects.isChecked():
            self.lbl_effects_hint.setText("Válassz lent csapatot és pozíciót, majd kérj predikciót vagy ajánlást.")
            self.lbl_effects_hint.show()

    def _update_explain_button(self):
        prediction = self._prediction_context is not None
        recommendation = self._recommendation_context is not None
        self.btn_explain.setText("Miért? – Predikció fókuszhőse" if prediction else
            "Miért? – Kijelölt hős + első 5" if recommendation else "Miért? – Részletes elemzés")
        self.btn_explain.setToolTip("Csak a kijelölt csapat és pozíció predikcióban szereplő hősének részletes elemzése."
            if prediction else "Ajánlás után a kijelölt hős + első öt, predikció után a fókuszhős elemezhető. Fókuszhöz válassz hőst a kijelölt pozíción, majd kérj predikciót.")
        self.btn_explain.setEnabled(not self._busy and (prediction or recommendation))

    def _set_busy(self, busy):
        self._busy = busy
        controls = (self.blue_picks + self.red_picks + self.blue_bans + self.red_bans
                    + [self.cb_rank, self.btn_predict, self.btn_rec, self.btn_clear,
                       self.btn_import, self.btn_lockfile, self.chk_meta, self.chk_effects,
                       self.btn_history, self.cb_history_limit, self.chk_robust, self.cb_sync_side,
                       self.btn_pool, self.cb_pool, self.chk_pool_only, self.btn_bans]
                    + self.grp_team.buttons() + self.grp_role.buttons())
        for control in controls:
            control.setEnabled(not busy)
        self.btn_drafts.setEnabled(not busy)
        self.action_save_draft.setEnabled(not busy)
        self.action_load_draft.setEnabled(not busy)
        self._update_explain_button()
        self.chk_personal.setEnabled(feature_flags.PERSONAL_HISTORY_ENABLED and not busy and self._personal_history is not None and bool(self._personal_history.games))
        self.btn_history.setEnabled(feature_flags.PERSONAL_HISTORY_ENABLED and (not busy or self._history_loading))
        self.cb_history_limit.setEnabled(feature_flags.PERSONAL_HISTORY_ENABLED and not busy)
        self.chk_history_service.setEnabled(feature_flags.PERSONAL_HISTORY_ENABLED and not busy)
        if feature_flags.PERSONAL_HISTORY_ENABLED and busy and self._history_loading:
            self.btn_history.setEnabled(True)

    def _start_task(self, operation, on_success, on_error=None):
        """Serialize model/client operations and keep GUI responsive."""
        if self._busy:
            return
        self._set_busy(True)
        self._task_finish_pending = False
        task = BackgroundTask(operation, self)
        self._task = task

        def receive(result):
            self._task_callback_active = True
            try:
                on_success(result)
            except Exception as exc:
                self._show_task_error(str(exc))
            finally:
                self._task_callback_active = False
                if self._task_finish_pending:
                    self._task_finish_pending = False
                    self._task_finished()

        task.succeeded.connect(receive)
        task.failed.connect(on_error or self._show_task_error)
        task.finished.connect(self._task_finished)
        task.start()

    def _show_task_error(self, message):
        self._reset_dash()
        self._clear_effect_badges()
        self.lbl_dash_title.setWordWrap(True)
        self.lbl_dash_title.setText(f"<span style='color: #EF5350;'>HIBA: {escape(message)}</span>")

    def _task_finished(self):
        sender = self.sender()
        if isinstance(sender, BackgroundTask) and sender is not self._task:
            return
        if self._task_callback_active:
            self._task_finish_pending = True
            return
        task = self._task
        self._task = None
        if self._history_loading:
            self._history_loading = False
            self.btn_history.setText("Előzmények frissítése")
        self._set_busy(False)
        manual_import = self._manual_import_active
        self._manual_import_active = False
        pending = None if manual_import else self._sync_pending
        self._sync_pending = None
        if pending is not None and self.chk_follow.isChecked():
            self._receive_followed_draft(pending, force=True)
        if task is not None:
            task.deleteLater()
        if manual_import:
            self._draft_follower.set_enabled(self.chk_follow.isChecked())

    def _on_explain(self):
        if self._busy or not (self._recommendation_context or self._prediction_context):
            return
        prediction = self._prediction_context is not None
        context = dict(self._prediction_context) if prediction else dict(self._recommendation_context, selected_id=self._selected_recommendation_id)
        self.lbl_dash_title.setText("Predikció fókuszhőse: részletes elemzés…" if prediction else "Kijelölt hős + első 5: részletes elemzés…")
        history = self._personal_history if feature_flags.PERSONAL_HISTORY_ENABLED and self.chk_personal.isChecked() else None
        def explain():
            if history is not None:
                from services.personal_history import LcuHistoryClient
                LcuHistoryClient(self._lockfile_path).verify_account(history)
            operation = self.controller.explain_prediction if prediction else self.controller.explain_recommendations
            return operation(**context, **({"history": history} if history is not None else {}))
        self._start_task(explain,
                         self._display_explanations)

    def _on_personal_toggled(self, *_):
        if not feature_flags.PERSONAL_HISTORY_ENABLED:
            self.chk_personal.setChecked(False)
            return
        if self._last_recommendations:
            if self.chk_personal.isChecked() and self._personal_history is not None and not any(
                    "personal" in row for row in self._last_recommendations):
                self._invalidate_recommendations()
                self._reset_dash()
                self.lbl_dash_title.setText("Saját hőseid jelöléséhez kérj új ajánlást.")
                return
            self._render_recommendation_rows()

    def _on_startup_history(self):
        # Restoring a checkbox must never silently forward session credentials.
        self._on_refresh_history(local_only=True)

    def _on_refresh_history(self, *, local_only=False):
        if not feature_flags.PERSONAL_HISTORY_ENABLED:
            return
        if self._history_loading and self._task is not None:
            self._task.requestInterruption()
            self.btn_history.setText("Leállítás…")
            return
        if self._busy:
            return
        self._personal_history = None
        self._invalidate_recommendations()
        self._reset_dash()
        self._history_loading = True
        count = self.cb_history_limit.currentData()
        allow_service = self.chk_history_service.isChecked() and not local_only
        self.lbl_history.setText(f"Betöltés: {count} meccs…")
        self.btn_history.setText("Beolvasás leállítása")
        path = self._lockfile_path
        def read():
            from services.personal_history import LcuHistoryClient
            return LcuHistoryClient(path).read_history(count,
                cancelled=lambda: QThread.currentThread().isInterruptionRequested(),
                **({"allow_client_service": True} if allow_service else {}))
        self._start_task(read, self._history_loaded, self._history_error)

    def _history_loaded(self, history):
        if not feature_flags.PERSONAL_HISTORY_ENABLED:
            self._personal_history = None
            return
        self._personal_history = history if history.games else None
        label = f"{history.name} · {len(history.games)} rangsorolt meccs"
        self.lbl_history.setText(self.lbl_history.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, 235))
        source = "Riot kliensszolgáltatás" if history.source == "riot-client-service" else "helyi LCU"
        fetched = (f"Forrás: {source}. Most {history.examined_games}/{history.requested_games} különböző meccs érkezett, "
                   f"ebből {history.ranked_games_seen} rangsorolt. A mentett meccsek száma: {len(history.games)}.")
        self.lbl_history.setToolTip(label + "\n" + fetched + "\n" + "\n".join(history.warnings))
        if history.limited and history.source == "lcu":
            self.lbl_dash_title.setText(
                f"A helyi kliens {history.examined_games}/{history.requested_games} meccset adott vissza, majd ugyanazokat ismételte.\n"
                "Régebbi meccsek: Beállítások → Saját előzmények → Régebbi meccsek (Riot).")
        elif history.limited:
            self.lbl_dash_title.setText(f"A Riot szolgáltatása {history.examined_games}/{history.requested_games} különböző meccset adott vissza.")
        elif history.requested_games and not history.ranked_games_seen and not history.games:
            self.lbl_dash_title.setText(f"A kapott {history.examined_games} meccs között nincs rangsorolt mérkőzés.")

    def _history_error(self, message):
        self._personal_history = None
        self.lbl_history.setText("Előzmények nem betöltve")
        self.lbl_history.setToolTip(message)

    def _set_sync_status(self, message):
        self.lbl_draft_sync.setText(self.lbl_draft_sync.fontMetrics().elidedText(
            message, Qt.TextElideMode.ElideRight, max(300, self.width() - 40)))
        self.lbl_draft_sync.setToolTip(message)

    def _read_followed_draft(self):
        from services.draft_import import LcuDraftClient
        return LcuDraftClient(lockfile_path=self._lockfile_path).read_draft()

    def _on_follow_toggled(self, enabled):
        self._sync_pending = None
        self._sync_last_snapshot = None
        self._sync_latest_snapshot = None
        self._draft_follower.set_enabled(enabled and not self._manual_import_active)

    def _draft_unavailable(self):
        self._sync_pending = None
        self._sync_last_snapshot = None
        self._sync_session_key = ""
        self._sync_manual_side = None
        self._sync_manual_roles = {}
        self._sync_latest_snapshot = None

    def _on_sync_side_changed(self, *_):
        self._sync_manual_side = self.cb_sync_side.currentData()
        self._sync_manual_roles = {}
        if self.chk_follow.isChecked() and self._sync_latest_snapshot is not None:
            self._receive_followed_draft(self._sync_latest_snapshot, force=True)

    def _load_roles(self):
        from services.role_distribution import DEFAULT_PATH, load_role_distribution
        try:
            stamp = DEFAULT_PATH.stat().st_mtime_ns
        except OSError:
            stamp = None
        changed = self._role_document is None or stamp != self._role_stamp
        if changed:
            self._role_document = load_role_distribution()
            self._role_stamp = stamp
        return self._role_document, changed

    def _sync_rank_changed(self, *_):
        if self.chk_follow.isChecked() and self._sync_last_snapshot is not None:
            self._receive_followed_draft(self._sync_last_snapshot, force=True)

    def _receive_followed_draft(self, snapshot, force=False):
        if not self.chk_follow.isChecked():
            return
        self._sync_latest_snapshot = snapshot
        if self._busy:
            self._sync_pending = snapshot
            return
        from services.draft_assignment import assign_draft
        document, changed = self._load_roles()
        previous = self._sync_last_snapshot
        if not force and not changed and snapshot == previous:
            return
        if snapshot.session_key and snapshot.session_key != self._sync_session_key:
            self._sync_manual_roles = {}
            self._sync_manual_side = None
            previous = None
        if snapshot.own_team and self._sync_manual_side and snapshot.own_team != self._sync_manual_side:
            self._sync_manual_roles = {}
            self._sync_manual_side = None
        self._sync_session_key = snapshot.session_key
        tier = self.cb_rank.currentText()
        try:
            result = assign_draft(snapshot, document, own_side=snapshot.own_team or self.cb_sync_side.currentData() or self._sync_manual_side,
                                  previous_roles=self._sync_manual_roles,
                                  tier="MIXED" if tier == "ALL RANKS" else tier)
            names = {index: name for name, index in self.name_map.items()}
            if any(cid is not None and cid not in names for row in result.values.values() for cid in row):
                raise ValueError("A beolvasott hős nincs a felület katalógusában. Frissítsd a hősadatokat.")
        except ValueError as error:
            self._set_sync_status(f"Draftkövetés: {error} A kézi beolvasással javítható.")
            return
        def signature(values):
            return tuple(tuple((values[key] + [None] * 5)[:5]) for key in ("blue", "red", "blue_bans", "red_bans"))
        current = {key: [cid if cid >= 0 else None for cid in self._get_ids(combos)]
                   for key, combos in (("blue", self.blue_picks), ("red", self.red_picks),
                                       ("blue_bans", self.blue_bans), ("red_bans", self.red_bans))}
        self._sync_applying = True
        try:
            if signature(current) != signature(result.values):
                self._apply_import(result.values)
            own = snapshot.own_team or self.cb_sync_side.currentData() or self._sync_manual_side
            if own and (previous is None or previous.own_team != snapshot.own_team):
                (self.btn_blue if own == "blue" else self.btn_red).setChecked(True)
                self._invalidate_recommendations()
                self._reset_dash()
            if snapshot.own_role is not None and (previous is None or previous.own_role != snapshot.own_role):
                self.grp_role.button(snapshot.own_role).setChecked(True)
                self._invalidate_recommendations()
                self._reset_dash()
            self._show_role_metadata(result.assignments)
        finally:
            self._sync_applying = False
        self._sync_last_snapshot = snapshot
        present = {(item.team, item.internal_id) for item in result.assignments}
        self._sync_manual_roles = {key: value for key, value in self._sync_manual_roles.items() if key in present}
        inferred = sum(item.source in ("statistics", "forced") for item in result.assignments if item.role is not None)
        assigned = sum(item.role is not None for item in result.assignments)
        count_bans = len(result.values["blue_bans"]) + len(result.values["red_bans"])
        mode = "Custom" if snapshot.custom_game else {400: "Normal draft", 420: "Ranked Solo/Duo", 440: "Ranked Flex"}.get(snapshot.queue_id, "Draft")
        summary = f"{mode}: {assigned}/{len(snapshot.picks)} látható hős besorolva, {count_bans} tiltás; {inferred} becsült pozíció."
        forced = sum(item.source == "forced" for item in result.assignments)
        if forced:
            summary += f" ⚠ {forced} kényszerített pozíció."
        if snapshot.roster_sizes != (0, 0):
            summary += f" Kliensnévsor: {sum(snapshot.roster_sizes)} fő."
        catalog_gaps = sum(note.startswith("Ismeretlen hősazonosító:") for note in result.warnings)
        if catalog_gaps:
            summary += f" ⚠ {catalog_gaps} katalógushiány; frissítsd a hősadatokat."
        notes = [note for note in result.warnings if result.unresolved or not note.startswith("A nem közölt szerepköröket")]
        if result.unresolved:
            unknown = ", ".join(names.get(pick.internal_id, str(pick.riot_id)) for pick in result.unresolved)
            summary += f" Kézi pozíció szükséges: {unknown} (LoL-kliens beolvasása)."
        self._set_sync_status(summary)
        self.lbl_draft_sync.setToolTip(summary + "\n" + "\n".join(notes))

    def _clear_role_slot(self, team, role):
        labels = self.blue_role_labels if team == "blue" else self.red_role_labels
        combos = self.blue_picks if team == "blue" else self.red_picks
        labels[role].setText(f"{self.roles[role]}:")
        labels[role].setToolTip(ROLE_SWAP_HINT)
        combos[role].setToolTip(ROLE_SWAP_HINT)

    def _clear_role_metadata(self):
        for team in ("blue", "red"):
            for role in range(5):
                self._clear_role_slot(team, role)

    def _show_role_metadata(self, assignments):
        self._clear_role_metadata()
        for item in assignments:
            if item.role is None:
                continue
            labels = self.blue_role_labels if item.team == "blue" else self.red_role_labels
            combos = self.blue_picks if item.team == "blue" else self.red_picks
            if self._get_ids([combos[item.role]]) != [item.internal_id]:
                continue
            if item.source == "forced":
                caption, source, color = "⚠ kényszer", "Teljes ellenfélcsapatból kényszerített, 5% alatti vagy adat nélküli pozíció. Ellenőrizd kézzel.", "#EF9A9A"
            elif item.source == "statistics":
                caption = f"~{100 * item.role_probability:.1f}%"
                source = "Statisztikából becsült szerepkör"
                color = "#E4AC43" if item.role_probability < 0.2 or item.sample_size < 20 else "#80CBC4"
            else:
                caption = "kliens" if item.source == "client" else "kézi"
                source = "A kliens által közölt szerepkör" if item.source == "client" else "Kézzel megerősített szerepkör"
                color = "#BDBDCB"
            tooltip = source
            if item.sample_size:
                distribution = ", ".join(f"{self.roles[role]}: {probability:.1%}" for role, probability in enumerate(item.probabilities))
                tooltip += f"\n{item.sample_size} megfigyelt választás; {distribution}."
            if item.source in ("statistics", "forced"):
                tooltip += "\nA szám a hős megfigyelt szerepgyakorisága, nem biztos szerepkör vagy győzelmi esély. A kiosztás ütközésmentes csapatfelállást keres."
            labels[item.role].setText(f"{self.roles[item.role]}:<br><span style='font-size: 10px; color: {color};'>{caption}</span>")
            labels[item.role].setToolTip(tooltip + "\n" + ROLE_SWAP_HINT)
            combos[item.role].setToolTip(tooltip + "\n" + ROLE_SWAP_HINT)

    def _confirm_manual_role(self, team, role):
        if self._sync_applying or not self.chk_follow.isChecked() or self._sync_last_snapshot is None or self._busy:
            return
        cid = self._get_ids([self.blue_picks[role] if team == "blue" else self.red_picks[role]])[0]
        if cid < 0 or not any(pick.internal_id == cid for pick in self._sync_last_snapshot.picks):
            return
        self._sync_manual_roles[(team, cid)] = role
        self._receive_followed_draft(self._sync_last_snapshot, force=True)

    def _display_explanations(self, records):
        from view.explanation_dialog import ExplanationDialog
        self.lbl_dash_title.setText("A predikció fókuszhősének elemzése elkészült." if self._prediction_context is not None
                                    else "Az ajánlások részletes elemzése elkészült.")
        ExplanationDialog(records, {v: k for k, v in self.name_map.items()}, self).exec()

    def _choose_lockfile(self):
        path, _ = QFileDialog.getOpenFileName(self, "A LoL-kliens lockfile fájlja",
                                               str(self._lockfile_path or preferred_lockfile()),
                                               "Lockfile (lockfile);;Minden fájl (*)")
        if path:
            following = self.chk_follow.isChecked()
            self.chk_follow.setChecked(False)
            self._lockfile_path = path
            saved = save_lockfile_path(path)
            self.btn_lockfile.setToolTip(f"Kiválasztott LoL lockfile: {path}")
            self.btn_import.setToolTip("Beolvasás a gépen elmentett kliensből." if saved
                                       else "Beolvasás a kiválasztott kliensből. Az útvonal most nem menthető.")
            self._draft_unavailable()
            if following:
                self.chk_follow.setChecked(True)

    def _on_import_draft(self):
        if self._busy:
            return
        self._manual_import_active = True
        self._sync_pending = None
        self._draft_follower.set_enabled(False)
        if self.chk_follow.isChecked():
            self._set_sync_status("Az automatikus követés a kézi draftellenőrzés alatt szünetel.")
        self._reset_dash()
        self.lbl_dash_title.setText("A LoL-kliens hősválasztásának beolvasása…")
        def read_client():
            from services.draft_import import LcuDraftClient
            return LcuDraftClient(lockfile_path=self._lockfile_path).read_draft()
        self._start_task(read_client, self._review_import)

    def _review_import(self, snapshot):
        from view.draft_import_dialog import DraftImportDialog
        from services.draft_assignment import assign_draft
        document, _ = self._load_roles()
        original = snapshot
        if snapshot.session_key and snapshot.session_key != self._sync_session_key:
            self._sync_manual_roles = {}
            self._sync_manual_side = None
        if snapshot.own_team and self._sync_manual_side != snapshot.own_team:
            self._sync_manual_side = None
        preview = None
        tier = self.cb_rank.currentText()
        tier = "MIXED" if tier == "ALL RANKS" else tier
        try:
            preview = assign_draft(snapshot, document, own_side=snapshot.own_team or self.cb_sync_side.currentData() or self._sync_manual_side,
                                   previous_roles=self._sync_manual_roles, tier=tier)
        except ValueError:
            pass  # The dialog can still resolve an unknown side manually.
        if preview is not None:
            metadata = {(item.team, item.internal_id): item for item in preview.assignments}
            own = snapshot.own_team or self.cb_sync_side.currentData() or self._sync_manual_side
            def absolute(pick):
                return own if pick.team == "own" else ("red" if own == "blue" else "blue") if pick.team == "enemy" else pick.team
            picks = tuple(replace(pick, role=metadata[(absolute(pick), pick.internal_id)].role) for pick in snapshot.picks)
            notes = tuple(note for note in preview.warnings if preview.unresolved or not note.startswith("A nem közölt szerepköröket"))
            inferred = [item for item in preview.assignments if item.source == "statistics"]
            if inferred:
                notes += ("A szerepkörök egy része adatbázis-statisztikából becsült; alkalmazás előtt ellenőrizheted és javíthatod.",)
            snapshot = replace(snapshot, picks=picks, warnings=notes)
        self.btn_lockfile.setToolTip(f"A gépen elmentett LoL lockfile helye: {preferred_lockfile()}")
        dialog = DraftImportDialog(snapshot, {v: k for k, v in self.name_map.items()}, self)
        fallback_side = self.cb_sync_side.currentData() or self._sync_manual_side
        if original.own_team is None and fallback_side is not None:
            dialog.own_side.setCurrentIndex(dialog.own_side.findData(fallback_side))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.lbl_dash_title.setText("A kliens beolvasása megszakítva.")
            return
        values = dialog.draft_values()
        if original.own_team and dialog.own_side.currentData() != original.own_team:
            raise ValueError("A választott oldal ellentmond a kliens által közölt saját csapatnak.")
        self._sync_manual_side = dialog.own_side.currentData()
        self._sync_session_key = original.session_key
        try:
            preview = assign_draft(original, document, own_side=self._sync_manual_side, tier=tier)
            expected = {(item.team, item.internal_id): item.role for item in preview.assignments}
            previous_manual = self._sync_manual_roles
            self._sync_manual_roles = {(team, cid): role for team in ("blue", "red")
                for role, cid in enumerate(values[team]) if cid is not None
                and (expected.get((team, cid)) != role or previous_manual.get((team, cid)) == role)}
        except ValueError:
            self._sync_manual_roles = {(team, cid): role for team in ("blue", "red")
                for role, cid in enumerate(values[team]) if cid is not None}
        self._apply_import(values)
        if preview is not None:
            chosen = {(team, cid): role for team in ("blue", "red") for role, cid in enumerate(values[team])
                      if cid is not None}
            metadata = []
            for item in preview.assignments:
                role = chosen.get((item.team, item.internal_id))
                if role is not None:
                    if role != item.role or (item.source != "client" and self._sync_manual_roles.get((item.team, item.internal_id)) == role):
                        item = replace(item, role=role, source="manual",
                                       role_probability=item.probabilities[role] if item.sample_size else None)
                    metadata.append(item)
            self._show_role_metadata(metadata)

    def _apply_import(self, values):
        """Apply only the reviewed snapshot, replacing the previous draft."""
        names = {v: k for k, v in self.name_map.items()}
        for key, combos in (("blue", self.blue_picks), ("red", self.red_picks),
                            ("blue_bans", self.blue_bans), ("red_bans", self.red_bans)):
            ids = values.get(key, [])
            for index, combo in enumerate(combos):
                champion_id = ids[index] if index < len(ids) else None
                name = names[champion_id] if champion_id is not None else ""
                combo.setCurrentIndex(combo.findText(name, Qt.MatchFlag.MatchExactly))
        self._invalidate_recommendations()
        self._reset_dash()
        self.lbl_dash_title.setText("A draft frissült.\nKérj ajánlást vagy predikciót.")

    def closeEvent(self, event):
        self._flush_ui_preferences()
        self._preferences_suspended = True
        self.chk_follow.setChecked(False)
        if self._busy or self._draft_follower.running:
            self._preferences_suspended = False
            event.ignore()
            self.lbl_dash_title.setText("A folyamat befejezése után bezárható az ablak.")
        else:
            super().closeEvent(event)
            
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
