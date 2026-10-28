"""Review local-client picks before replacing the editable draft."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHeaderView, QLabel,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout,
)
from pathlib import Path


class DraftImportDialog(QDialog):
    """Unknown roles and uncertain sides require an explicit user assignment."""

    def __init__(self, snapshot, champion_names, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot
        self.champion_names = {int(key): value for key, value in champion_names.items()}
        self.team_selectors = []
        self.role_selectors = []
        self.setWindowTitle("Hősök beolvasása a LoL-kliensből")
        self.resize(780, 590)
        self.setStyleSheet("""
            QDialog { background-color: #20202A; }
            QLabel { color: #E0E0E0; }
            QTableWidget { background-color: #292934; color: #E0E0E0; gridline-color: #444450; }
            QHeaderView::section { background-color: #363645; color: #E0E0E0; padding: 6px; }
            QComboBox { background-color: #363645; color: #E0E0E0; padding: 5px; }
            QComboBox QAbstractItemView { background-color: #292934; color: #E0E0E0; }
            QPushButton { padding: 7px 14px; }
        """)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Csak a már rögzített választások kerülnek beolvasásra. Ellenőrizd a csapatokat és a pozíciókat. "
            "Az ismeretlen szerepköröket rendeld hozzá kézzel, vagy válaszd a Kihagyás lehetőséget."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("A saját csapatom oldala:"))
        self.own_side = QComboBox()
        self.own_side.addItem("Válassz oldalt…", None)
        self.own_side.addItem("Kék", "blue")
        self.own_side.addItem("Piros", "red")
        if snapshot.own_team:
            self.own_side.setCurrentIndex(self.own_side.findData(snapshot.own_team))
        side_row.addWidget(self.own_side)
        side_row.addStretch()
        layout.addLayout(side_row)
        if snapshot.warnings:
            notes = QLabel("\n".join(snapshot.warnings))
            notes.setWordWrap(True)
            notes.setStyleSheet("color: #e4ac43;")
            layout.addWidget(notes)
        self.table = QTableWidget(len(snapshot.picks), 4)
        self.table.setHorizontalHeaderLabels(["Hős", "Forrás", "Csapat", "Pozíció"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        for row, pick in enumerate(snapshot.picks):
            name = self.champion_names.get(pick.internal_id, f"Ismeretlen ({pick.riot_id})")
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setIcon(QIcon(str(Path(__file__).resolve().parents[1] / f"assets/icons/{pick.internal_id}.png")))
            self.table.setItem(row, 0, item)
            source = "Saját csapat" if self._is_own(pick) else "Ellenfél"
            source_item = QTableWidgetItem(source)
            source_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 1, source_item)
            team_combo = QComboBox()
            team_combo.addItem("Válassz…", None)
            team_combo.addItem("Kék", "blue")
            team_combo.addItem("Piros", "red")
            self.table.setCellWidget(row, 2, team_combo)
            self.team_selectors.append(team_combo)
            role_combo = QComboBox()
            role_combo.addItem("Válassz pozíciót…", None)
            for role, role_name in enumerate(("Top", "Jungle", "Mid", "ADC", "Support")):
                role_combo.addItem(role_name, role)
            role_combo.addItem("Kihagyás", -1)
            if pick.role is not None:
                role_combo.setCurrentIndex(role_combo.findData(pick.role))
            self.table.setCellWidget(row, 3, role_combo)
            self.role_selectors.append(role_combo)
            team_combo.currentIndexChanged.connect(self._refresh_validation)
            role_combo.currentIndexChanged.connect(self._refresh_validation)
        self.ban_summary = QLabel()
        self.ban_summary.setWordWrap(True)
        layout.addWidget(self.ban_summary)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        replace_note = QLabel("Alkalmazáskor a jelenlegi választásokat és tiltásokat ez a beolvasott draft váltja fel.")
        replace_note.setWordWrap(True)
        layout.addWidget(replace_note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Draft kitöltése")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Mégse")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.own_side.currentIndexChanged.connect(self._set_teams)
        self._set_teams()

    def _is_own(self, pick):
        return pick.team == "own" or (self.snapshot.own_team is not None and pick.team == self.snapshot.own_team)

    def _set_teams(self, *_):
        own = self.own_side.currentData()
        for pick, combo in zip(self.snapshot.picks, self.team_selectors):
            side = None if own is None else (own if self._is_own(pick) else ("red" if own == "blue" else "blue"))
            combo.setCurrentIndex(combo.findData(side))
        self._refresh_validation()

    def _bans(self):
        own, enemy = list(self.snapshot.own_bans), list(self.snapshot.enemy_bans)
        # Also accept snapshots created by integrations supplying absolute sides.
        if not own and not enemy and self.snapshot.own_team:
            own = list(self.snapshot.blue_bans if self.snapshot.own_team == "blue" else self.snapshot.red_bans)
            enemy = list(self.snapshot.red_bans if self.snapshot.own_team == "blue" else self.snapshot.blue_bans)
        return (own, enemy) if self.own_side.currentData() == "blue" else (enemy, own)

    def draft_values(self):
        """Return None for empty slots; internal champion 0 is Aatrox, not empty."""
        if self.own_side.currentData() is None:
            raise ValueError("Válaszd ki a saját csapatod Kék vagy Piros oldalát.")
        blue_bans, red_bans = self._bans()
        values = {"blue": [None] * 5, "red": [None] * 5,
                  "blue_bans": blue_bans, "red_bans": red_bans}
        used = set()
        for pick, team_combo, role_combo in zip(self.snapshot.picks, self.team_selectors, self.role_selectors):
            role = role_combo.currentData()
            if role == -1:
                continue
            team = team_combo.currentData()
            if team not in ("blue", "red") or role is None:
                raise ValueError("Minden megtartott hősnél válassz csapatot és pozíciót.")
            if values[team][role] is not None:
                raise ValueError("Egy csapatban egy pozícióhoz csak egy hőst rendelhetsz.")
            if pick.internal_id in used:
                raise ValueError("Ugyanaz a hős nem szerepelhet kétszer a draftban.")
            if pick.internal_id in blue_bans + red_bans:
                raise ValueError("Kiválasztott hős nem szerepelhet a tiltások között.")
            values[team][role] = pick.internal_id
            used.add(pick.internal_id)
        if not used and not blue_bans and not red_bans:
            raise ValueError("Még nincs beolvasható rögzített választás vagy tiltás.")
        return values

    def _refresh_validation(self, *_):
        if not hasattr(self, "buttons"):
            return
        blue_bans, red_bans = self._bans()
        names = lambda values: ", ".join(self.champion_names.get(value, str(value)) for value in values) or "–"
        if self.own_side.currentData() is None:
            self.ban_summary.setText(f"Saját tiltások: {names(self.snapshot.own_bans)}\nEllenfél tiltásai: {names(self.snapshot.enemy_bans)}")
        else:
            self.ban_summary.setText(f"Kék tiltások: {names(blue_bans)}\nPiros tiltások: {names(red_bans)}")
        try:
            self.draft_values()
            self.status.setText("A draft kitölthető. A pozíciókat később a főablakban is módosíthatod.")
            self.status.setStyleSheet("color: #72b77a;")
            enabled = True
        except ValueError as error:
            self.status.setText(str(error))
            self.status.setStyleSheet("color: #e4ac43;")
            enabled = False
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enabled)
