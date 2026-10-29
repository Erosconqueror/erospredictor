"""One editor for played, comfort, learning and named champion lists."""

from copy import deepcopy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from services.champion_pools import ALL_MINE, TAGS, include_history, save_library


class ChampionPoolDialog(QDialog):
    def __init__(self, document, mapping, names, history=None, parent=None):
        super().__init__(parent)
        self.document, self.mapping, self.history = deepcopy(document), mapping, history
        self.rows = sorted(((riot, names[index]) for riot, index in mapping.items() if index in names), key=lambda row: row[1].casefold())
        self.setWindowTitle("Saját hősök és listák")
        self.resize(790, 650)
        layout = QVBoxLayout(self)
        intro = QLabel("A Játszom, Komfort és Tanulnám jelölések együtt is használhatók. Saját listába jelölés nélküli hőst is felvehetsz. A jelölések nem módosítják a modell százalékát.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.lists = QComboBox()
        self.lists.addItems([ALL_MINE] + sorted(document["lists"]))
        self.lists.currentTextChanged.connect(self._list_changed)
        row.addWidget(self.lists, 1)
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Új lista neve…")
        self.new_name.setMaxLength(64)
        row.addWidget(self.new_name)
        add = QPushButton("Létrehozás")
        add.clicked.connect(self._add_list)
        row.addWidget(add)
        self.remove = QPushButton("Lista törlése")
        self.remove.clicked.connect(self._remove_list)
        row.addWidget(self.remove)
        layout.addLayout(row)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Hős keresése…")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        self.table = QTableWidget(len(self.rows), 5)
        self.table.setHorizontalHeaderLabels(["Hős", *TAGS.values(), "Listatag"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.itemChanged.connect(self._item_changed)
        layout.addWidget(self.table)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        history_button = QPushButton("Játszott hősök hozzáadása a betöltött előzményből")
        history_button.setEnabled(history is not None and bool(history.games))
        history_button.setToolTip("Kizárólag a már betöltött saját meccseket használja; új lekérést nem indít.")
        history_button.clicked.connect(self._include_history)
        row.addWidget(history_button)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Mentés")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Mégse")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._current_list = ALL_MINE
        self._populate()

    def _populate(self):
        self.table.blockSignals(True)
        members = set(self.document["lists"].get(self._current_list, []))
        for row, (riot, name) in enumerate(self.rows):
            label = QTableWidgetItem(name)
            label.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 0, label)
            tags = self.document["champions"].get(riot, [])
            for column, tag in enumerate(TAGS, 1):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if tag in tags else Qt.CheckState.Unchecked)
                self.table.setItem(row, column, item)
            item = QTableWidgetItem()
            flags = Qt.ItemFlag.ItemIsUserCheckable
            if self._current_list != ALL_MINE:
                flags |= Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if (bool(tags) if self._current_list == ALL_MINE else riot in members) else Qt.CheckState.Unchecked)
            self.table.setItem(row, 4, item)
        self.table.blockSignals(False)
        self.remove.setEnabled(self._current_list != ALL_MINE)
        self._filter(self.search.text())

    def _capture(self):
        members = []
        for row, (riot, _) in enumerate(self.rows):
            tags = [tag for column, tag in enumerate(TAGS, 1) if self.table.item(row, column).checkState() == Qt.CheckState.Checked]
            if tags:
                self.document["champions"][riot] = tags
            else:
                self.document["champions"].pop(riot, None)
            if self.table.item(row, 4).checkState() == Qt.CheckState.Checked:
                members.append(riot)
        if self._current_list != ALL_MINE:
            # Preserve catalog IDs not available in this application version.
            visible = {riot for riot, _ in self.rows}
            hidden = [riot for riot in self.document["lists"].get(self._current_list, []) if riot not in visible]
            self.document["lists"][self._current_list] = members + hidden

    def _item_changed(self, item):
        if self._current_list == ALL_MINE and 1 <= item.column() <= 3:
            self.table.blockSignals(True)
            checked = any(self.table.item(item.row(), column).checkState() == Qt.CheckState.Checked for column in (1, 2, 3))
            self.table.item(item.row(), 4).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.table.blockSignals(False)

    def _list_changed(self, name):
        self._capture()
        self._current_list = name
        self._populate()

    def _add_list(self):
        name = self.new_name.text().strip()
        if not name or name == ALL_MINE or name in self.document["lists"] or len(self.document["lists"]) >= 50:
            self.status.setText("Adj meg egy új, legfeljebb 64 karakteres listanevet; legfeljebb 50 lista lehet.")
            return
        self._capture()
        self.document["lists"][name] = []
        self.lists.addItem(name)
        self.lists.setCurrentText(name)
        self.new_name.clear()
        self.status.clear()

    def _remove_list(self):
        if self._current_list == ALL_MINE:
            return
        name = self._current_list
        self.lists.setCurrentText(ALL_MINE)
        self.document["lists"].pop(name, None)
        self.lists.removeItem(self.lists.findText(name))

    def _filter(self, text):
        for row, (_, name) in enumerate(self.rows):
            self.table.setRowHidden(row, text.casefold() not in name.casefold())

    def _include_history(self):
        self._capture()
        self.document = include_history(self.document, self.history, self.mapping)
        self._populate()

    def _save(self):
        self._capture()
        try:
            save_library(self.document)
        except (OSError, ValueError):
            self.status.setText("A hőslista most nem menthető. Az ablakban lévő módosítások megmaradtak; próbáld újra.")
            return
        self.accept()
