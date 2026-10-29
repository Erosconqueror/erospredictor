"""Role-specific ban lists with explicit population/personal evidence."""

from html import escape

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
                             QTextBrowser, QVBoxLayout)

from services.ban_advisor import MIN_MATCHUP_SAMPLE, MIN_PERSONAL_SAMPLE


class BanAdvisorDialog(QDialog):
    role_changed = pyqtSignal(int)

    def __init__(self, lists, names, roles, initial_role, title, history_note, parent=None):
        super().__init__(parent)
        self.lists, self.names, self.roles = lists, names, roles
        self.setWindowTitle(title)
        self.resize(850, 690)
        layout = QVBoxLayout(self)
        intro = QLabel("Válaszd ki, melyik pozícióra kérsz bant. A lista a kiválasztott hőspool lane-párosításait vizsgálja. "
            f"Legalább {MIN_PERSONAL_SAMPLE} saját meccsnél a kiválasztott pozícióban, ugyanazon lane-ellenfél ellen az általános és a személyes győzelmi arány fele-fele súlyt kap.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Pozíció:"))
        self.cb_role = QComboBox()
        for index, role in enumerate(roles):
            self.cb_role.addItem(role, index)
        self.cb_role.setCurrentIndex(initial_role)
        row.addWidget(self.cb_role)
        row.addStretch()
        layout.addLayout(row)
        self.history_notice = QLabel(history_note)
        self.history_notice.setTextFormat(Qt.TextFormat.PlainText)
        self.history_notice.setWordWrap(True)
        layout.addWidget(self.history_notice)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        layout.addWidget(self.details, 1)
        note = QLabel("A százalékok meccses győzelmi arányok és ezek súlyozott mutatói, nem ban utáni esélyek. "
            "Az alacsonyabb összesített érték nehezebb párosítást jelent. A sorrend a mintaméretet, a pool lefedettségét és az ellenfél gyakoriságát is figyelembe veszi.")
        note.setWordWrap(True)
        layout.addWidget(note)
        close = QPushButton("Bezárás")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
        self.cb_role.currentIndexChanged.connect(self._role_changed)
        self._render()

    def _role_changed(self):
        self._render()
        self.role_changed.emit(self.cb_role.currentData())

    def _render(self):
        role = self.cb_role.currentData()
        records = self.lists.get(role, [])
        html = f"<h2>{escape(self.roles[role])} · Top {len(records)} banötlet</h2>"
        if not records:
            html += (f"<p>Nincs elég adat vagy hátrányos párosítás ezen a pozíción. Az általános alaphoz "
                     f"párosításonként legalább {MIN_MATCHUP_SAMPLE} meccs kell. A már pickelt, bannolt és a saját poolba tartozó hősök kimaradnak.</p>")
        for index, item in enumerate(records, 1):
            name = escape(self.names.get(item["id"], str(item["id"])))
            own = item["personal"]
            games = own["games"] if own else 0
            own_rate = f"{own['winrate']:.1f}% · {own['wins']}/{games} győzelem" if own else "Nincs megfelelő saját meccs"
            combined = item["personal_weight"] > 0
            weighting = "50% általános + 50% saját" if combined else f"100% általános · saját minta: {games}/{MIN_PERSONAL_SAMPLE}"
            html += f"<h3>{index}. {name}</h3><table cellpadding='6' cellspacing='0'>"
            html += ("<tr><th align='left'>Általános (pool)</th><th align='left'>Saját (pool / pozíció)</th><th align='left'>Összesített mutató</th></tr>"
                     f"<tr><td>{item['global_winrate']:.1f}% · {item['global_matches']} megfigyelés</td>"
                     f"<td>{own_rate}</td><td><b>{item['combined_winrate']:.1f}%</b></td></tr></table>"
                     f"<p><b>{weighting}</b><br>Mintabeli pozíciós gyakoriság: {item['sample_frequency']:.1f}%. "
                     f"Adat a pool {item['covered']}/{item['pool_size']} hőséhez.</p><ul>")
            for pair in item["matchups"]:
                champion = escape(self.names.get(pair["champion"], str(pair["champion"])))
                html += f"<li>{champion}: {pair['winrate']:.1f}% általános győzelmi arány · {pair['matches']} meccs</li>"
            html += "</ul>"
        self.details.setHtml(html)
