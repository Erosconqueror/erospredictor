"""Optional visual composition review, editable profiles and tag filters."""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QGridLayout, QProgressBar, QComboBox, QPushButton, QScrollArea,
    QDoubleSpinBox, QFormLayout, QCheckBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView)
from PyQt6.QtCore import Qt
from services.champion_traits import LABELS, team_summary
from services.update_progress import atomic_json


def scroll(content):
    content.setObjectName("traitPanel")
    content.setStyleSheet("QWidget#traitPanel { background: #20202A; } QDoubleSpinBox { background: #292934; color: #E0E0E0; border: 1px solid #444450; padding: 4px; }")
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(content)
    return area


class TraitDialog(QDialog):
    def __init__(self, window, champion_id=None):
        super().__init__(window)
        self.window = window
        self.library = window._get_trait_library()
        inverse = {cid: riot for riot, cid in window._pool_mapping.items()}
        opposite = window.red_picks if window.grp_team.checkedId() == 0 else window.blue_picks
        self.library = self.library.for_opponents([inverse[cid] for cid in window._get_ids(opposite) if cid in inverse])
        self.setWindowTitle("Kísérleti csapatelemzés · a winrate változatlan")
        self.resize(900, 720)
        layout = QVBoxLayout(self)
        note = QLabel("Kísérleti profilok és mért adatok • Ismeretlen ≠ gyenge • A winrate változatlan")
        layout.addWidget(note)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        teams = QWidget()
        grid = QGridLayout(teams)
        summaries = []
        tier = window.cb_rank.currentText()
        self.tier = "MIXED" if tier == "ALL RANKS" else tier
        inverse = {cid: riot for riot, cid in window._pool_mapping.items()}
        for col, (side, picks, color) in enumerate((("Kék", window.blue_picks, "#42A5F5"),
                                                   ("Piros", window.red_picks, "#EF8088"))):
            other = window.red_picks if side == "Kék" else window.blue_picks
            contextual = self.library.for_opponents([inverse[cid] for cid in window._get_ids(other) if cid in inverse])
            profiles = [contextual.profile(inverse[cid], role, self.tier)
                        for role, cid in enumerate(window._get_ids(picks)) if cid in inverse]
            summary = team_summary(profiles)
            summaries.append(summary)
            grid.addWidget(QLabel(f"<b>{side} csapat</b> · {len(profiles)}/5 hős"), 0, col*2, 1, 2)
            for row, (key, caption) in enumerate(LABELS.items(), 1):
                value = summary[key]
                grid.addWidget(QLabel(caption), row, col*2)
                bar = QProgressBar()
                bar.setRange(0, 50)
                score = value["score"]
                bar.setValue(round(score*10) if score is not None else 0)
                bar.setFormat(f"{score:.1f}/5 · {value['known']}/5 ismert" if score is not None else "Nincs adat")
                bar.setStyleSheet("QProgressBar { background: #242936; border: 1px solid #485064; border-radius: 3px; color: white; text-align: center; min-height: 22px;}"
                                  f"QProgressBar::chunk {{ background: {color}; width: 12px; margin: 1px; border-radius: 0; border-top-left-radius: 0; border-bottom-left-radius: 0; }}")
                bar.setToolTip("Kísérleti profilösszesítés, nem győzelmi esély. A hiányzó profil nem nulla érték.")
                grid.addWidget(bar, row, col*2+1)
            mix = " · ".join(f"{key}: {value}" for key, value in summary["range_mix"].items())
            grid.addWidget(QLabel(mix), len(LABELS)+1, col*2, 1, 2)
            strengths = [LABELS[key] for key in LABELS if (summary[key]["score"] or 0) >= 4]
            weak = [LABELS[key] for key in LABELS if summary[key]["score"] is not None and summary[key]["score"] < 2.5]
            caption = "Erősség: " + (", ".join(strengths[:2]) or "—") + "\nHiány: " + (", ".join(weak[:2]) or "—")
            grid.addWidget(QLabel(caption), len(LABELS)+2, col*2, 1, 2)
            damage = summary["damage_mix"]
            text = (f"Sebzésarány: fizikai {damage['physical']:.0%} · mágikus {damage['magic']:.0%} · true {damage['true']:.0%} ({summary['damage_known']}/5 ismert)"
                    if damage else "Sebzésarány: még nincs elegendő minta")
            damage_label = QLabel(text)
            damage_label.setWordWrap(True)
            grid.addWidget(damage_label, len(LABELS)+3, col*2, 1, 2)
        tabs.addTab(scroll(teams), "Csapatok")
        filters = QWidget()
        form = QVBoxLayout(filters)
        self.checks = {}
        selected = window._trait_filters
        for key, caption in LABELS.items():
            checkbox = QCheckBox(caption)
            checkbox.setChecked(key in selected)
            self.checks[key] = checkbox
            form.addWidget(checkbox)
        self.threshold = QComboBox()
        self.threshold.addItems([str(i) for i in range(1, 6)])
        self.threshold.setCurrentText(str(window._trait_minimum))
        form.addWidget(QLabel("Elvárt minimum (1–5):"))
        form.addWidget(self.threshold)
        self.mode = QCheckBox("Minden kijelölt tulajdonság szükséges")
        self.mode.setChecked(window._trait_match_all)
        form.addWidget(self.mode)
        apply = QPushButton("Szűrők alkalmazása")
        apply.clicked.connect(self.apply_filters)
        form.addWidget(apply)
        tabs.addTab(scroll(filters), "Ajánlási tagszűrők")
        editor = QWidget()
        edit = QFormLayout(editor)
        self.champion = QComboBox()
        for name in sorted(window.name_map):
            cid = window.name_map[name]
            if cid in inverse:
                self.champion.addItem(name, inverse[cid])
        self.role = QComboBox()
        self.role.addItems(window.roles)
        self.role.setCurrentIndex(window.grp_role.checkedId())
        edit.addRow("Hős", self.champion)
        edit.addRow("Pozíció", self.role)
        self.controls = {}
        for key, caption in LABELS.items():
            spin = QDoubleSpinBox()
            spin.setRange(0, 5)
            spin.setSingleStep(.5)
            spin.setSpecialValueText("Nincs adat")
            self.controls[key] = spin
            edit.addRow(caption, spin)
        self.evidence = QLabel()
        self.evidence.setWordWrap(True)
        edit.addRow(self.evidence)
        self.metric_widget = QWidget()
        self.metric_layout = QFormLayout(self.metric_widget)
        edit.addRow(self.metric_widget)
        save = QPushButton("Hősprofil mentése ezen a gépen")
        save.clicked.connect(self.save_profile)
        edit.addRow(save)
        self.champion.currentIndexChanged.connect(self.load_profile)
        self.role.currentIndexChanged.connect(self.load_profile)
        tabs.addTab(scroll(editor), "Hősprofilok és adatok")
        self.mobility_table = QTableWidget(0, 4)
        self.mobility_table.setHorizontalHeaderLabels(["Hős", "Harci mobilitás", "Map presence", "Besorolás"])
        self.mobility_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.mobility_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.mobility_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.mobility_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.mobility_table.setStyleSheet("QTableWidget { background:#20202A; color:#E0E0E0; gridline-color:#444450; } "
            "QHeaderView::section { background:#343440; color:#E0E0E0; padding:6px; }")
        self.mobility_table.setToolTip("Kísérleti képességprofilok, nem mért teljesítmény. Súgó: indoklás és forrás. Dupla kattintás: hősprofil szerkesztése.")
        self.mobility_table.cellDoubleClicked.connect(lambda row, _col: self._edit_mobility_row(row, tabs))
        tabs.addTab(self.mobility_table, "Mobilitáslista")
        self.role.currentIndexChanged.connect(self.refresh_mobility_table)
        self.refresh_mobility_table()
        self.load_profile()
        if champion_id in inverse:
            self.champion.setCurrentIndex(self.champion.findData(inverse[champion_id]))
            self.role.setCurrentIndex(window.grp_role.checkedId())
            tabs.setCurrentIndex(2)
        close = QPushButton("Bezárás")
        close.clicked.connect(self.accept)
        layout.addWidget(close)

    def refresh_mobility_table(self):
        table = self.mobility_table
        table.setSortingEnabled(False)
        table.setRowCount(self.champion.count())
        for row in range(self.champion.count()):
            riot = self.champion.itemData(row)
            profile = self.library.profile(riot, self.role.currentIndex(), self.tier)
            review = self.library.capabilities.get("champions", {}).get(str(riot), {})
            values = [self.champion.itemText(row)] + [f"{profile['values'][key]:g}/5" if key in profile['values'] else "—"
                      for key in ("mobility", "map_presence")]
            values.append("Ellenfél ellopható R-jétől függ" if review.get("name") == "Sylas" else
                          "Felhasználói pontosítás" if review.get("review") == "user_confirmed" else "Képességalapú, kísérleti")
            if any(profile["sources"].get(key) == "kézzel megadott" for key in ("mobility", "map_presence")):
                values[-1] = "Helyi felülírás"
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, riot)
                item.setToolTip("Pozíció: " + self.role.currentText() + "\n" +
                    "\n".join(profile["sources"].get(key, "Ismeretlen") for key in ("mobility", "map_presence")))
                table.setItem(row, col, item)
        table.setSortingEnabled(True)
        table.sortItems(0)

    def _edit_mobility_row(self, row, tabs):
        riot = self.mobility_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.champion.setCurrentIndex(self.champion.findData(riot))
        tabs.setCurrentIndex(2)

    def load_profile(self):
        profile = self.library.profile(self.champion.currentData(), self.role.currentIndex(), self.tier)
        for key, spin in self.controls.items():
            spin.setValue(profile["values"].get(key, 0))
            spin.setToolTip(profile["sources"].get(key, "Ismeretlen; nem vezethető le megbízhatóan a rendelkezésre álló adatból."))
        self._loaded_values = {key: spin.value() for key, spin in self.controls.items()}
        lines = [f"Megfigyelt minta: {profile['samples']} · patchek: {', '.join(self.library.patches) or 'még nincs'}"]
        lines.append(f"Statikus alapadat: {self.library.basics.get('patch', '—')} · profil: osztálybecslés / kézi pontosítás")
        stats = self.library.basics.get("champions", {}).get(str(self.champion.currentData()), {}).get("stats", {})
        lines.append(f"Frontline alapadatok: HP {stats.get('hp', '—')}, HP/növekedés {stats.get('hpperlevel', '—')}, páncél {stats.get('armor', '—')}")
        lines.append(f"Alaptámadás hatótávja: {profile['attack_range'] or '—'}; külön a harci távolságtól.")
        self.evidence.setText("\n".join(lines))
        while self.metric_layout.rowCount():
            self.metric_layout.removeRow(0)
        captions = {"cc_seconds_per_min": ("Kontroll", "s/perc", 10),
                    "immobilizations_per_min": ("Immobilizálás", "/perc", 3),
                    "damage_per_min": ("Hőssebzés", "/perc", 1500),
                    "mitigated_per_min": ("Enyhített sebzés", "/perc", 3000),
                    "taken_per_min": ("Kapott sebzés", "/perc", 3000),
                    "healing_per_min": ("Gyógyítás összesen", "/perc", 1500),
                    "ally_heal_per_min": ("Csapattárs gyógyítása", "/perc", 1500),
                    "ally_shield_per_min": ("Csapattárs pajzsolása", "/perc", 1500),
                    "gold_per_min": ("Arany", "/perc", 700)}
        for key, caption in (("physical_share", "Fizikai sebzés"), ("magic_share", "Mágikus sebzés"), ("true_share", "True sebzés")):
            captions[key] = (caption, "%", 1)
        for key, (caption, unit, ceiling) in captions.items():
            total, count = profile["metrics"].get(key, (0, 0))
            bar = QProgressBar()
            bar.setRange(0, 100)
            mean = total/count if count else 0
            bar.setValue(round(min(1, mean/ceiling)*100))
            bar.setFormat(f"{mean*(100 if unit=='%' else 1):.1f} {unit} · n={count}" if count else "Nincs adat")
            bar.setStyleSheet("QProgressBar { border: 1px solid #485064; background: #242936; color: white; text-align: center; min-height: 20px;} QProgressBar::chunk {background:#477477; width:10px; margin:1px;}")
            bar.setToolTip("Mért átlag az öt patch érvényes mintáiból; a sáv megjelenítési skálája kísérleti. Nem hőserő vagy winrate-módosító.")
            self.metric_layout.addRow(caption, bar)

    def save_profile(self):
        from services.client_settings import settings_path
        riot = str(self.champion.currentData())
        manual = self.library.overrides.setdefault(riot, {}).setdefault(str(self.role.currentIndex()), {})
        for key, spin in self.controls.items():
            if spin.value() != self._loaded_values[key]:
                manual[key] = spin.value() if spin.value() >= 1 else None
        try:
            atomic_json(settings_path().with_name("champion_trait_profiles.json"), self.library.overrides)
        except OSError:
            self.evidence.setText("A profil most nem menthető ezen a gépen. Próbáld újra.")
            return
        self.window._trait_library_cache = None
        self.window._invalidate_recommendations()
        self.refresh_mobility_table()
        self.evidence.setText("Profil elmentve. Új ajánláskor használja; a winrate-et nem módosítja.")

    def apply_filters(self):
        w = self.window
        w._trait_filters = [key for key, checkbox in self.checks.items() if checkbox.isChecked()]
        w._trait_minimum = int(self.threshold.currentText())
        w._trait_match_all = self.mode.isChecked()
        for key, value in (("trait_filters", w._trait_filters), ("trait_minimum", w._trait_minimum), ("trait_match_all", w._trait_match_all)):
            w._remember_ui_preference(key, value)
        w._invalidate_recommendations()
        self.accept()
