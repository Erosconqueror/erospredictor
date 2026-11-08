"""Two reversible arrangements of the same controls and draft state."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWIDGETSIZE_MAX


def apply_layout(window, compact):
    compact = bool(compact)
    if getattr(window, "_compact_hud_active", None) == compact:
        return
    window._compact_hud_active = compact
    window.setUpdatesEnabled(False)
    try:
        window.header_spacer.show()
        window.header_spacer.setFixedWidth(80 if compact else 270)
        window.header_title.setText("Erospredictor  ·  Draft asszisztens" if compact else
                                   "League of Legends Hősválasztási Segédprogram")
        window.header_title.setStyleSheet(
            "font-size: 23px; font-weight: 800; color: white; margin: 5px;" if compact else
            "font-size: 28px; font-weight: 900; letter-spacing: 1px; margin: 10px; color: white;")
        window.header_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.header.layout().setStretch(1, 1)
        window.header.layout().setAlignment(window.header_title, Qt.AlignmentFlag.AlignVCenter)
        window.header.layout().setContentsMargins(0, 0, 0 if compact else 7, 0)
        window.main_layout.removeWidget(window.controls_panel)
        window.controls_panel.layout().removeWidget(window.btn_predict)
        window.action_row.removeWidget(window.btn_predict)
        window.rank_toolbar.removeWidget(window.btn_clear)
        window.clear_panel.removeWidget(window.btn_clear)
        window.team_row.removeItem(window.role_row)
        window.focus_panel.layout().removeItem(window.role_row)
        window.focus_panel.layout().removeItem(window.action_row)
        if window.pool_row.count() == 5:
            window._hud_pool_spacer = window.pool_row.takeAt(4)
        window.controls_panel.layout().removeItem(window.action_row)
        window.action_row.removeWidget(window.btn_explain)
        for widget in (window.btn_explain, window.lbl_effects_hint):
            window.focus_panel.layout().removeWidget(widget)
            window.controls_panel.layout().removeWidget(widget)
        window.bans_panel.show()
        window.main_layout.addWidget(window.controls_panel)
        for column in (window.blue_team_layout, window.red_team_layout):
            column.setAlignment(Qt.AlignmentFlag.AlignTop if compact else Qt.AlignmentFlag(0))
            column.setSpacing(8 if compact else -1)
        if compact:
            window.focus_panel.layout().insertLayout(1, window.role_row)
            window.action_row.insertWidget(0, window.btn_predict)
            window.action_row.addWidget(window.btn_explain)
            window.controls_panel.layout().addLayout(window.action_row)
            window.controls_panel.layout().addWidget(window.lbl_effects_hint)
            window.rank_toolbar.addWidget(window.btn_clear)
            window.focus_panel.setTitle("")
            window.focus_panel.setStyleSheet("QGroupBox { border: none; margin-top: 0; padding-top: 0; }")
            window.focus_panel.setFixedWidth(1000)
            for button in (window.btn_predict, window.btn_rec, window.btn_bans,
                           window.btn_options, window.btn_explain):
                button.setFixedSize(152, 40)
            for button in window.grp_role.buttons():
                button.setFixedSize(152, 38)
            window.btn_explain.setText("Miért?")
            window.action_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            window.btn_options.setText("Beállítások…")
            window.btn_blue.setText("Kék")
            window.btn_red.setText("Piros")
            window.btn_blue.setFixedSize(160, 35)
            window.btn_red.setFixedSize(160, 35)
            for row in (window.team_row, window.role_row, window.pool_row, window.action_row):
                row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                row.setSpacing(8)
            window.pool_row.itemAt(0).widget().setFixedWidth(60)
            window.chk_pool_only.setFixedWidth(150)
            window.btn_pool.setFixedWidth(150)
            window.controls_bottom.setStretch(0, 1)
            window.controls_bottom.setStretch(1, 0)
            window.controls_bottom.setStretch(2, 1)
            window.controls_panel.layout().setSpacing(8)
            window.wr_container.layout().setContentsMargins(10, 20, 10, 5)
            window.setMinimumSize(1180, 820)
        else:
            window.pool_row.addItem(window._hud_pool_spacer)
            window.focus_panel.layout().insertLayout(1, window.role_row)
            window.focus_panel.layout().insertLayout(3, window.action_row)
            window.focus_panel.layout().addWidget(window.btn_explain)
            window.focus_panel.layout().addWidget(window.lbl_effects_hint)
            window.controls_panel.layout().insertWidget(0, window.btn_predict,
                                                        alignment=Qt.AlignmentFlag.AlignHCenter)
            window.clear_panel.addWidget(window.btn_clear,
                alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
            window.focus_panel.setTitle("Hős Ajánló")
            window.focus_panel.setStyleSheet("")
            window.focus_panel.setFixedWidth(700)
            window.btn_predict.setFixedSize(300, 55)
            window.btn_rec.setFixedSize(250, 40)
            window.btn_explain.setMinimumWidth(0)
            window.btn_explain.setMaximumWidth(QWIDGETSIZE_MAX)
            window.btn_explain.setMinimumHeight(0)
            window.btn_explain.setMaximumHeight(QWIDGETSIZE_MAX)
            for button in (window.btn_bans, window.btn_options):
                button.setMinimumWidth(0)
                button.setMaximumWidth(QWIDGETSIZE_MAX)
            window.action_row.setAlignment(Qt.AlignmentFlag(0))
            window.btn_options.setText("Beállítások / előzmények…")
            window.btn_blue.setText("Kék Csapat (Blue)")
            window.btn_red.setText("Piros Csapat (Red)")
            window.btn_blue.setFixedWidth(160)
            window.btn_red.setFixedWidth(160)
            for button in window.grp_role.buttons():
                button.setFixedSize(115, 38)
            for row in (window.team_row, window.role_row, window.pool_row):
                row.setAlignment(Qt.AlignmentFlag(0))
                row.setSpacing(-1)
            for widget in (window.pool_row.itemAt(0).widget(), window.chk_pool_only, window.btn_pool):
                widget.setMinimumWidth(0)
                widget.setMaximumWidth(QWIDGETSIZE_MAX)
            window.controls_bottom.setStretch(0, 1)
            window.controls_bottom.setStretch(1, 0)
            window.controls_bottom.setStretch(2, 1)
            window.controls_panel.layout().setSpacing(12)
            window.wr_container.layout().setContentsMargins(10, 80, 10, 5)
            window.setMinimumSize(1180, 920)
        style_workspace(window, compact)
        window._update_explain_button()
    finally:
        window.setUpdatesEnabled(True)


def style_workspace(window, compact):
    """Mirror the two teams around the result panel, keeping real widgets."""
    window.teams_layout.setSpacing(12 if compact else 6)
    for side, card, column, title, labels, badges, picks, color, background in (
        ("blue", window.blue_team_card, window.blue_team_layout, window.blue_team_title,
         window.blue_role_labels, window.blue_effects, window.blue_picks, "#5CADEF", "#182331"),
        ("red", window.red_team_card, window.red_team_layout, window.red_team_title,
         window.red_role_labels, window.red_effects, window.red_picks, "#F08088", "#291F2B"),
    ):
        card.setStyleSheet("")
        if compact:
            border = "#30465E" if side == "blue" else "#50333F"
            card.setStyleSheet(f"QFrame#{side}TeamCard {{ background: {background}; border: 1px solid {border}; border-radius: 12px; }}")
        column.setContentsMargins(*(12, 14, 12, 14) if compact else (0, 0, 0, 0))
        title.setText(("KÉK CSAPAT" if side == "blue" else "PIROS CSAPAT") if compact else
                      ("Kék Csapat (Blue Team)" if side == "blue" else "Piros Csapat (Red Team)"))
        title.setStyleSheet(f"color: {color}; font-size: {'15' if compact else '18'}px; font-weight: bold; padding-bottom: 8px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.setAlignment(title, Qt.AlignmentFlag.AlignVCenter if compact else Qt.AlignmentFlag.AlignCenter)
        for role, (label, badge, pick) in enumerate(zip(labels, badges, picks)):
            row = column.itemAt(role + 1).layout()
            for widget in (label, badge, pick):
                row.removeWidget(widget)
            widgets = ((badge, label, pick) if side == "blue" else (pick, label, badge)) if compact else (
                (badge, label, pick) if side == "blue" else (label, pick, badge))
            for widget in widgets:
                row.addWidget(widget)
            row.setSpacing(6)
            row.setAlignment(Qt.AlignmentFlag.AlignHCenter if compact else Qt.AlignmentFlag(0))
            label.setFixedWidth(64 if compact else 65)
            label.setStyleSheet(f"font-weight: bold; color: #A6AEC1; font-size: {13 if compact else 15}px;")
            label.setAlignment((Qt.AlignmentFlag.AlignRight if side == "blue" else Qt.AlignmentFlag.AlignLeft)
                               | Qt.AlignmentFlag.AlignVCenter if compact else Qt.AlignmentFlag.AlignVCenter)
            badge.setFixedWidth(72 if compact else 92)
            pick.setStyleSheet("QComboBox { font-size: 14px; border: 1px solid #3C4558; background: #252C3A; border-radius: 8px; padding: 5px; color: #F0F3FA; }"
                              "QComboBox QLineEdit { background: transparent; border: none; padding: 0; }"
                              "QComboBox:hover { border-color: #64738C; } QComboBox:focus { border-color: white; }"
                              if compact else "font-size: 14px; padding: 3px;")
    for combos, border, background in ((window.blue_bans, "#345A78", "#192734"),
                                       (window.red_bans, "#70414D", "#2C202A")):
        for combo in combos:
            combo.setStyleSheet(f"QComboBox {{ font-size: 14px; padding: 3px; border: 1px solid {border}; background: {background}; border-radius: 6px; }}"
                               "QComboBox QLineEdit { background: transparent; border: none; padding: 0; }"
                               "QComboBox:hover { border-color: #8190A8; } QComboBox:focus { border-color: white; }"
                               if compact else "font-size: 14px; padding: 3px;")
