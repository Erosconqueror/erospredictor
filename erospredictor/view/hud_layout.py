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
        window.header_spacer.setVisible(not compact)
        window.header_title.setText("Erospredictor  ·  Draft asszisztens" if compact else
                                   "League of Legends Hősválasztási Segédprogram")
        window.header_title.setStyleSheet(
            "font-size: 23px; font-weight: 800; color: white; margin: 5px;" if compact else
            "font-size: 28px; font-weight: 900; letter-spacing: 1px; margin: 10px; color: white;")
        window.main_layout.removeWidget(window.controls_panel)
        window.controls_panel.layout().removeWidget(window.btn_predict)
        window.action_row.removeWidget(window.btn_predict)
        window.rank_toolbar.removeWidget(window.btn_clear)
        window.clear_panel.removeWidget(window.btn_clear)
        window.team_row.removeItem(window.role_row)
        window.focus_panel.layout().removeItem(window.role_row)
        window.focus_panel.layout().removeItem(window.action_row)
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
            window.team_row.addLayout(window.role_row, 1)
            window.action_row.insertWidget(0, window.btn_predict)
            window.action_row.addWidget(window.btn_explain)
            window.controls_panel.layout().addLayout(window.action_row)
            window.controls_panel.layout().addWidget(window.lbl_effects_hint)
            window.rank_toolbar.addWidget(window.btn_clear)
            window.focus_panel.setTitle("")
            window.focus_panel.setStyleSheet("QGroupBox { border: none; margin-top: 0; padding-top: 0; }")
            window.focus_panel.setFixedWidth(1000)
            window.btn_predict.setFixedSize(160, 40)
            window.btn_rec.setFixedSize(160, 40)
            window.btn_explain.setFixedWidth(140)
            window.btn_explain.setText("Miért?")
            window.btn_bans.setFixedWidth(120)
            window.btn_options.setFixedWidth(150)
            window.action_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            window.btn_options.setText("Beállítások…")
            window.btn_blue.setText("Kék")
            window.btn_red.setText("Piros")
            window.btn_blue.setFixedWidth(110)
            window.btn_red.setFixedWidth(110)
            window.controls_bottom.setStretch(0, 0)
            window.controls_bottom.setStretch(1, 1)
            window.controls_bottom.setStretch(2, 0)
            window.controls_panel.layout().setSpacing(4)
            window.wr_container.layout().setContentsMargins(10, 20, 10, 5)
            window.setMinimumSize(1180, 820)
        else:
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
            for button in (window.btn_bans, window.btn_options):
                button.setMinimumWidth(0)
                button.setMaximumWidth(QWIDGETSIZE_MAX)
            window.action_row.setAlignment(Qt.AlignmentFlag(0))
            window.btn_options.setText("Beállítások / előzmények…")
            window.btn_blue.setText("Kék Csapat (Blue)")
            window.btn_red.setText("Piros Csapat (Red)")
            window.btn_blue.setFixedWidth(160)
            window.btn_red.setFixedWidth(160)
            window.controls_bottom.setStretch(0, 1)
            window.controls_bottom.setStretch(1, 0)
            window.controls_bottom.setStretch(2, 1)
            window.controls_panel.layout().setSpacing(12)
            window.wr_container.layout().setContentsMargins(10, 80, 10, 5)
            window.setMinimumSize(1180, 920)
        window._update_explain_button()
    finally:
        window.setUpdatesEnabled(True)
