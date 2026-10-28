"""Optional detailed view for the first five champion recommendations."""

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout
from services.explanation_format import explanations_html


class ExplanationDialog(QDialog):
    def __init__(self, records, champion_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Miért ezeket a hősöket ajánlja?")
        self.resize(850, 720)
        self.setStyleSheet("QDialog { background: #20202A; } "
                           "QTextBrowser { background: #181820; color: #E0E0E0; "
                           "font-size: 14px; padding: 12px; }")
        layout = QVBoxLayout(self)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        self.details.setHtml(explanations_html(records, champion_names))
        layout.addWidget(self.details)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.button(QDialogButtonBox.StandardButton.Close).setText("Bezárás")
        close.rejected.connect(self.reject)
        layout.addWidget(close)
