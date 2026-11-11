"""Scroll the whole recommendation document instead of sizing a long label."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QTextBrowser


class RecommendationList(QTextBrowser):
    linkActivated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._html = ""
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(lambda url: self.linkActivated.emit(url.toString()))

    def setText(self, html):
        position = self.verticalScrollBar().value()
        self._html = html
        self.setHtml(html)
        self.verticalScrollBar().setValue(position)

    def text(self):
        return self._html

    def clear(self):
        self._html = ""
        super().clear()
