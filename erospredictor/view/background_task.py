"""Run one application operation without blocking the Qt event loop."""

from PyQt6.QtCore import QThread, pyqtSignal


class BackgroundTask(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self):
        try:
            self.succeeded.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
