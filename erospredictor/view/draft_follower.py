"""Optional, serialized local-client polling without blocking the Qt event loop."""

from dataclasses import dataclass

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from services.draft_import import DraftImportError, DraftUnavailable
from view.background_task import BackgroundTask


# Workers have no QObject parent: destroying a window must not destroy a running
# QThread. Keep them alive until finished even if their receiver disappears.
_active_reads = set()


@dataclass(frozen=True)
class _ReadResult:
    generation: int
    snapshot: object = None
    status: str | None = None
    unavailable: bool = False


def _release_task(task):
    _active_reads.discard(task)
    task.deleteLater()


class DraftFollower(QObject):
    """One bounded read at a time; generation checks discard disabled results."""

    snapshot_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    draft_unavailable = pyqtSignal()

    def __init__(self, read_snapshot, is_busy, parent=None, interval_ms=1500):
        super().__init__(parent)
        self._read_snapshot = read_snapshot
        self._is_busy = is_busy
        self._enabled = False
        self._generation = 0
        self._task = None
        self._last_status = None
        self._unavailable = False
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, int(interval_ms)))
        self._timer.timeout.connect(self.poll)

    @property
    def enabled(self):
        return self._enabled

    @property
    def running(self):
        return self._task is not None and self._task.isRunning()

    def _status(self, message):
        if self._last_status != message:
            self._last_status = message
            self.status_changed.emit(message)

    @pyqtSlot(bool)
    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._generation += 1
        self._unavailable = False
        if enabled:
            self._status("Automatikus klienskövetés bekapcsolva…")
            self._timer.start()
            self.poll()
        else:
            self._timer.stop()
            self._status("Az automatikus klienskövetés kikapcsolva.")

    @pyqtSlot()
    def poll(self):
        if not self._enabled or self._task is not None:
            return
        try:
            if self._is_busy():
                return
        except Exception:
            self._status("A klienskövetés most nem indítható. Próbáld újra később.")
            return
        generation = self._generation
        read = self._read_snapshot

        def operation():
            try:
                return _ReadResult(generation, snapshot=read())
            except DraftUnavailable:
                return _ReadResult(generation, status="Nincs aktív hősválasztás, várakozás…", unavailable=True)
            except DraftImportError as exc:
                return _ReadResult(generation, status=str(exc) or "A helyi LoL-kliens nem érhető el.")
            except Exception:
                # Never surface arbitrary transport exception text: it may
                # contain a URL, local token or other connection credentials.
                return _ReadResult(generation, status="A kliens adatainak beolvasása nem sikerült. Próbáld újra később.")

        task = BackgroundTask(operation)
        task._draft_generation = generation
        self._task = task
        _active_reads.add(task)
        task.succeeded.connect(self._on_result)
        task.failed.connect(self._on_failure)
        task.finished.connect(self._on_finished)
        task.finished.connect(lambda worker=task: _release_task(worker))
        task.start()

    @pyqtSlot(object)
    def _on_result(self, result):
        if not self._enabled or result.generation != self._generation:
            return
        if result.unavailable:
            self._status(result.status)
            if not self._unavailable:
                self._unavailable = True
                self.draft_unavailable.emit()
            return
        if result.status is not None:
            self._status(result.status)
            return
        self._unavailable = False
        self._last_status = None
        self.snapshot_ready.emit(result.snapshot)

    @pyqtSlot(str)
    def _on_failure(self, _message):
        task = self._task
        if self._enabled and task is not None and task._draft_generation == self._generation:
            self._status("A kliens adatainak beolvasása nem sikerült. Próbáld újra később.")

    @pyqtSlot()
    def _on_finished(self):
        if self.sender() is self._task:
            self._task = None
