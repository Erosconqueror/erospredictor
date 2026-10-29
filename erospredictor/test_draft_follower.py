"""Qt worker scheduling and stale-result protection for the local draft follower."""

import importlib.util
import os
from threading import Event, get_ident
import unittest


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 required")
class DraftFollowerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from view.draft_follower import DraftFollower
        self.calls = []
        self.snapshots, self.statuses, self.unavailable = [], [], []
        self.busy = False
        self.release = Event()
        self.started = Event()
        self.result = object()
        self.error = None

        def read():
            self.calls.append(get_ident())
            self.started.set()
            self.release.wait(2)
            if self.error is not None:
                raise self.error
            return self.result

        self.follower = DraftFollower(read, lambda: self.busy, interval_ms=10000)
        self.follower.snapshot_ready.connect(self.snapshots.append)
        self.follower.status_changed.connect(self.statuses.append)
        self.follower.draft_unavailable.connect(lambda: self.unavailable.append(True))
        self.addCleanup(self.cleanup)

    def wait_until(self, predicate, message="Condition was not reached"):
        from PyQt6.QtCore import QElapsedTimer
        from PyQt6.QtTest import QTest
        timer = QElapsedTimer()
        timer.start()
        while not predicate() and timer.elapsed() < 3000:
            QTest.qWait(10)
        self.assertTrue(predicate(), message)

    def finish(self):
        self.release.set()
        self.wait_until(lambda: self.follower._task is None)

    def cleanup(self):
        self.follower.set_enabled(False)
        self.finish()
        self.follower.deleteLater()
        self.app.processEvents()

    def test_disabled_follower_does_not_read(self):
        self.follower.poll()
        self.assertFalse(self.follower.enabled)
        self.assertFalse(self.follower.running)
        self.assertEqual(self.calls, [])

    def test_enabled_read_runs_on_worker_and_emits_snapshot(self):
        self.follower.set_enabled(True)
        self.wait_until(self.started.is_set)
        self.assertTrue(self.follower.running)
        self.finish()
        self.assertNotEqual(self.calls, [get_ident()])
        self.assertEqual(self.snapshots, [self.result])

    def test_busy_main_operation_skips_new_reads(self):
        self.busy = True
        self.follower.set_enabled(True)
        self.follower.poll()
        self.assertFalse(self.follower.running)
        self.assertEqual(self.calls, [])
        self.busy = False
        self.follower.poll()
        self.wait_until(self.started.is_set)
        self.finish()
        self.assertEqual(len(self.calls), 1)

    def test_polling_cannot_overlap_a_pending_read(self):
        self.follower.set_enabled(True)
        self.wait_until(self.started.is_set)
        for _ in range(5):
            self.follower.poll()
        self.finish()
        self.assertEqual(len(self.calls), 1)

    def test_disable_returns_before_worker_finishes_and_ignores_late_snapshot(self):
        self.follower.set_enabled(True)
        self.wait_until(self.started.is_set)
        self.follower.set_enabled(False)
        self.assertTrue(self.follower.running)
        self.finish()
        self.assertEqual(self.snapshots, [])

    def test_disable_and_reenable_discards_previous_generation_then_reads_fresh(self):
        old = self.result
        self.follower.set_enabled(True)
        self.wait_until(self.started.is_set)
        self.follower.set_enabled(False)
        self.follower.set_enabled(True)
        self.finish()
        self.assertEqual(self.snapshots, [])
        self.result = object()
        self.follower.poll()
        self.finish()
        self.assertEqual(self.snapshots, [self.result])
        self.assertIsNot(self.snapshots[0], old)

    def test_no_active_draft_is_a_waiting_state_and_one_transition_signal(self):
        from services.draft_import import DraftUnavailable
        self.error = DraftUnavailable("A private provider message is unnecessary")
        self.follower.set_enabled(True)
        self.finish()
        self.assertEqual(self.snapshots, [])
        self.assertIn("Nincs aktív hősválasztás", self.statuses[-1])
        self.assertEqual(self.unavailable, [True])
        self.follower.poll()
        self.finish()
        self.assertEqual(self.unavailable, [True])
        self.error = None
        self.follower.poll()
        self.finish()
        self.error = DraftUnavailable("gone")
        self.follower.poll()
        self.finish()
        self.assertEqual(len(self.unavailable), 2)

    def test_expected_error_is_displayed_but_arbitrary_exception_text_is_not(self):
        from services.draft_import import DraftImportError
        self.error = DraftImportError("Indítsd el a helyi klienst.")
        self.follower.set_enabled(True)
        self.finish()
        self.assertEqual(self.statuses[-1], "Indítsd el a helyi klienst.")
        self.error = RuntimeError("private-local-auth-token")
        self.follower.poll()
        self.finish()
        self.assertNotIn("private-local-auth-token", " ".join(self.statuses))
        self.assertEqual(self.snapshots, [])
        self.assertEqual(self.unavailable, [])

    def test_late_unavailable_result_does_not_clear_a_new_generation(self):
        from services.draft_import import DraftUnavailable
        self.error = DraftUnavailable("gone")
        self.follower.set_enabled(True)
        self.wait_until(self.started.is_set)
        self.follower.set_enabled(False)
        self.follower.set_enabled(True)
        self.finish()
        self.assertEqual(self.unavailable, [])
        self.assertNotIn("Nincs aktív", self.statuses[-1])

    def test_result_can_arrive_after_main_becomes_busy(self):
        self.follower.set_enabled(True)
        self.wait_until(self.started.is_set)
        self.busy = True
        self.finish()
        self.assertEqual(self.snapshots, [self.result])

    def test_timer_keeps_following_until_switched_off(self):
        from PyQt6.QtTest import QTest
        self.follower._timer.setInterval(30)
        self.release.set()
        self.follower.set_enabled(True)
        self.wait_until(lambda: len(self.calls) >= 2)
        self.follower.set_enabled(False)
        self.finish()
        count = len(self.calls)
        QTest.qWait(100)
        self.assertEqual(len(self.calls), count)


if __name__ == "__main__":
    unittest.main()
