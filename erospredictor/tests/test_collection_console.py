import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from services.collection_console import CollectionLogTail, ApiWatch, api_event, _sent


class CollectionConsoleTests(unittest.TestCase):
    def test_tail_skips_old_logs_and_waits_for_complete_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"worker.log"
            path.write_text("[Collection] old run\n", encoding="utf-8")
            tail = CollectionLogTail(path)
            output = io.StringIO()
            with redirect_stdout(output):
                with path.open("a", encoding="utf-8") as stream:
                    stream.write("private unrelated log\n[Collection] EUW1 | GOLD")
                tail.drain("[development] ")
                self.assertEqual(output.getvalue(), "")
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(": collecting\n")
                tail.drain("[development] ")
                tail.drain()
            self.assertEqual(output.getvalue(), "[development] [Collection] EUW1 | GOLD: collecting\n")

    def test_stall_is_throttled_and_response_announces_recovery(self):
        now = [1000]
        watch = ApiWatch("development/EUW1", clock=lambda: now[0])
        output = io.StringIO()
        with redirect_stdout(output):
            now[0] += 179
            watch.tick()
            self.assertEqual(output.getvalue(), "")
            now[0] += 1
            watch.tick()
            watch.tick()
            self.assertEqual(output.getvalue().count("nem erkezett API-valasz"), 1)
            watch.observe(dict(at=now[0], kind="response", route="europe", status=200))
            self.assertIn("ujra valaszol", output.getvalue())
            self.assertFalse(watch.stalled)

    def test_rate_limit_wait_is_distinguished_from_network_retry(self):
        for reason in ("rate_limit", "retry"):
            now = [1000]
            watch = ApiWatch("default/KR", clock=lambda: now[0])
            watch.observe(dict(at=1000, kind="wait", seconds=600, reason=reason))
            now[0] = 1200
            output = io.StringIO()
            with redirect_stdout(output):
                watch.tick()
            self.assertEqual("ismert API-keretvarakozas" in output.getvalue(), reason == "rate_limit")
            self.assertEqual(watch.stalled, reason == "retry")

    def test_two_level_relay_surfaces_api_state_and_never_forwards_raw_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            worker, profile = Path(folder)/"worker.log", Path(folder)/"profile.log"
            watch = ApiWatch("development/BR1", clock=lambda: 1000)
            child_tail = CollectionLogTail(worker, watch)
            parent_tail = CollectionLogTail(profile)
            event = dict(kind="response", at=1001, route="americas", status=403)
            worker.write_text("[CollectorAPI] " + json.dumps(event) + "\n[Collection] BR1 | IRON\n", encoding="utf-8")
            with profile.open("w", encoding="utf-8") as output, redirect_stdout(output):
                child_tail.drain("[development] ")
            result = io.StringIO()
            with redirect_stdout(result):
                parent_tail.drain()
            self.assertIn("HTTP 403", result.getvalue())
            self.assertIn("[development] [Collection] BR1 | IRON", result.getvalue())
            self.assertNotIn("CollectorAPI", result.getvalue())

    def test_telemetry_is_opt_in_and_healthy_responses_are_throttled(self):
        output = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), redirect_stdout(output):
            api_event("test-route", "response", status=200)
        self.assertEqual(output.getvalue(), "")
        _sent.clear()
        with patch.dict("os.environ", {"ERO_COLLECT_WINDOW": "test"}), redirect_stdout(output):
            api_event("test-route", "response", status=200)
            api_event("test-route", "response", status=200)
        self.assertEqual(output.getvalue().count("CollectorAPI"), 1)

    def test_longer_rate_wait_is_not_lost_to_heartbeat_throttling(self):
        _sent.clear()
        output = io.StringIO()
        with patch.dict("os.environ", {"ERO_COLLECT_WINDOW": "test"}), redirect_stdout(output), \
                patch("services.collection_console.time.time", side_effect=[1000, 1001]):
            api_event("test-route", "wait", reason="rate_limit", seconds=20)
            api_event("test-route", "wait", reason="rate_limit", seconds=600)
        self.assertEqual(output.getvalue().count("CollectorAPI"), 2)

    def test_expired_network_backoff_does_not_mislabel_a_new_budget_wait(self):
        from services.riot_rate_limit import RiotRateLimiter
        now = [1000]
        limiter = RiotRateLimiter(clock=lambda: now[0], sleeper=lambda n: now.__setitem__(0, now[0]+n),
                                  logger=None, app_limits=((1, 240),))
        limiter._blocked_reasons["europe"] = "retry"
        limiter._blocked_until["europe"] = 999
        with patch("services.riot_rate_limit.api_event") as event:
            limiter.acquire("europe", "match", "get")
            limiter.acquire("europe", "match", "get")
        self.assertEqual(event.call_args.kwargs["reason"], "rate_limit")
