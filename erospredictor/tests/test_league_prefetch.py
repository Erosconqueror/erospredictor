"""League prefetch overlaps work without sharing HTTP sessions or advancing SQL cursors."""
import unittest
from types import SimpleNamespace
from threading import Event
from unittest.mock import patch, MagicMock

from services.league_prefetch import LeaguePrefetcher, PrefetchGate
from services.riot_rate_limit import RiotRateLimiter


class PrefetchTests(unittest.TestCase):
    def test_separate_http_client_prefetch_is_consumed_once(self):
        client = MagicMock()
        client.get_league_exp_players.return_value = [{"puuid": "next-player"}]
        original = SimpleNamespace(api_key="test", rate_limiter=RiotRateLimiter(logger=None))
        with patch("model.riot.Riot", return_value=client):
            helper = LeaguePrefetcher(original, "euw1", "europe")
        try:
            helper.schedule("GOLD", "I", 2)
            fallback = MagicMock(return_value=[])
            self.assertEqual(helper.get("GOLD", "I", 2, fallback), [{"puuid": "next-player"}])
            fallback.assert_not_called()
            client.get_league_exp_players.assert_called_once_with("GOLD", "I", page=2)
            self.assertEqual(helper.get("GOLD", "I", 2, fallback), [])
        finally:
            helper.close()

    def test_failed_speculation_falls_back_to_normal_retry_path(self):
        client = MagicMock()
        client.get_league_exp_players.side_effect = ConnectionError("offline")
        with patch("model.riot.Riot", return_value=client):
            helper = LeaguePrefetcher(SimpleNamespace(api_key="test", rate_limiter=RiotRateLimiter(logger=None)), "kr", "asia")
        try:
            helper.schedule("GOLD", "II", 2)
            fallback = MagicMock(return_value=[{"puuid": "recovered"}])
            self.assertEqual(helper.get("GOLD", "II", 2, fallback), [{"puuid": "recovered"}])
            fallback.assert_called_once()
        finally:
            helper.close()

    def test_gate_uses_the_same_budget_and_cancel_is_interruptible(self):
        stop = Event()
        shared = RiotRateLimiter(logger=None)
        gate = PrefetchGate(shared, stop)
        gate.wait_until("euw1", "league", "entries")
        self.assertEqual(len(shared._apps["euw1"]["attempts"]), 1)
        self.assertEqual(gate.call(lambda: "ok", "euw1", 0, retry_transient=False), "ok")
        stop.set()
        with self.assertRaises(InterruptedError):
            gate.wait_until("euw1", "league", "entries")
