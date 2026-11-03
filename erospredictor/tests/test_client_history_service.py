"""First-party history uses mocked transports; no real session credentials."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import requests

from services import client_history_service as hs
from services.draft_import import DraftImportError


SERVER = "https://euc1-red.pp.sgp.pvp.net"


def response(status=200, games=None, headers=None):
    return SimpleNamespace(status_code=status, headers=headers or {},
        json=lambda: {"games": [{"json": game} for game in (games or [])]})


class ClientHistoryServiceTests(unittest.TestCase):
    def setUp(self):
        hs.RiotClientHistory._cooldowns.clear()
        self.addCleanup(hs.RiotClientHistory._cooldowns.clear)
        self.transport = MagicMock()
        self.elapsed = 0
        self.client = hs.RiotClientHistory(SERVER, "private-session", "own/puuid",
            transport=self.transport, clock=lambda: self.elapsed, sleeper=self.sleep)

    def sleep(self, seconds):
        self.elapsed += seconds

    def test_only_configured_region_and_player_platform_server_are_selected(self):
        config = """region_data:
  EUW:
    servers:
      player_platform_edge:
        player_platform_edge_url: https://wrong-region.pp.sgp.pvp.net
  EUNE:
    servers:
      league_edge:
        league_edge_url: https://not-the-history-service.lol.sgp.pvp.net
      player_platform_edge:
        player_platform_edge_url: 'https://euc1-red.pp.sgp.pvp.net'
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system.yaml"
            path.write_text(config, encoding="utf-8")
            self.assertEqual(hs.server_from_config(path, "EUNE"), SERVER)
            with self.assertRaises(DraftImportError):
                hs.server_from_config(path, "NA")

    def test_server_validation_rejects_foreign_hosts_redirect_paths_and_credentials(self):
        for server in (None, "http://euc1-red.pp.sgp.pvp.net", "https://example.org",
                       SERVER + ".example.org", SERVER + "/path", SERVER + "?url=x",
                       "https://user:secret@euc1-red.pp.sgp.pvp.net",
                       "https://euc1-red.pp.sgp.pvp.net:8443", "https://[invalid"):
            with self.subTest(server=server), self.assertRaises(DraftImportError):
                hs.validate_server(server)
        self.assertEqual(hs.validate_server(SERVER + "/"), SERVER)

    def test_paging_is_bounded_and_uses_tls_without_redirect_or_developer_key(self):
        self.transport.get.side_effect = [response(games=[{"gameId": i} for i in range(50)]),
                                          response(games=[{"gameId": 51}])]
        pages = list(self.client.pages(1000))
        self.assertEqual([len(page) for page in pages], [50, 1])
        calls = self.transport.get.call_args_list
        self.assertEqual([call.kwargs["params"]["startIndex"] for call in calls], [0, 50])
        self.assertEqual(calls[0].args[0], SERVER + "/match-history-query/v1/products/lol/player/own%2Fpuuid/SUMMARY")
        for call in calls:
            self.assertEqual(call.kwargs["headers"], {"Authorization": "Bearer private-session"})
            self.assertTrue(call.kwargs["verify"])
            self.assertFalse(call.kwargs["allow_redirects"])
            self.assertNotIn("auth", call.kwargs)
            self.assertEqual(call.kwargs["params"]["count"], 50)
        self.assertGreaterEqual(self.elapsed, 0.5)

    def test_owned_session_ignores_environment_proxies_and_closes(self):
        self.client._transport = None
        with patch.object(hs.requests, "Session", return_value=self.transport):
            self.transport.get.return_value = response()
            self.assertEqual(list(self.client.pages(100)), [])
        self.assertFalse(self.transport.trust_env)
        self.transport.close.assert_called_once()

    def test_429_honors_retry_after_instead_of_sending_another_request_immediately(self):
        self.transport.get.side_effect = [response(429, headers={"Retry-After": "3"}),
                                         response(games=[{"gameId": 1}])]
        self.assertEqual(list(self.client.pages(100)), [[{"gameId": 1}]])
        self.assertEqual(self.transport.get.call_count, 2)
        self.assertGreaterEqual(self.elapsed, 3)

    def test_long_retry_after_survives_refresh_without_extra_requests(self):
        self.transport.get.return_value = response(429, headers={"Retry-After": "180"})
        with self.assertRaisesRegex(DraftImportError, "180"):
            list(self.client.pages(100))
        another = hs.RiotClientHistory(SERVER, "new-session", "own/puuid",
            transport=self.transport, clock=lambda: self.elapsed, sleeper=self.sleep)
        with self.assertRaisesRegex(DraftImportError, "várakozást"):
            list(another.pages(100))
        self.assertEqual(self.transport.get.call_count, 1)

    def test_network_and_server_failures_retry_but_never_leak_response_or_credentials(self):
        self.transport.get.side_effect = [requests.ConnectionError("private-session"), response(503), response()]
        self.assertEqual(list(self.client.pages(100)), [])
        self.assertEqual(self.transport.get.call_count, 3)
        self.assertGreaterEqual(self.elapsed, 3)
        self.transport.get.side_effect = requests.ConnectionError("private-session")
        with self.assertRaises(DraftImportError) as error:
            list(self.client.pages(100))
        self.assertNotIn("private-session", str(error.exception))

    def test_auth_redirect_and_malformed_results_do_not_retry_or_follow(self):
        for result in (response(401), response(403), response(302),
                       SimpleNamespace(status_code=200, json=lambda: {"games": "invalid"}),
                       SimpleNamespace(status_code=200, json=lambda: {"games": [{"secret": "private-session"}]})):
            self.transport.reset_mock()
            self.transport.get.return_value = result
            with self.subTest(result=result.status_code), self.assertRaises(DraftImportError) as error:
                list(self.client.pages(100))
            self.assertNotIn("private-session", str(error.exception))
            self.transport.get.assert_called_once()

    def test_cancellation_interrupts_backoff_without_another_request(self):
        self.transport.get.return_value = response(429, headers={"Retry-After": "30"})
        with self.assertRaisesRegex(DraftImportError, "leállítva"):
            list(self.client.pages(100, cancelled=lambda: self.elapsed >= 0.5))
        self.transport.get.assert_called_once()
        self.assertLess(self.elapsed, 1)


if __name__ == "__main__":
    unittest.main()
