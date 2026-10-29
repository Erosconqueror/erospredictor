"""Own-account history, conservative visual markers and local-only transport."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit, parse_qs

from services import personal_history as ph
from services import feature_flags
from services.draft_import import DraftImportError
from services.explanation_format import personal_history_html


ACCOUNT = {"puuid": "own-account", "summonerId": 42, "gameName": "Tester", "tagLine": "EUW"}
MAPPING = {"266": 0, "103": 1, "86": 2}


def game(game_id=1, win=True, own_team=100, summary=False, queue=420):
    players, identities = [], []
    for team in (100, 200):
        for role, position in enumerate(("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")):
            pid = len(players) + 1
            own = team == own_team and role == 0
            champion = 266 if own else 103 if team == own_team and role == 1 else 86 if team != own_team and role == 0 else 1000 + pid
            players.append({"participantId": pid, "championId": champion, "teamId": team,
                            "teamPosition": position, "stats": {"win": win if team == own_team else not win}})
            identities.append({"participantId": pid, "player": dict(ACCOUNT) if own else
                               {"puuid": "other-person-identity", "summonerName": "Not stored"}})
    if summary:
        own_ids = {item["participantId"] for item in identities if item["player"].get("puuid") == ACCOUNT["puuid"]}
        players = [item for item in players if item["participantId"] in own_ids]
        identities = [item for item in identities if item["participantId"] in own_ids]
    return {"gameId": game_id, "platformId": "EUW1", "gameCreation": 1700000000000 + game_id,
            "queueId": queue, "mapId": 11, "gameDuration": 1800,
            "participants": players, "participantIdentities": identities}


def profile(count=20, wins=12):
    rows = tuple(ph.parse_game(game(i, i <= wins), ACCOUNT) for i in range(1, count + 1))
    return ph.PersonalHistory(ACCOUNT["puuid"], "Tester#EUW", rows)


class PersonalEvidenceTests(unittest.TestCase):
    def setUp(self):
        enabled = patch.object(feature_flags, "PERSONAL_HISTORY_ENABLED", True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_own_identity_and_red_side_are_not_inferred_from_array_order(self):
        raw = game(own_team=200, win=False)
        raw["participants"].reverse()
        row = ph.parse_game(raw, ACCOUNT)
        self.assertEqual(row["champion"], 266)
        self.assertEqual(row["enemy"], 86)
        self.assertFalse(row["win"])
        self.assertEqual(row["role"], 0)
        self.assertTrue(any(item["champion"] == 103 for item in row["allies"]))

    def test_ambiguous_bot_roles_and_missing_details_do_not_invent_matchups(self):
        self.assertIsNone(ph.participant_role({"timeline": {"lane": "BOTTOM", "role": "DUO"}}))
        self.assertEqual(ph.participant_role({"timeline": {"lane": "BOTTOM", "role": "DUO_SUPPORT"}}), 4)
        row = ph.parse_game(game(summary=True), ACCOUNT)
        self.assertIsNone(row["enemy"])
        self.assertEqual(row["allies"], [])
        self.assertFalse(row["complete"])

    def test_other_accounts_remakes_and_non_ranked_matches_are_excluded(self):
        self.assertIsNone(ph.parse_game(game(), dict(ACCOUNT, puuid="another-account")))
        for change in ({"queueId": 450}, {"mapId": 12}, {"gameDuration": 200}):
            self.assertIsNone(ph.parse_game(dict(game(), **change), ACCOUNT))
        raw = game()
        raw["participants"][0]["stats"]["gameEndedInEarlySurrender"] = True
        raw["gameDuration"] = 900
        self.assertIsNotNone(ph.parse_game(raw, ACCOUNT))
        raw["gameDuration"] = 180
        self.assertIsNone(ph.parse_game(raw, ACCOUNT))

    def test_one_lucky_win_is_not_a_good_marker_and_regular_play_has_its_own_marker(self):
        self.assertIsNone(profile(1, 2).evidence(0, 0, 2, [-1, 1, -1, -1, -1], MAPPING)["marker"])
        self.assertEqual(profile(3, 4).evidence(0, 0, 2, [-1, 1, -1, -1, -1], MAPPING)["marker"], "played")
        self.assertEqual(profile(20, 14).evidence(0, 0, 2, [-1, 1, -1, -1, -1], MAPPING)["marker"], "good")
        self.assertEqual(profile(20, 4).evidence(0, 0, 2, [-1, 1, -1, -1, -1], MAPPING)["marker"], "played")

    def test_role_specific_matchup_and_synergy_include_sample_threshold(self):
        small = profile(3, 2).evidence(0, 0, 2, [-1, 1, -1, -1, -1], MAPPING)
        self.assertFalse(small["matchup"]["enough"])
        self.assertFalse(small["synergies"][0]["enough"])
        large = profile(20, 14).evidence(0, 0, 2, [-1, 1, -1, -1, -1], MAPPING)
        self.assertTrue(large["matchup"]["enough"])
        self.assertEqual(large["matchup"]["games"], 20)
        self.assertTrue(large["synergies"][0]["enough"])
        other_role = profile().evidence(0, 2, 2, [-1, 1, -1, -1, -1], MAPPING)
        self.assertEqual(other_role["scope"], "all")
        self.assertEqual(other_role["role_games"], 0)
        self.assertIsNone(other_role["matchup"])
        self.assertEqual(other_role["synergies"], [])

    def test_annotations_keep_model_probability_order_and_original_records(self):
        recs = [{"id": 1, "wr": 60.0}, {"id": 0, "wr": 54.0}]
        before = deepcopy(recs)
        annotated = ph.annotate_recommendations(recs, profile(), MAPPING, [-1, 1, -1, -1, -1], [2, -1, -1, -1, -1], "blue", 0)
        self.assertEqual([(item["id"], item["wr"]) for item in annotated], [(1, 60.0), (0, 54.0)])
        self.assertEqual(recs, before)
        self.assertIsNone(annotated[0]["personal"])
        self.assertIsNotNone(annotated[1]["personal"])

    def test_format_escapes_champion_names_and_explains_small_samples(self):
        evidence = profile(3, 2).evidence(0, 0, 2, [-1, 1, -1, -1, -1], MAPPING)
        html = personal_history_html(evidence, {1: "<ally>", 2: "<enemy>"}, detailed=True)
        self.assertIn("&lt;ally&gt;", html)
        self.assertIn("kis minta", html)
        self.assertIn("nem módosítják", html)


class HistoryTransportTests(unittest.TestCase):
    def setUp(self):
        enabled = patch.object(feature_flags, "PERSONAL_HISTORY_ENABLED", True)
        enabled.start()
        self.addCleanup(enabled.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.lockfile = self.root / "lockfile"
        self.lockfile.write_text("LeagueClient:123:34567:private-auth-token:https")
        setting = patch.object(ph, "settings_path", return_value=self.root / "preferences/settings.json")
        setting.start()
        self.addCleanup(setting.stop)
        self.pages = {0: [game(summary=True)]}
        self.details, self.calls = {}, []
        self.current = lambda: dict(ACCOUNT)
        self.transport = MagicMock()
        self.transport.get.side_effect = self.respond
        self.client = ph.LcuHistoryClient(self.lockfile, transport=self.transport)

    def respond(self, url, **kwargs):
        self.calls.append((url, kwargs))
        path, query = urlsplit(url).path, parse_qs(urlsplit(url).query)
        if path == ph.CURRENT:
            payload = self.current()
        elif path.endswith("/matches"):
            payload = {"platformId": "EUW1", "games": {"games": self.pages.get(int(query["begIndex"][0]), [])}}
        else:
            game_id = int(path.rsplit("/", 1)[1])
            payload = self.details.get(game_id, game(game_id))
        return SimpleNamespace(status_code=200, json=lambda: deepcopy(payload))

    def test_local_authentication_cache_dedup_and_reuse_without_other_player_identity_storage(self):
        first = self.client.read_history()
        self.assertEqual(first.name, "Tester#EUW")
        self.assertEqual(len(first.games), 1)
        self.client.read_history()
        detail_calls = [url for url, _ in self.calls if "/v1/games/" in url]
        self.assertEqual(len(detail_calls), 1)
        for url, options in self.calls:
            self.assertTrue(url.startswith("https://127.0.0.1:34567/"))
            self.assertFalse(options["allow_redirects"])
            self.assertEqual(options["timeout"], (1, 2))
        cache = self.client._cache_path(ACCOUNT["puuid"]).read_text()
        self.assertNotIn("private-auth-token", cache)
        self.assertNotIn("other-person-identity", cache)
        self.assertNotIn("Not stored", cache)

    def test_account_switch_before_cache_commit_stops_import(self):
        calls = 0
        def current():
            nonlocal calls
            calls += 1
            return dict(ACCOUNT, puuid=ACCOUNT["puuid"] if calls == 1 else "other-account")
        self.current = current
        with self.assertRaisesRegex(DraftImportError, "Fiókváltás"):
            self.client.read_history()
        self.assertFalse(self.client._cache_path(ACCOUNT["puuid"]).exists())

    def test_verify_account_rejects_a_different_login(self):
        self.current = lambda: dict(ACCOUNT, puuid="other-account")
        with self.assertRaisesRegex(DraftImportError, "Másik fiók"):
            self.client.verify_account(profile())

    def test_repeated_pages_and_non_ranked_pages_are_handled_without_losing_later_ranked_games(self):
        self.pages = {0: [game(i, queue=450) for i in range(1, 21)],
                      20: [game(i, queue=450) for i in range(21, 41)], 40: [game(41)]}
        self.assertEqual(len(self.client.read_history().games), 1)
        self.pages = {0: [game(i) for i in range(1, 21)], 20: [game(i) for i in range(1, 21)]}
        self.calls.clear()
        self.client.read_history(1000)
        self.assertEqual(len([url for url, _ in self.calls if "/matches?" in url]), 2)

    def test_actual_twenty_nonranked_games_are_not_reported_as_a_thousand_without_ranked_history(self):
        page = [game(i, queue=2400) for i in range(1, 21)]
        self.pages = {0: page, 20: page}
        with patch.object(self.client, "_client_service", side_effect=AssertionError("External request")) as remote:
            result = self.client.read_history(1000)
        remote.assert_not_called()
        self.assertEqual(result.games, ())
        self.assertEqual((result.examined_games, result.requested_games, result.ranked_games_seen), (20, 1000, 0))
        self.assertTrue(result.limited)
        self.assertEqual(result.source, "lcu")
        self.assertIn("ismételte", " ".join(result.warnings))
        self.assertIn("nem azt jelenti", " ".join(result.warnings))
        self.assertEqual(len([url for url, _ in self.calls if "/matches?" in url]), 2)
        self.assertFalse(any("token" in url or "/games/" in url for url, _ in self.calls))

    def test_eligible_summary_missing_own_identity_is_hydrated_before_it_is_rejected(self):
        summary = game(summary=True)
        summary["participantIdentities"] = []
        self.pages = {0: [summary]}
        result = self.client.read_history()
        self.assertEqual(len(result.games), 1)
        self.assertTrue(result.games[0]["complete"])
        self.assertEqual(result.games[0]["enemy"], 86)

    def test_explicit_service_uses_flat_participants_and_keeps_only_normalized_cache(self):
        raw = game()
        for participant, identity in zip(raw["participants"], raw.pop("participantIdentities")):
            participant["puuid"] = identity["player"]["puuid"]
            participant["win"] = participant.pop("stats")["win"]
        service = MagicMock()
        service.pages.return_value = iter([[raw]])
        with patch.object(self.client, "_client_service", return_value=service) as create:
            result = self.client.read_history(1000, allow_client_service=True)
        create.assert_called_once()
        self.assertEqual(result.source, "riot-client-service")
        self.assertEqual((result.examined_games, result.ranked_games_seen), (1, 1))
        self.assertTrue(result.games[0]["complete"])
        self.assertEqual(result.games[0]["champion"], 266)
        self.assertFalse(any("/v1/games/" in url for url, _ in self.calls))
        self.assertNotIn("other-person-identity", self.client._cache_path(ACCOUNT["puuid"]).read_text())

    def test_service_token_is_read_only_after_explicit_opt_in_and_bound_to_current_account(self):
        self.client._system_config_path = self.root / "system.yaml"
        with patch.object(ph, "server_from_config", return_value="https://euc1-red.pp.sgp.pvp.net") as server, \
                patch.object(self.client, "_get", side_effect=[{"region": "EUNE"},
                    {"accessToken": "private-session", "subject": "another-account"}]) as get, \
                patch.object(ph, "RiotClientHistory", side_effect=AssertionError("Foreign account")) as service:
            with self.assertRaisesRegex(DraftImportError, "Fiókváltás"):
                self.client._client_service(self.transport, None, ACCOUNT)
        service.assert_not_called()
        self.assertEqual(get.call_args_list[-1].args[-1], ph.ENTITLEMENTS)
        self.assertEqual(server.call_args.args[-1], "EUNE")

    def test_opted_in_service_uses_the_clients_region_and_rechecks_login_before_forwarding(self):
        self.client._system_config_path = self.root / "system.yaml"
        with patch.object(ph, "server_from_config", return_value="https://euc1-red.pp.sgp.pvp.net"), \
                patch.object(self.client, "_get", side_effect=[{"region": "EUNE"},
                    {"accessToken": "private-session", "subject": ACCOUNT["puuid"]}]), \
                patch.object(self.client, "_account", return_value=ACCOUNT) as current, \
                patch.object(ph, "RiotClientHistory") as service:
            self.client._client_service(self.transport, None, ACCOUNT)
        current.assert_called_once()
        service.assert_called_once_with("https://euc1-red.pp.sgp.pvp.net", "private-session", ACCOUNT["puuid"])
        with patch.object(ph, "server_from_config", return_value="https://euc1-red.pp.sgp.pvp.net"), \
                patch.object(self.client, "_get", side_effect=[{"region": "EUNE"}, {"accessToken": "private-session"}]), \
                patch.object(self.client, "_account", return_value=dict(ACCOUNT, puuid="switched")), \
                patch.object(ph, "RiotClientHistory") as service:
            with self.assertRaisesRegex(DraftImportError, "Fiókváltás"):
                self.client._client_service(self.transport, None, ACCOUNT)
        service.assert_not_called()

    def test_cancellation_keeps_completed_pages_for_next_refresh(self):
        self.pages = {0: [game(i) for i in range(1, 21)], 20: [game(21)]}
        count = 0
        def current():
            nonlocal count
            count += 1
            return dict(ACCOUNT)
        self.current = current
        with self.assertRaisesRegex(DraftImportError, "leállítva"):
            self.client.read_history(cancelled=lambda: count >= 2)
        self.assertEqual(len(self.client._cached(ACCOUNT)), 20)

    def test_corrupt_or_foreign_cache_is_ignored_and_endpoints_are_restricted(self):
        path = self.client._cache_path(ACCOUNT["puuid"])
        path.parent.mkdir(parents=True)
        path.write_text("broken json")
        self.assertEqual(self.client._cached(ACCOUNT), {})
        path.write_text(json.dumps({"version": 1, "puuid": "another-account", "games": []}))
        self.assertEqual(self.client._cached(ACCOUNT), {})
        for endpoint in ("https://example.org", "/lol-match-history/v1/games/../../escape",
                         "/lol-match-history/v1/products/lol/someone-else/matches?begIndex=0&endIndex=19"):
            self.assertFalse(self.client._allowed_endpoint(endpoint))

    def test_no_history_or_missing_details_returns_safe_empty_or_partial_results(self):
        self.pages = {0: []}
        self.assertEqual(self.client.read_history().games, ())
        self.pages = {0: [game(summary=True)]}
        self.details[1] = {}
        result = self.client.read_history()
        self.assertIsNone(result.games[0]["enemy"])
        self.assertTrue(result.warnings)


class ControllerHistoryTests(unittest.TestCase):
    def setUp(self):
        enabled = patch.object(feature_flags, "PERSONAL_HISTORY_ENABLED", True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_history_annotations_do_not_run_more_models_or_modify_probabilities(self):
        from controller.controller import Controller
        ctrl = Controller.__new__(Controller)
        ctrl.db = SimpleNamespace(get_champion_mapping=lambda: MAPPING)
        ctrl.core_model = MagicMock()
        ctrl.core_model.recommend_champs.return_value = [{"id": 0, "wr": 57.123}]
        ctrl.core_model.explain_recommendations.return_value = [{"id": 0, "wr": 57.123, "is_selected": True}]
        arguments = ("GOLD", [-1, 1, -1, -1, -1], [2, -1, -1, -1, -1])
        recs = ctrl.recommend_champs(*arguments, [], "blue", 0, history=profile())
        self.assertEqual(recs[0]["wr"], 57.123)
        self.assertIsNotNone(recs[0]["personal"])
        ctrl.core_model.recommend_champs.assert_called_once()
        details = ctrl.explain_recommendations(*arguments, "blue", 0, recs, selected_id=0, history=profile())
        self.assertTrue(details[0]["is_selected"])
        self.assertEqual(details[0]["wr"], 57.123)
        self.assertIsNotNone(details[0]["personal"])


class DisabledHistoryTests(unittest.TestCase):
    def setUp(self):
        disabled = patch.object(feature_flags, "PERSONAL_HISTORY_ENABLED", False)
        disabled.start()
        self.addCleanup(disabled.stop)

    def test_shipped_feature_flag_is_enabled(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_test_shipped_feature_flags", Path(feature_flags.__file__))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(module.PERSONAL_HISTORY_ENABLED)

    def test_history_read_and_account_verification_stop_before_discovery_or_network(self):
        transport = MagicMock()
        client = ph.LcuHistoryClient(transport=transport)
        with patch.object(ph, "discover_lockfiles", side_effect=AssertionError("Client discovery")) as discover, \
                patch.object(ph.requests, "Session", side_effect=AssertionError("Network session")) as session, \
                patch.object(client, "_cache_path", side_effect=AssertionError("Cache accessed")) as cache:
            with self.assertRaisesRegex(DraftImportError, "szüneteltetve"):
                client.read_history()
            with self.assertRaisesRegex(DraftImportError, "szüneteltetve"):
                client.verify_account(profile())
        discover.assert_not_called()
        session.assert_not_called()
        cache.assert_not_called()
        transport.get.assert_not_called()

    def test_controller_ignores_supplied_history_while_feature_is_disabled(self):
        from controller.controller import Controller
        ctrl = Controller.__new__(Controller)
        ctrl.db = SimpleNamespace(get_champion_mapping=MagicMock(side_effect=AssertionError("History mapping")))
        ctrl.core_model = MagicMock()
        ctrl.core_model.recommend_champs.return_value = [{"id": 0, "wr": 57.123}]
        ctrl.core_model.explain_recommendations.return_value = [{"id": 0, "wr": 57.123, "is_selected": True}]
        arguments = ("GOLD", [-1, 1, -1, -1, -1], [2, -1, -1, -1, -1])
        with patch.object(ph, "annotate_recommendations", side_effect=AssertionError("History annotations")) as annotate:
            recs = ctrl.recommend_champs(*arguments, [], "blue", 0, history=profile())
            details = ctrl.explain_recommendations(*arguments, "blue", 0, recs, selected_id=0, history=profile())
        annotate.assert_not_called()
        self.assertNotIn("personal", recs[0])
        self.assertNotIn("personal", details[0])
        self.assertEqual(recs[0]["wr"], 57.123)
        self.assertTrue(details[0]["is_selected"])


if __name__ == "__main__":
    unittest.main()
