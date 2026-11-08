"""Final client rosters override estimates, without relying on roster order."""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import requests

from services.draft_import import (DraftPick, DraftSnapshot, DraftUnavailable,
    LcuDraftClient, parse_gameflow_session)
from services.draft_assignment import assign_draft


MAPPING = {"266": 0, "101": 1, "201": 2}


def flow():
    return {"phase": "GameStart", "gameData": {"gameId": 100, "queue": {"id": 440},
        "teamOne": [{"championId": 266, "summonerId": 42, "selectedPosition": "TOP",
                     "gameName": "Me", "tagLine": "EU"}],
        "teamTwo": [{"championId": 201, "gameName": "Braum", "tagLine": "EU"},
                    {"championId": 101, "gameName": "Xerath", "tagLine": "EU"}]}}


class LoadingRosterTests(unittest.TestCase):
    def test_final_enemy_positions_override_manual_and_statistical_roles(self):
        players = [{"riotId": "Xerath#EU", "team": "CHAOS", "position": "TOP"},
                   {"riotId": "Braum#EU", "team": "CHAOS", "position": "UTILITY"}]
        snapshot = parse_gameflow_session(flow(), MAPPING, {"summonerId": 42}, players)
        result = assign_draft(snapshot, {}, previous_roles={("red", 1): 2})
        self.assertEqual(result.values["red"], [1, None, None, None, 2])
        self.assertTrue(all(item.source == "client" for item in result.assignments))
        self.assertEqual(snapshot.own_team, "blue")
        self.assertEqual(snapshot.own_role, 0)
        self.assertEqual(snapshot.phase, "GameStart")

    def test_missing_positions_never_use_display_order(self):
        snapshot = parse_gameflow_session(flow(), MAPPING, {"summonerId": 42})
        result = assign_draft(snapshot, {})
        self.assertEqual(result.values["red"], [None] * 5)
        self.assertEqual(len(result.unresolved), 2)
        self.assertIn("becslések", snapshot.warnings[0])

    def test_matching_game_retains_bans_and_known_own_side(self):
        previous = DraftSnapshot((), blue_bans=(5,), red_bans=(6,), own_team="blue",
                                 session_key="game:100", own_role=0)
        result = parse_gameflow_session(flow(), MAPPING, previous=previous)
        self.assertEqual((result.blue_bans, result.red_bans), ((5,), (6,)))
        self.assertEqual(result.own_team, "blue")
        result = parse_gameflow_session(flow(), MAPPING, previous=replace(previous, session_key="game:101"))
        self.assertEqual(result.blue_bans, ())
        self.assertIsNone(result.own_team)

    def test_only_same_side_and_unique_identity_can_confirm_a_role(self):
        player = {"riotId": "Xerath#EU", "team": "ORDER", "position": "TOP"}
        result = parse_gameflow_session(flow(), MAPPING, {"summonerId": 42}, [player])
        self.assertIsNone(next(p.role for p in result.picks if p.internal_id == 1))
        player["team"] = "CHAOS"
        result = parse_gameflow_session(flow(), MAPPING, {"summonerId": 42}, [player, player])
        self.assertIsNone(next(p.role for p in result.picks if p.internal_id == 1))

    def test_raw_champion_name_can_join_when_gameflow_omits_identity(self):
        result = parse_gameflow_session(flow(), MAPPING, {"summonerId": 42},
            [{"team": "CHAOS", "rawChampionName": "game_character_displayname_Xerath", "position": "MIDDLE"}],
            names={"1": "Xerath"})
        self.assertEqual(next(p.role for p in result.picks if p.internal_id == 1), 2)

    def test_finished_game_does_not_reimport_a_stale_roster(self):
        for phase in ("Lobby", "EndOfGame", "WaitingForStats", "None"):
            with self.assertRaises(DraftUnavailable):
                parse_gameflow_session(dict(flow(), phase=phase), MAPPING)

    def test_transient_live_api_gap_keeps_last_confirmed_position_for_this_game(self):
        value = flow()
        value["gameData"]["teamTwo"][1]["selectedPosition"] = "UTILITY"
        first = parse_gameflow_session(value, MAPPING, {"summonerId": 42},
            [{"riotId": "Xerath#EU", "team": "CHAOS", "position": "MIDDLE"}])
        second = parse_gameflow_session(value, MAPPING, {"summonerId": 42}, previous=first)
        self.assertEqual(next(p.role for p in second.picks if p.internal_id == 1), 2)
        value["gameData"]["gameId"] = 101
        third = parse_gameflow_session(value, MAPPING, {"summonerId": 42}, previous=first)
        self.assertEqual(next(p.role for p in third.picks if p.internal_id == 1), 4)

    def test_local_fallback_keeps_lcu_auth_away_from_live_api(self):
        with tempfile.TemporaryDirectory() as folder:
            lockfile = Path(folder) / "lockfile"
            lockfile.write_text("LeagueClient:123:34567:private-test-token:https")
            for champ_select_status in (404, 200):
                with self.subTest(champ_select_status=champ_select_status):
                    transport = MagicMock()
                    responses = []
                    for status, payload in ((champ_select_status, {"myTeam": [], "theirTeam": []}),
                                            (200, flow()), (200, {"summonerId": 42}),
                                            (200, [{"riotId": "Xerath#EU", "team": "CHAOS", "position": "MIDDLE"}])):
                        response = MagicMock(status_code=status)
                        response.json.return_value = payload
                        responses.append(response)
                    transport.get.side_effect = responses
                    with patch("services.draft_import.saved_lockfile", return_value=lockfile):
                        result = LcuDraftClient(lockfile, MAPPING, transport).read_draft()
                    self.assertEqual(result.phase, "GameStart")
                    call = transport.get.call_args_list[-1]
                    self.assertEqual(call.args[0], "https://127.0.0.1:2999/liveclientdata/playerlist")
                    self.assertNotIn("auth", call.kwargs)
                    self.assertFalse(call.kwargs["allow_redirects"])

    def test_live_not_ready_still_returns_gameflow_roster(self):
        transport = MagicMock()
        transport.get.side_effect = requests.ConnectionError("private-test-token")
        self.assertEqual(LcuDraftClient(transport=transport)._live_players(transport), [])


if __name__ == "__main__":
    unittest.main()
