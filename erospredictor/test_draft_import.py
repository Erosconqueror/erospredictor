"""Offline fixtures for the optional League-client draft importer."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from services.draft_import import (
    DraftImportError, DraftPick, DraftSnapshot, LcuCredentials, LcuDraftClient,
    discover_lockfiles, parse_draft_session, read_lockfile, resolve_own_side,
)


MAPPING = {"266": 0, "103": 1, "84": 2, "12": 4, "32": 6}


def draft(own_side=100):
    own = {"cellId": 0, "championId": 266, "championPickIntent": 103,
           "assignedPosition": "top", "summonerId": 42}
    if own_side is not None:
        own["teamId"] = own_side
    return {
        "myTeam": [own],
        "theirTeam": [{"cellId": 5, "championId": 103, "assignedPosition": ""}],
        "actions": [[{"actorCellId": 0, "championId": 266, "type": "pick", "completed": True},
                     {"actorCellId": 5, "championId": 103, "type": "pick", "completed": True}]],
        "bans": {"myTeamBans": [84, 0], "theirTeamBans": [12]},
    }


class DraftParsingTests(unittest.TestCase):
    def test_internal_zero_is_a_real_champion(self):
        result = parse_draft_session(draft(), MAPPING)
        self.assertEqual(result.picks[0], DraftPick("blue", 266, 0, 0, 0))
        self.assertEqual(result.blue_bans, (2,))
        self.assertEqual(result.red_bans, (4,))

    def test_red_side_never_inferred_from_cell_zero(self):
        result = parse_draft_session(draft(200), MAPPING)
        self.assertEqual(result.own_team, "red")
        self.assertEqual(result.picks[0].team, "red")
        self.assertEqual(result.picks[1].team, "blue")
        self.assertEqual(result.blue_bans, (4,))
        self.assertEqual(result.red_bans, (2,))

    def test_gameflow_identity_resolves_actual_red_side(self):
        gameflow = {"gameData": {"teamOne": [{"summonerId": 99}], "teamTwo": [{"summonerId": 42}]}}
        self.assertEqual(resolve_own_side(draft(None), gameflow), "red")

    def test_current_summoner_puuid_can_resolve_side(self):
        value = draft(None)
        value["myTeam"][0].pop("summonerId")
        gameflow = {"gameData": {"teamOne": [{"puuid": "local-only"}], "teamTwo": []}}
        self.assertEqual(resolve_own_side(value, gameflow, {"puuid": "local-only"}), "blue")

    def test_unknown_side_requires_explicit_review(self):
        result = parse_draft_session(draft(None), MAPPING)
        self.assertIsNone(result.own_team)
        self.assertEqual([pick.team for pick in result.picks], ["own", "enemy"])
        self.assertEqual(result.own_bans, (2,))
        self.assertEqual(result.blue_bans, ())
        self.assertTrue(result.warnings)

    def test_enemy_roles_stay_unresolved(self):
        result = parse_draft_session(draft(), MAPPING)
        self.assertIsNone(result.picks[1].role)

    def test_hover_and_intent_are_not_imported(self):
        value = draft()
        value["actions"][0][0]["completed"] = False
        result = parse_draft_session(value, MAPPING)
        self.assertEqual([pick.riot_id for pick in result.picks], [103])
        value["myTeam"][0]["championId"] = 0
        value["actions"] = []
        result = parse_draft_session(value, MAPPING)
        self.assertEqual([pick.riot_id for pick in result.picks], [103])

    def test_champion_trades_use_current_locked_roster(self):
        value = draft()
        value["myTeam"][0]["championId"] = 32
        result = parse_draft_session(value, MAPPING)
        self.assertEqual(result.picks[0].internal_id, 6)

    def test_unknown_champions_are_reported_and_omitted(self):
        value = draft()
        value["myTeam"][0]["championId"] = 987654
        result = parse_draft_session(value, MAPPING)
        self.assertEqual(len(result.picks), 1)
        self.assertTrue(any("987654" in note for note in result.warnings))

    def test_malformed_or_non_five_player_draft_is_rejected(self):
        for value in ({}, {"myTeam": "invalid"}, {"myTeam": [], "theirTeam": "invalid"}):
            with self.assertRaises(DraftImportError):
                parse_draft_session(value, MAPPING)
        value = draft()
        value["myTeam"] *= 6
        with self.assertRaises(DraftImportError):
            parse_draft_session(value, MAPPING)

    def test_ambiguous_gameflow_is_not_guessed(self):
        gameflow = {"gameData": {"teamOne": [{"summonerId": 42}], "teamTwo": [{"summonerId": 42}]}}
        self.assertIsNone(resolve_own_side(draft(None), gameflow))


class LocalClientTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.lockfile = Path(self.folder.name) / "lockfile"
        self.lockfile.write_text("LeagueClient:1234:34567:private-test-token:https", encoding="utf-8")

    def test_lockfile_parsing_does_not_expose_token(self):
        credentials = read_lockfile(self.lockfile)
        self.assertEqual(credentials.port, 34567)
        self.assertNotIn("private-test-token", repr(credentials))

    def test_lockfile_rejects_non_local_or_invalid_connection_data(self):
        for content in ("LeagueClient:1234:34567:secret:http", "LeagueClient:1234:999999:secret:https", "not-a-lockfile"):
            self.lockfile.write_text(content, encoding="utf-8")
            with self.assertRaises(DraftImportError) as caught:
                read_lockfile(self.lockfile)
            self.assertNotIn("secret", str(caught.exception))

    def test_explicit_folder_selects_its_lockfile(self):
        self.assertEqual(discover_lockfiles(self.folder.name), [self.lockfile])

    def test_read_draft_uses_loopback_no_redirect_and_bounded_timeout(self):
        transport = MagicMock()
        response = MagicMock(status_code=200)
        response.json.return_value = draft()
        transport.get.return_value = response
        result = LcuDraftClient(self.lockfile, MAPPING, transport).read_draft()
        self.assertEqual(result.own_team, "blue")
        url = transport.get.call_args.args[0]
        options = transport.get.call_args.kwargs
        self.assertEqual(url, "https://127.0.0.1:34567/lol-champ-select/v1/session")
        self.assertFalse(options["allow_redirects"])
        self.assertEqual(options["timeout"], (1, 2))
        self.assertEqual(options["auth"], ("riot", "private-test-token"))

    def test_missing_draft_has_actionable_error(self):
        transport = MagicMock()
        transport.get.return_value.status_code = 404
        with self.assertRaisesRegex(DraftImportError, "nincs aktív hősválasztás"):
            LcuDraftClient(self.lockfile, MAPPING, transport).read_draft()

    def test_connection_error_does_not_leak_authentication(self):
        transport = MagicMock()
        transport.get.side_effect = requests.ConnectionError("private-test-token")
        with self.assertRaises(DraftImportError) as caught:
            LcuDraftClient(self.lockfile, MAPPING, transport).read_draft()
        self.assertNotIn("private-test-token", str(caught.exception))

    def test_redirect_is_not_followed_or_treated_as_a_snapshot(self):
        transport = MagicMock()
        transport.get.return_value.status_code = 302
        with self.assertRaises(DraftImportError):
            LcuDraftClient(self.lockfile, MAPPING, transport).read_draft()

    def test_unknown_side_requests_gameflow_and_retains_red_side(self):
        transport = MagicMock()
        first, second = MagicMock(status_code=200), MagicMock(status_code=200)
        first.json.return_value = draft(None)
        second.json.return_value = {"gameData": {"teamTwo": [{"summonerId": 42}], "teamOne": []}}
        transport.get.side_effect = [first, second]
        result = LcuDraftClient(self.lockfile, MAPPING, transport).read_draft()
        self.assertEqual(result.own_team, "red")
        self.assertEqual(transport.get.call_count, 2)


@unittest.skipUnless(importlib.util.find_spec("PyQt6"), "PyQt6 is not installed")
class ImportDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def make_dialog(self, snapshot):
        from view.draft_import_dialog import DraftImportDialog
        value = DraftImportDialog(snapshot, {0: "Aatrox", 1: "Ahri", 2: "Akali", 4: "Alistar"})
        self.addCleanup(value.close)
        return value

    def test_unresolved_role_blocks_application_until_assigned(self):
        value = self.make_dialog(parse_draft_session(draft(), MAPPING))
        with self.assertRaisesRegex(ValueError, "pozíciót"):
            value.draft_values()
        value.role_selectors[1].setCurrentIndex(value.role_selectors[1].findData(2))
        result = value.draft_values()
        self.assertEqual(result["blue"][0], 0)
        self.assertIsNone(result["blue"][1])
        self.assertEqual(result["red"][2], 1)

    def test_unknown_side_must_be_chosen(self):
        value = self.make_dialog(parse_draft_session(draft(None), MAPPING))
        with self.assertRaisesRegex(ValueError, "oldalát"):
            value.draft_values()
        value.own_side.setCurrentIndex(value.own_side.findData("red"))
        value.role_selectors[1].setCurrentIndex(value.role_selectors[1].findData(2))
        result = value.draft_values()
        self.assertEqual(result["red"][0], 0)
        self.assertEqual(result["blue"][2], 1)
        self.assertEqual(result["red_bans"], [2])

    def test_duplicate_roles_block_application(self):
        snap = DraftSnapshot((DraftPick("blue", 266, 0, 0, 0), DraftPick("blue", 103, 1, 0, 1)), own_team="blue")
        value = self.make_dialog(snap)
        with self.assertRaisesRegex(ValueError, "pozícióhoz"):
            value.draft_values()


if __name__ == "__main__":
    unittest.main()
