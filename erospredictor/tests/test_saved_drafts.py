"""Portable draft files preserve inputs and reject invalid or stale catalogs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services import saved_drafts as drafts

MAPPING = {"266": 0, "103": 1, "86": 2, "201": 3, "55": 4, "20": 5}
VALUES = {"blue": [0, None, 1, None, None], "red": [2, None, None, None, 3],
          "blue_bans": [4, None, None, None, None], "red_bans": [None, None, 4, None, None]}


class SavedDraftTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "draft.json"
        self.document = drafts.make_document(VALUES, MAPPING, "GOLD", "red", 4, {0: "Aatrox", 1: "Ahri"})

    def test_roundtrip_preserves_slots_bans_rank_focus_and_internal_zero(self):
        before = deepcopy(VALUES)
        drafts.save_draft(self.path, self.document)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["draft"]["blue"][0], 266)
        self.assertIsNone(saved["draft"]["blue"][1])
        self.assertEqual(saved["champion_names"]["266"], "Aatrox")
        restored = drafts.load_draft(self.path, MAPPING)
        self.assertEqual(restored.values, VALUES)
        self.assertEqual((restored.tier, restored.team, restored.role), ("GOLD", "red", 4))
        self.assertEqual(VALUES, before)

    def test_loading_after_catalog_reordering_preserves_actual_heroes(self):
        drafts.save_draft(self.path, self.document)
        changed = {riot: index + 10 for index, riot in enumerate(MAPPING)}
        restored = drafts.load_draft(self.path, changed)
        self.assertEqual(restored.values["blue"][0], changed["266"])
        self.assertEqual(restored.values["blue"][2], changed["103"])
        self.assertEqual(restored.values["red"][4], changed["201"])

    def test_partial_and_empty_drafts_are_supported(self):
        empty = {key: [-1] * 5 for key in drafts.SLOTS}
        document = drafts.make_document(empty, MAPPING, "MIXED", "blue", 2)
        drafts.save_draft(self.path, document)
        restored = drafts.load_draft(self.path, MAPPING)
        self.assertTrue(all(row == [None] * 5 for row in restored.values.values()))
        self.assertEqual(restored.role, 2)

    def test_duplicates_banned_picks_invalid_shapes_and_focus_are_rejected(self):
        changes = [dict(VALUES, red=[0, None, None, None, None]),
                   dict(VALUES, blue_bans=[4, 4, None, None, None]),
                   dict(VALUES, blue_bans=[0, None, None, None, None]),
                   dict(VALUES, blue=[0]), dict(VALUES, blue=[True, None, None, None, None])]
        for values in changes:
            with self.subTest(values=values), self.assertRaises(drafts.DraftFileError):
                drafts.make_document(values, MAPPING, "GOLD", "red", 4)
        for tier, team, role in (("UNKNOWN", "red", 4), ("GOLD", "enemy", 4), ("GOLD", "red", True)):
            with self.assertRaises(drafts.DraftFileError):
                drafts.make_document(VALUES, MAPPING, tier, team, role)

    def test_unknown_or_unrenderable_champions_are_not_silently_skipped(self):
        drafts.save_draft(self.path, self.document)
        for mapping, available in (({"266": 0}, None), (MAPPING, {0, 1, 2})):
            with self.assertRaises(drafts.DraftFileError):
                drafts.load_draft(self.path, mapping, available_ids=available)
        self.document["draft"]["red"][4] = 805
        self.path.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(drafts.DraftFileError, "805"):
            drafts.load_draft(self.path, MAPPING)

    def test_bad_json_oversize_and_unsupported_versions_fail_safely(self):
        for data in ("broken", "[1,2,3]", json.dumps(dict(self.document, version=True)),
                     json.dumps(dict(self.document, version=2)), "[" * 2000 + "]" * 2000,
                     " " * (drafts.MAX_FILE_BYTES + 1)):
            self.path.write_text(data)
            with self.assertRaises(drafts.DraftFileError):
                drafts.load_draft(self.path, MAPPING)
        self.path.write_text(json.dumps(self.document), encoding="utf-8-sig")
        self.assertEqual(drafts.load_draft(self.path, MAPPING).values, VALUES)

    def test_export_does_not_include_accounts_credentials_or_cached_predictions(self):
        self.document.update(api_key="private-key", account="private-account", prediction={"winrate": 99})
        self.document["focus"]["token"] = "private-token"
        drafts.save_draft(self.path, self.document)
        contents = self.path.read_text()
        for forbidden in ("private-key", "private-account", "private-token", "prediction"):
            self.assertNotIn(forbidden, contents)

    def test_failed_atomic_save_preserves_previous_draft_and_removes_temporary_file(self):
        drafts.save_draft(self.path, self.document)
        previous = self.path.read_bytes()
        self.document["focus"]["team"] = "blue"
        with patch("services.update_progress.os.replace", side_effect=PermissionError("locked")):
            with self.assertRaises(drafts.DraftFileError):
                drafts.save_draft(self.path, self.document)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.path.parent.glob(".checkpoint-*")), [])


if __name__ == "__main__":
    unittest.main()
