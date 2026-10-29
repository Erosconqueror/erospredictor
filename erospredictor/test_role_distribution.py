"""Role-frequency export uses observed, validated DB positions and no network."""

from copy import deepcopy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import export_role_distribution as exporter
import update_pipeline as pipeline
from services.role_distribution import (
    build_role_distribution, empty_role_distribution, load_role_distribution,
    merge_role_distributions, role_probabilities, save_role_distribution,
)
from test_update_pipeline import FakeDB, install_test_catalog, record as pipeline_record


def match(mid="one", tier="GOLD", patch_version="16.7", support_xerath=False):
    blue = [266, 64, 101, 22, 412]
    if support_xerath:
        blue[2], blue[4] = blue[4], blue[2]
    return {"match_id": mid, "tier": tier, "patch": patch_version, "region": "euw1",
            "blue_team": blue, "red_team": [86, 11, 103, 51, 201], "blue_win": True}


class RoleDistributionTests(unittest.TestCase):
    def test_conditional_role_frequency_uses_both_teams_and_real_sample_counts(self):
        rows = [match(str(i), support_xerath=i >= 8) for i in range(10)]
        before = deepcopy(rows)
        document = build_role_distribution(rows, patches=["16.7"], region="euw1", tiers=["GOLD"])
        self.assertEqual(document["tiers"]["GOLD"]["101"]["counts"], [0, 0, 8, 0, 2])
        self.assertEqual(role_probabilities(document, 101, "GOLD"), ([0.0, 0.0, 0.8, 0.0, 0.2], 10))
        self.assertEqual(role_probabilities(document, 201, "GOLD"), ([0, 0, 0, 0, 1], 10))
        self.assertEqual(sum(entry["total"] for entry in document["tiers"]["MIXED"].values()), 100)
        self.assertEqual(rows, before)

    def test_scope_filters_deduplication_and_invalid_teams_do_not_bias_counts(self):
        bad = match("bad")
        bad["red_team"][0] = bad["blue_team"][0]
        zero = match("zero")
        zero["blue_team"][0] = 0
        old, foreign = match("old", patch_version="16.6"), dict(match("foreign"), region="kr")
        data = [match(), match(), bad, zero, old, foreign, match("other", tier="DIAMOND")]
        doc = build_role_distribution(data, patches=["16.7"], region="euw1", tiers=["GOLD"])
        self.assertEqual(doc["scope"]["matches"], 1)
        self.assertEqual(doc["scope"]["excluded_records"], {"duplicate_match": 1, "invalid_teams": 2, "outside_scope": 3})
        self.assertEqual(doc["tiers"]["MIXED"]["266"]["total"], 1)

    def test_postgres_array_strings_are_supported_and_fractional_ids_are_rejected(self):
        row = match()
        row["blue_team"] = "{266,64,101,22,412}"
        doc = build_role_distribution([row])
        self.assertEqual(role_probabilities(doc, "266"), ([1, 0, 0, 0, 0], 1))
        invalid = match()
        invalid["blue_team"][0] = 266.9
        self.assertEqual(build_role_distribution([invalid])["scope"]["matches"], 0)

    def test_mixed_counts_are_not_rank_or_outcome_weighted(self):
        rows = [match("gold1"), match("gold2"), match("diamond", tier="DIAMOND")]
        rows[1]["blue_win"] = False
        doc = build_role_distribution(rows)
        self.assertEqual(doc["tiers"]["GOLD"]["266"]["total"], 2)
        self.assertEqual(doc["tiers"]["DIAMOND"]["266"]["total"], 1)
        self.assertEqual(doc["tiers"]["MIXED"]["266"]["total"], 3)

    def test_unknown_or_corrupt_data_falls_back_without_an_invented_sample(self):
        doc = build_role_distribution([match()])
        self.assertEqual(role_probabilities(doc, 101, "SILVER")[1], 1)
        self.assertEqual(role_probabilities(doc, 999999), ([0.2] * 5, 0))
        self.assertEqual(role_probabilities(dict(doc, version=2), 101), ([0.2] * 5, 0))
        self.assertEqual(role_probabilities(dict(doc, roles=list(reversed(doc["roles"]))), 101), ([0.2] * 5, 0))
        for invalid in (None, {}, {"tiers": {"MIXED": {"101": {"total": 100, "probabilities": [1, 0, 0, 0, 0]}}}}):
            self.assertEqual(role_probabilities(invalid, 101), ([0.2] * 5, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roles.json"
            self.assertEqual(load_role_distribution(path), empty_role_distribution())
            path.write_text("broken JSON")
            self.assertEqual(load_role_distribution(path), empty_role_distribution())
            doc["tiers"]["MIXED"]["101"]["probabilities"][0] = float("nan")
            path.write_text(json.dumps(doc))
            self.assertEqual(load_role_distribution(path), empty_role_distribution())

    def test_partial_refresh_preserves_other_ranks_and_their_scopes(self):
        previous = build_role_distribution([match("old", tier="DIAMOND", patch_version="16.6")], patches=["16.6"])
        current = build_role_distribution([match("new")], patches=["16.7"], tiers=["GOLD"])
        before = deepcopy(previous)
        merged = merge_role_distributions(previous, current, ["GOLD"])
        self.assertEqual(merged["tiers"]["DIAMOND"], previous["tiers"]["DIAMOND"])
        self.assertEqual(merged["tier_scopes"]["DIAMOND"]["patches"], ["16.6"])
        self.assertEqual(merged["tier_scopes"]["MIXED"]["tiers"], ["GOLD"])
        self.assertEqual(merged["scope"]["preserved_tiers"], ["DIAMOND"])
        self.assertEqual(previous, before)

    def test_atomic_write_keeps_previous_file_on_failed_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roles.json"
            path.write_bytes(b"original")
            with patch("services.update_progress.os.replace", side_effect=OSError("full disk")):
                with self.assertRaises(OSError):
                    save_role_distribution(path, build_role_distribution([match()]))
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).glob(".checkpoint-*")), [])


class RoleExportTests(unittest.TestCase):
    def test_help_does_not_open_database_or_import_ml(self):
        with patch.dict("sys.modules", {"torch": None, "torch_geometric": None, "psycopg2": None}), \
                patch.object(exporter, "read_records", side_effect=AssertionError("DB opened")), \
                redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as caught:
            exporter.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--all-patches", output.getvalue())

    def test_database_session_is_read_only_and_never_initializes_tables(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [("one", "euw1", "GOLD", "16.7", [1, 2, 3, 4, 5], [6, 7, 8, 9, 10])]
        connect = MagicMock(return_value=connection)
        rows = exporter.read_records(["16.7"], ["GOLD"], "euw1", connect)
        connection.set_session.assert_called_once_with(readonly=True, autocommit=True)
        query, params = cursor.execute.call_args.args
        self.assertTrue(query.startswith("SELECT "))
        self.assertNotIn("CREATE", query)
        self.assertEqual(params, (["GOLD"], "euw1", ["16.7"]))
        self.assertEqual(rows[0]["blue_team"], [1, 2, 3, 4, 5])
        connection.close.assert_called_once()

    def test_export_preserves_previous_json_when_filtered_database_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roles.json"
            path.write_text("original")
            args = exporter.parser().parse_args(["--patches", "16.7", "--output", str(path)])
            with patch.object(exporter, "read_records", return_value=[]):
                with self.assertRaisesRegex(ValueError, "Nincs érvényes"):
                    exporter.run(args, log=lambda _: None)
            self.assertEqual(path.read_text(), "original")

    def test_export_reads_only_selected_scope_then_writes_riot_id_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roles.json"
            args = exporter.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--output", str(path)])
            with patch.object(exporter, "read_records", return_value=[match()]) as read:
                self.assertEqual(exporter.run(args, log=lambda _: None), 0)
            read.assert_called_once_with(["16.7"], ["GOLD"], args.region)
            loaded = load_role_distribution(path)
            self.assertEqual(loaded["scope"]["source"], "database_export")
            self.assertEqual(loaded["tiers"]["GOLD"]["266"]["total"], 1)


class RolePublicationTests(unittest.TestCase):
    def test_resumed_publication_uses_frozen_role_counts_without_reopening_database(self):
        args = pipeline.parser().parse_args(["--patches", "16.7", "--tiers", "GOLD", "--skip-scrape",
                                            "--no-catalog", "--matches-per-tier", "4", "--min-matches", "4"])
        db = FakeDB([pipeline_record(str(i)) for i in range(4)])
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory)
            install_test_catalog(app)
            previous = build_role_distribution([match("old", tier="DIAMOND", patch_version="16.6")], patches=["16.6"])
            save_role_distribution(app / "data/champion_roles.json", previous)
            original = (app / "data/champion_roles.json").read_bytes()
            def fit(args, patches, db, records, stage, log):
                (stage / "models").mkdir()
                (stage / "models/complete.pth").write_bytes(b"new model")
                return {}, [Path("models/complete.pth")]
            with patch.object(pipeline, "APP_DIR", app), \
                    patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: db)}), \
                    patch.object(pipeline, "train_outputs", side_effect=fit) as train:
                with patch.object(pipeline, "publish_outputs", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        pipeline.run(args, log=lambda _: None)
                self.assertEqual((app / "data/champion_roles.json").read_bytes(), original)
                db.records.append(pipeline_record("after frozen snapshot"))
                with patch.dict("sys.modules", {"model.data_manager": SimpleNamespace(DataManager=lambda **kw: self.fail("DB reopened"))}):
                    self.assertEqual(pipeline.run(args, log=lambda _: None), 0)
            self.assertEqual(train.call_count, 1)
            document = load_role_distribution(app / "data/champion_roles.json")
            self.assertEqual(document["scope"]["matches"], 4)
            self.assertEqual(document["tiers"]["GOLD"]["1"]["total"], 4)
            self.assertEqual(document["tiers"]["DIAMOND"], previous["tiers"]["DIAMOND"])
            manifest = json.loads((app / "data/update_manifest.json").read_text())
            self.assertEqual(manifest["champion_role_distribution"]["matches"], 4)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"), "ML packages required for trainer import")
    def test_interactive_trainer_exports_the_preprocessors_actual_patch_window(self):
        import trainer_script
        prep = SimpleNamespace(_get_matches=lambda: [match()], allowed_patches=["16.7"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roles.json"
            with patch.object(trainer_script, "ROLE_DISTRIBUTION_PATH", path):
                document = trainer_script.refresh_role_statistics(prep)
            self.assertEqual(document["scope"]["source"], "interactive_trainer")
            self.assertEqual(load_role_distribution(path)["scope"]["patches"], ["16.7"])


if __name__ == "__main__":
    unittest.main()
