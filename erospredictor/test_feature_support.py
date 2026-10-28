"""Checks for safe optional presentation and local model preservation."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.explanation_format import explanations_html


class ExplanationPresentationTests(unittest.TestCase):
    def record(self):
        return dict(id=0, wr=53.5, baseline_win_prob=50.0, candidate_delta_pp=3.5,
                    components=[dict(model="gnn", win_prob=53.5, weight=1.0)],
                    interactions=[dict(champion_id=1, side="opponent", role_index=0,
                                       delta_pp=1.25, is_lane_opponent=True)],
                    lane_matchup=None, bot_synergy=None)

    def test_names_are_escaped_and_no_historical_sample_is_invented(self):
        rendered = explanations_html([self.record()], {0: "<script>x</script>", 1: "A&B"})
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("A&amp;B", rendered)
        self.assertIn("Nincs elérhető minta", rendered)

    def test_only_top_five_render_without_mutating_diagnostics(self):
        records = [self.record() for _ in range(7)]
        before = copy.deepcopy(records)
        rendered = explanations_html(records, {0: "Aatrox", 1: "Ahri"})
        self.assertEqual(rendered.count("Aatrox —"), 5)
        self.assertEqual(records, before)

    def test_missing_component_is_visible(self):
        record = self.record()
        record["unavailable_models"] = {"roleaware": "private exception"}
        rendered = explanations_html([record], {0: "Aatrox", 1: "Ahri"})
        self.assertIn("Nem elérhető", rendered)
        self.assertIn("Szerepkört ismerő modell", rendered)
        self.assertNotIn("private exception", rendered)


class PreserveTrainedModelsTests(unittest.TestCase):
    def test_public_downloader_does_not_replace_local_pipeline_output(self):
        from services import updater
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "update_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with patch.object(updater.os.path, "isfile", return_value=manifest.is_file()), \
                    patch.object(updater.requests, "get") as request:
                updater.update_files()
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
