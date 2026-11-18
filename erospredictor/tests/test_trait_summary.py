"""Large profile deviations only; no model calls or penalties for unknowns."""

import unittest
from unittest.mock import MagicMock, patch
from PyQt6.QtTest import QTest
from services.champion_traits import (TraitLibrary, composition_outliers,
                                      recommendation_badges, recommendation_team_digest)


def library_and_roles(high_reference=False):
    basics = {"champions": {str(i): {} for i in range(1, 11)}}
    overrides = {str(i): {"all": {"frontline": 1 if i <= 5 else 5,
                                  "range": 1 if i <= 5 else 5}} for i in range(1, 11)}
    entries = {}
    for i in range(1, 11):
        n = 10000 if (i > 5) == high_reference else 1
        entries[str(i)] = {"counts": [n]*5, "probabilities": [.2]*5, "total": 5*n}
    from services.role_distribution import ROLE_NAMES
    return TraitLibrary(basics, {}, overrides), {"version": 1, "roles": list(ROLE_NAMES), "tiers": {"MIXED": entries}}


class DeviationTests(unittest.TestCase):
    def test_recommendation_badges_are_short_and_unknown_is_not_negative(self):
        library, roles = library_and_roles()
        library.overrides["6"]["all"]["peel"] = 1
        badges = recommendation_badges(library, "6", 0, "MIXED", roles)
        self.assertLessEqual(len(badges), 5)
        self.assertEqual(sum(badge["tone"] == "positive" for badge in badges), 1)
        self.assertEqual(sum(badge["tone"] == "negative" for badge in badges), 1)
        self.assertEqual(recommendation_badges(TraitLibrary({}, {}, {}), "1", 0, "MIXED"), [])

    def test_team_digest_is_limited_and_empty_draft_has_no_invented_needs(self):
        library, roles = library_and_roles(True)
        digest = recommendation_team_digest(library, [(str(i), i-1) for i in range(1, 6)], "MIXED", roles)
        self.assertIn("Frontline", digest["needs"])
        self.assertLessEqual(len(digest["needs"])+len(digest["character"]), 8)
        self.assertLessEqual(len(digest["needs"]), 5)
        self.assertLessEqual(len(digest["character"]), 5)
        self.assertEqual(recommendation_team_digest(library, [], "MIXED", roles)["needs"], [])

    def test_hero_budget_is_three_two_or_two_three_without_padding(self):
        from services.champion_traits import LABELS
        keys = [key for key in LABELS if key not in ("range", "reach", "spacing")]
        for positives, negatives in ((8, 3), (3, 8), (3, 3), (8, 0)):
            with self.subTest(positives=positives, negatives=negatives):
                values = dict.fromkeys(keys[:positives], 5)
                values.update(dict.fromkeys(keys[positives:positives+negatives], 1))
                library = TraitLibrary({}, {}, {"1": {"all": values}})
                badges = recommendation_badges(library, "1", 0, "MIXED")
                plus = sum(tag["tone"] == "positive" for tag in badges)
                minus = sum(tag["tone"] == "negative" for tag in badges)
                self.assertLessEqual(plus+minus, 5)
                self.assertLessEqual(max(plus, minus), 3)
                self.assertLessEqual(min(plus, minus), 2)
                if negatives:
                    self.assertEqual((plus, minus), (2, 3) if negatives > positives else (3, 2))
                if negatives == 0:
                    self.assertEqual(minus, 0)

    def test_team_budget_is_five_three_or_four_four_and_extra_slots_need_extremes(self):
        from services.champion_traits import _balanced_tags
        for strengths, needs, expected in ((6, 3, (5, 3)), (3, 6, (3, 5)), (4, 6, (4, 4))):
            candidates = [dict(label=str(i), tone=tone, priority=10-i*.01, extreme=True)
                          for tone, count in (("positive", strengths), ("negative", needs)) for i in range(count)]
            selected = _balanced_tags(candidates, team=True)
            self.assertEqual((sum(t["tone"] == "positive" for t in selected),
                              sum(t["tone"] == "negative" for t in selected)), expected)
            for item in candidates:
                item["extreme"] = False
            self.assertLessEqual(len(_balanced_tags(candidates, team=True)), 4)
            self.assertLessEqual(len(_balanced_tags(candidates)), 3)

    def test_prediction_team_summary_applies_the_same_budget(self):
        from services.champion_traits import composition_summary, LABELS
        tags = [dict(key=key, label=label, descriptive=False, direction="high" if i < 6 else "low",
                     score=5 if i < 6 else 1, delta=2 if i < 6 else -2, deviation=2, known=5, mean=3)
                for i, (key, label) in enumerate(list(LABELS.items())[:12])]
        with patch("services.champion_traits.composition_outliers", return_value={"tags": tags, "picks": 5}):
            summary = composition_summary(None, [])
        self.assertEqual(len(summary["tags"]), 8)
        positive = sum(tag["direction"] == "high" for tag in summary["tags"])
        self.assertIn((positive, 8-positive), ((5, 3), (4, 4), (3, 5)))

    def test_high_and_low_outliers_and_range_description(self):
        library, roles = library_and_roles()
        result = composition_outliers(library, [(str(i), i-6) for i in range(6, 11)], distributions=roles)
        tags = {tag["key"]: tag for tag in result["tags"]}
        self.assertEqual(tags["frontline"]["direction"], "high")
        self.assertGreater(tags["frontline"]["delta"], .5)
        self.assertTrue(tags["range"]["descriptive"])
        self.assertNotIn("Erős", tags["range"]["label"])
        library, roles = library_and_roles(True)
        result = composition_outliers(library, [(str(i), i-1) for i in range(1, 6)], distributions=roles)
        self.assertEqual(next(tag for tag in result["tags"] if tag["key"] == "frontline")["direction"], "low")

    def test_no_comparison_without_reference_or_known_traits(self):
        library, roles = library_and_roles()
        result = composition_outliers(library, [("6", 0)])
        self.assertEqual(result["tags"], [])
        self.assertIn("Kevés profiladat", result["note"])
        result = composition_outliers(library, [("unknown", role) for role in range(5)], distributions=roles)
        self.assertEqual(result["tags"], [])

    def test_same_visible_roles_and_repeatable_reference(self):
        library, roles = library_and_roles()
        picks = [("6", 2)]
        first = composition_outliers(library, picks, distributions=roles)
        second = composition_outliers(library, picks, distributions=roles)
        self.assertEqual(first, second)
        self.assertEqual(first["picks"], 1)
        self.assertEqual(first["references"], 128)


class InlineSummaryTests(unittest.TestCase):
    def test_recommendation_shows_badges_and_team_footer_without_changing_scores(self):
        library, roles = library_and_roles()
        library.overrides["6"]["all"]["peel"] = 1
        self.window._pool_mapping = {"6": 0, "7": 1}
        self.window.chk_traits.setChecked(True)
        self.controller.recommend_champs.return_value = [dict(id=0, wr=53.5, display_color="#FFFFFF")]
        with patch.object(self.window, "_get_trait_library", return_value=library), \
                patch.object(self.window, "_load_roles", return_value=(roles, False)):
            self.window._on_recommend()
            self.wait()
        self.assertIn("+ Frontline", self.window.lbl_recommend.text())
        self.assertIn("Peel / disengage", self.window.lbl_recommend.text())
        self.assertFalse(self.window.lbl_recommend_team.isHidden())
        self.assertEqual(self.window._last_recommendations[0]["wr"], 53.5)
        self.controller.predict_match.assert_not_called()
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.assertTrue(self.window.lbl_recommend_team.isHidden())

    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFontDatabase
        cls.app = QApplication.instance() or QApplication([])
        cls.font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")

    @classmethod
    def tearDownClass(cls):
        from PyQt6.QtGui import QFontDatabase
        if cls.font_id >= 0:
            QFontDatabase.removeApplicationFont(cls.font_id)

    def setUp(self):
        from ui_test_support import isolate_preferences
        isolate_preferences(self)
        from view.main_window import MainWindow
        self.window = MainWindow(["Aatrox", "Ahri"], {"Aatrox": 0, "Ahri": 1})
        self.addCleanup(self.window.close)
        self.controller = MagicMock()
        self.controller.validate_draft.return_value = (True, "")
        self.controller.predict_match.return_value = dict(blue_text="Kék: 55%", red_text="Piros: 45%",
            bar_val=55, blue_win_prob=55, red_win_prob=45)
        self.window.controller = self.controller

    def wait(self):
        for _ in range(300):
            if not self.window._busy:
                break
            QTest.qWait(10)
        self.assertFalse(self.window._busy)

    def test_disabled_feature_does_not_load_or_show_summary(self):
        with patch.object(self.window, "_get_trait_library", side_effect=AssertionError("disabled")):
            self.window._on_predict()
            self.wait()
        self.assertTrue(self.window.trait_summary_panel.isHidden())
        self.controller.predict_match.assert_called_once()

    def test_bordered_chips_wrap_at_small_window_without_hiding_percentages(self):
        self.window.chk_traits.setChecked(True)
        self.window.resize(1180, 820)
        badges = [dict(label=label, tone=tone, score=score) for label, tone, score in (
            ("Frontline", "positive", 5), ("CC alkalmazhatóság", "positive", 4),
            ("Peel / disengage", "positive", 4), ("Mobilitás", "negative", 2),
            ("Waveclear", "negative", 1))]
        recs = [dict(id=0, wr=53.5, display_color="#FFFFFF", experimental_badges=badges)]
        context = dict(team="blue", r_idx=0, div="MIXED")
        self.window._display_recommendations(recs, context)
        self.window.show()
        QTest.qWait(50)
        browser = self.window.lbl_recommend
        self.assertLessEqual(browser.document().size().width(), browser.viewport().width()+1)
        self.assertIn("53.50%", browser.toPlainText())
        self.assertEqual(browser.text().count("data:image/png;base64,"), 5)

    def test_maximum_score_is_visually_distinct_from_threshold_score(self):
        from view.trait_summary import _chip_image
        threshold, _, _ = _chip_image("+ Frontline", "positive", 4)
        maximum, _, _ = _chip_image("+ Frontline", "positive", 5)
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QByteArray
        def image(uri):
            return QImage.fromData(QByteArray.fromBase64(uri.split(",", 1)[1].encode()))
        a, b = image(threshold), image(maximum)
        # Sample the middle of the upper border: same border/text tint,
        # with stronger green and a brighter border for the 5/5 profile.
        self.assertGreater(b.pixelColor(b.width()//2, 2).green(), a.pixelColor(a.width()//2, 2).green())

    def test_profile_failure_preserves_prediction_and_invalidation_hides_tags(self):
        self.window.chk_traits.setChecked(True)
        with patch.object(self.window, "_get_trait_library", side_effect=ValueError("profile")):
            self.window._on_predict()
            self.wait()
        self.assertEqual(self.window.bar_wr.value(), 55)
        self.assertFalse(self.window.trait_summary_panel.isHidden())
        self.assertIn("nem elérhető", self.window.lbl_trait_blue.text())
        self.controller.predict_match.assert_called_once()
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.assertTrue(self.window.trait_summary_panel.isHidden())

    def test_prediction_displays_deviations_without_extra_model_calls(self):
        library, roles = library_and_roles()
        self.window._pool_mapping = {"6": 0, "7": 1}
        self.window.blue_picks[0].setCurrentText("Aatrox")
        self.window.red_picks[0].setCurrentText("Ahri")
        self.window.chk_traits.setChecked(True)
        with patch.object(self.window, "_get_trait_library", return_value=library), \
                patch.object(self.window, "_load_roles", return_value=(roles, False)):
            self.window._on_predict()
            self.wait()
        self.assertIn("Erős Frontline", self.window.lbl_trait_blue.text())
        self.assertIn("referenciaátlag", self.window.lbl_trait_blue.toolTip())
        self.assertEqual(self.window.bar_wr.value(), 55)
        self.assertEqual(self.controller.predict_match.call_count, 2)  # Prediction + existing focus baseline.
