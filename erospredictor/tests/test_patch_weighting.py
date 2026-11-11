"""Numeric patch recency and a nonzero lower bound for long windows."""

import unittest
from services.patch_weighting import patch_weights


class PatchWeightTests(unittest.TestCase):
    def test_five_patch_window_keeps_two_latest_at_full_weight(self):
        self.assertEqual(patch_weights(["16.18", "16.19", "16.20", "16.21", "16.22"]),
                         {"16.22": 1.0, "16.21": 1.0, "16.20": .9, "16.19": .8, "16.18": .7})

    def test_order_duplicates_and_year_boundaries_do_not_change_age(self):
        self.assertEqual(patch_weights(["16.9", "17.1", "16.10", "16.9", "16.24"]),
                         {"17.1": 1.0, "16.24": 1.0, "16.10": .9, "16.9": .8})

    def test_long_windows_keep_a_tenth_floor(self):
        weights = patch_weights([f"16.{number}" for number in range(1, 27)])
        self.assertEqual(weights["16.1"], .1)
        self.assertTrue(all(.1 <= weight <= 1 for weight in weights.values()))

    def test_empty_and_single_patch_windows(self):
        self.assertEqual(patch_weights([]), {})
        self.assertEqual(patch_weights(["16.22"]), {"16.22": 1.0})
