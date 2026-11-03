"""Offline joint role assignment; observed roles never become duplicate lanes."""

from copy import deepcopy
import unittest

from services.draft_import import DraftPick, DraftSnapshot
from services.draft_assignment import assign_draft


def entry(counts):
    total = sum(counts)
    return {"counts": counts, "probabilities": [value / total for value in counts], "total": total}


def distribution(**champions):
    return {"version": 1, "roles": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"],
            "tiers": {"MIXED": {key: entry(value) for key, value in champions.items()}}}


def pick(riot_id, internal_id, role=None, team="enemy", cell_id=5):
    return DraftPick(team, riot_id, internal_id, role, cell_id)


class JointRoleAssignmentTests(unittest.TestCase):
    def test_xerath_braum_are_assigned_mid_and_support_jointly(self):
        stats = distribution(**{"101": [0, 0, 400, 0, 600], "201": [0, 0, 1, 0, 999]})
        snapshot = DraftSnapshot((pick(101, 10), pick(201, 20)), own_team="blue")
        before = deepcopy(snapshot)
        result = assign_draft(snapshot, stats)
        self.assertEqual(result.values["red"], [None, None, 10, None, 20])
        self.assertTrue(result.complete)
        self.assertEqual(snapshot, before)
        evidence = {item.internal_id: item for item in result.assignments}
        self.assertEqual(evidence[10].role_probability, 0.4)
        self.assertEqual(evidence[20].role_probability, 0.999)
        self.assertEqual(evidence[10].source, "statistics")

    def test_full_roster_has_at_most_one_champion_in_each_role(self):
        stats = distribution(**{
            "1": [900, 0, 100, 0, 0], "2": [0, 950, 0, 0, 50],
            "3": [0, 0, 600, 0, 400], "4": [0, 0, 0, 990, 10], "5": [0, 0, 10, 0, 990]})
        snapshot = DraftSnapshot(tuple(pick(i + 1, i) for i in range(5)), own_team="blue")
        result = assign_draft(snapshot, stats)
        self.assertEqual(result.values["red"], [0, 1, 2, 3, 4])
        self.assertEqual(len(set(result.values["red"])), 5)

    def test_own_reported_role_is_hard_and_internal_zero_is_real(self):
        stats = distribution(**{"266": [0, 0, 0, 0, 1000], "103": [900, 0, 100, 0, 0]})
        snapshot = DraftSnapshot((pick(266, 0, 0, "own"), pick(103, 1, None, "own")), own_team="blue")
        result = assign_draft(snapshot, stats)
        self.assertEqual(result.values["blue"], [0, None, 1, None, None])
        self.assertEqual(result.assignments[0].source, "client")
        self.assertEqual(result.assignments[0].role_probability, 0.0)

    def test_enemy_reported_role_is_not_trusted(self):
        snapshot = DraftSnapshot((pick(101, 10, 4), pick(201, 20, 4)), own_team="red")
        stats = distribution(**{"101": [0, 0, 700, 0, 300], "201": [0, 0, 0, 0, 1000]})
        result = assign_draft(snapshot, stats)
        self.assertEqual(result.values["blue"][2], 10)
        self.assertEqual(result.values["blue"][4], 20)

    def test_missing_data_leaves_unknown_pick_unresolved_instead_of_using_uniform_fallback(self):
        snapshot = DraftSnapshot((pick(999, 5), pick(201, 20)), own_team="blue")
        result = assign_draft(snapshot, distribution(**{"201": [0, 0, 0, 0, 100]}))
        self.assertEqual(result.values["red"][4], 20)
        self.assertEqual(result.unresolved, (pick(999, 5, team="red"),))
        self.assertFalse(result.complete)
        self.assertIsNone(result.assignments[0].role_probability)
        self.assertEqual(result.assignments[0].probabilities, ())

    def test_two_only_support_champions_do_not_gain_an_invented_zero_percent_lane(self):
        snapshot = DraftSnapshot((pick(101, 10), pick(201, 20)), own_team="blue")
        result = assign_draft(snapshot, distribution(**{"101": [0, 0, 0, 0, 100], "201": [0, 0, 0, 0, 1000]}))
        self.assertEqual(result.values["red"][4], 20)
        self.assertEqual(len(result.unresolved), 1)
        self.assertEqual(result.unresolved[0].internal_id, 10)
        self.assertEqual(sum(cid is not None for cid in result.values["red"]), 1)

    def test_manual_role_resolves_missing_statistics_and_client_still_takes_priority(self):
        snapshot = DraftSnapshot((pick(266, 0, 0, "own"), pick(999, 5)), own_team="blue")
        result = assign_draft(snapshot, {}, previous_roles={("blue", 0): 4, ("red", 5): 2})
        self.assertEqual(result.values["blue"][0], 0)
        self.assertEqual(result.values["red"][2], 5)
        self.assertTrue(result.complete)
        self.assertEqual(result.assignments[1].source, "manual")
        self.assertIsNone(result.assignments[1].role_probability)
        self.assertTrue(any("felülírta" in warning for warning in result.warnings))

    def test_conflicting_own_roles_and_manual_roles_fail_without_mutation(self):
        snapshot = DraftSnapshot((pick(266, 0, 0, "own"), pick(103, 1, 0, "own")), own_team="blue")
        with self.assertRaisesRegex(ValueError, "ütköznek"):
            assign_draft(snapshot, {})
        other = DraftSnapshot((pick(266, 0), pick(103, 1)), own_team="blue")
        with self.assertRaisesRegex(ValueError, "ütköznek"):
            assign_draft(other, {}, previous_roles={("red", 0): 0, ("red", 1): 0})

    def test_a_teammates_client_role_also_overrides_a_stale_manual_assignment(self):
        snapshot = DraftSnapshot((pick(103, 0, None, "own"), pick(266, 5, 0, "own")), own_team="blue")
        result = assign_draft(snapshot, distribution(**{"103": [900, 0, 100, 0, 0]}),
                              previous_roles={("blue", 0): 0})
        self.assertEqual(result.values["blue"][0], 5)
        self.assertEqual(result.values["blue"][2], 0)
        self.assertEqual(result.assignments[0].source, "statistics")
        self.assertTrue(any("csapattárshoz" in warning for warning in result.warnings))

    def test_side_must_be_known_and_conflicting_explicit_side_is_rejected(self):
        snapshot = DraftSnapshot((pick(266, 0, 0, "own"),))
        with self.assertRaisesRegex(ValueError, "oldala nem biztos"):
            assign_draft(snapshot, {})
        result = assign_draft(snapshot, {}, own_side="red")
        self.assertEqual(result.values["red"][0], 0)
        with self.assertRaisesRegex(ValueError, "ellentmond"):
            assign_draft(DraftSnapshot((), own_team="blue"), {}, own_side="red")

    def test_bans_follow_own_side_and_duplicate_cross_team_bans_are_allowed(self):
        snapshot = DraftSnapshot((pick(266, 0, 0, "own"),), own_team="red", own_bans=(5,), enemy_bans=(5, 6))
        result = assign_draft(snapshot, {})
        self.assertEqual(result.values["red_bans"], [5])
        self.assertEqual(result.values["blue_bans"], [5, 6])
        with self.assertRaisesRegex(ValueError, "tiltások"):
            assign_draft(DraftSnapshot((pick(266, 0, 0, "own"),), own_team="blue", blue_bans=(0,)), {})

    def test_roster_display_order_does_not_choose_lane(self):
        stats = distribution(**{"101": [0, 0, 400, 0, 600], "201": [0, 0, 1, 0, 999]})
        first = DraftSnapshot((pick(101, 10, cell_id=9), pick(201, 20, cell_id=5)), own_team="blue")
        second = DraftSnapshot(tuple(reversed(first.picks)), own_team="blue")
        self.assertEqual(assign_draft(first, stats).values, assign_draft(second, stats).values)

    def test_empty_partial_draft_keeps_bans_and_no_fake_assignments(self):
        result = assign_draft(DraftSnapshot((), own_team="blue", blue_bans=(0,)), {})
        self.assertEqual(result.values["blue"], [None] * 5)
        self.assertEqual(result.values["blue_bans"], [0])
        self.assertTrue(result.complete)
        self.assertEqual(result.assignments, ())

    def test_tier_specific_distribution_is_used_with_mixed_fallback(self):
        stats = distribution(**{"101": [0, 0, 800, 0, 200]})
        stats["tiers"]["GOLD"] = {"101": entry([0, 0, 100, 0, 900])}
        snapshot = DraftSnapshot((pick(101, 10),), own_team="blue")
        self.assertEqual(assign_draft(snapshot, stats, tier="GOLD").values["red"][4], 10)
        self.assertEqual(assign_draft(snapshot, stats, tier="MISSING").values["red"][2], 10)

    def test_under_five_percent_role_cannot_resolve_partial_team_conflict(self):
        stats = distribution(**{"101": [0, 0, 4, 0, 96], "201": [0, 0, 0, 0, 100]})
        snapshot = DraftSnapshot((pick(101, 10), pick(201, 20)), own_team="blue")
        result = assign_draft(snapshot, stats)
        self.assertEqual(sum(cid is not None for cid in result.values["red"]), 1)
        self.assertTrue(any("5%" in note for note in result.warnings))
        self.assertFalse(any(item.source == "forced" for item in result.assignments))
        stats["tiers"]["MIXED"]["101"] = entry([0, 0, 5, 0, 95])
        result = assign_draft(snapshot, stats)
        self.assertEqual(result.values["red"][2], 10)
        self.assertEqual(result.values["red"][4], 20)

    def test_only_full_enemy_roster_can_force_a_rare_or_unobserved_position(self):
        stats = distribution(**{"1": [100, 0, 0, 0, 0], "2": [0, 100, 0, 0, 0],
                                "3": [0, 0, 4, 0, 96], "4": [0, 0, 0, 100, 0], "5": [0, 0, 0, 0, 100]})
        full = DraftSnapshot(tuple(pick(i + 1, i) for i in range(5)), own_team="blue")
        result = assign_draft(full, stats)
        self.assertTrue(result.complete)
        self.assertEqual(result.values["red"], [0, 1, 2, 3, 4])
        self.assertEqual(result.assignments[2].source, "forced")
        self.assertEqual(result.assignments[2].role_probability, .04)
        self.assertTrue(result.scenarios[0].forced)
        own = DraftSnapshot(tuple(pick(i + 1, i, team="own") for i in range(5)), own_team="blue")
        self.assertFalse(assign_draft(own, stats).complete)

    def test_alternative_layouts_are_unique_normalized_and_obey_the_threshold(self):
        stats = distribution(**{"101": [0, 0, 60, 0, 40], "201": [0, 0, 30, 0, 70]})
        result = assign_draft(DraftSnapshot((pick(101, 10), pick(201, 20)), own_team="blue"), stats)
        self.assertEqual(len(result.scenarios), 2)
        self.assertAlmostEqual(sum(item.weight for item in result.scenarios), 1)
        self.assertEqual({item.red for item in result.scenarios}, {(None, None, 10, None, 20), (None, None, 20, None, 10)})
        self.assertTrue(all(not item.forced for item in result.scenarios))


if __name__ == "__main__":
    unittest.main()
