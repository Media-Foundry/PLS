import unittest

import numpy as np

from pls.editflow.decision_gating import (
    certified_best_from_intervals,
    exact_best_rank,
    finite_sample_quantile,
    query_until_certified,
    shrinkage_slope,
    top_m_exact_verification,
)


class EditFlowDecisionGatingTests(unittest.TestCase):
    def test_top_m_reports_exact_decision_regret(self):
        low = np.asarray([3.0, 2.0, 0.0, 4.0, 1.0, 0.0])
        exact = np.asarray([1.0, 5.0, 0.0, 4.0, 2.0, 0.0])
        groups = np.asarray([0, 0, 0, 1, 1, 1])
        top_one = top_m_exact_verification(low, exact, groups, 1)
        top_two = top_m_exact_verification(low, exact, groups, 2)
        self.assertAlmostEqual(top_one["true_best_inclusion"], 0.5)
        self.assertAlmostEqual(top_one["mean_regret"], 2.0)
        self.assertAlmostEqual(top_two["true_best_inclusion"], 1.0)
        self.assertAlmostEqual(top_two["mean_regret"], 0.0)

    def test_interval_certificate_requires_strict_separation(self):
        self.assertEqual(certified_best_from_intervals([2.0, 0.0], [3.0, 1.0]), 0)
        self.assertIsNone(certified_best_from_intervals([1.0, 0.0], [3.0, 1.0]))

    def test_query_policy_is_correct_when_intervals_cover(self):
        low = np.asarray([1.0, 0.9, -1.0])
        exact = np.asarray([0.8, 1.0, -0.9])
        result = query_until_certified(low, exact, radius=0.25)
        self.assertTrue(result["simultaneous_coverage"])
        self.assertTrue(result["correct"])
        self.assertLessEqual(result["queries"], 3)

    def test_finite_quantile_and_shrinkage_are_well_formed(self):
        self.assertEqual(finite_sample_quantile([1.0, 2.0, 3.0], alpha=0.25), 3.0)
        slope = shrinkage_slope([1.0], [2.0], global_slope=1.0, shrinkage=1.0)
        self.assertAlmostEqual(slope, 1.5)

    def test_exact_best_rank_uses_cheap_order(self):
        self.assertEqual(exact_best_rank([3.0, 2.0, 1.0], [0.0, 5.0, 1.0]), 2)


if __name__ == "__main__":
    unittest.main()
