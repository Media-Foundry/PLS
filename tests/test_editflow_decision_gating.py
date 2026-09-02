import unittest

import numpy as np

from pls.editflow.decision_gating import (
    certified_best_from_intervals,
    empirical_upper_cvar,
    epsilon_optimal_nonconformity,
    exact_best_rank,
    finite_sample_quantile,
    legacy_conservative_quantile,
    margin_candidate_indices,
    query_until_certified,
    regret_summary,
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
        values = np.arange(1.0, 11.0)
        self.assertEqual(finite_sample_quantile(values, alpha=0.2), 9.0)
        self.assertEqual(legacy_conservative_quantile(values, alpha=0.2), 10.0)
        slope = shrinkage_slope([1.0], [2.0], global_slope=1.0, shrinkage=1.0)
        self.assertAlmostEqual(slope, 1.5)

    def test_epsilon_margin_and_scaled_candidate_sets(self):
        low = np.asarray([1.0, 0.8, 0.5])
        exact = np.asarray([0.0, 0.9, 1.0])
        self.assertAlmostEqual(epsilon_optimal_nonconformity(low, exact, 0.0), 0.5)
        self.assertAlmostEqual(epsilon_optimal_nonconformity(low, exact, 0.15), 0.2)
        np.testing.assert_array_equal(margin_candidate_indices(low, 0.21), [0, 1])
        np.testing.assert_array_equal(
            margin_candidate_indices(low, 0.21, scale=[1.0, 0.5, 4.0]), [0, 2]
        )

    def test_tail_regret_metrics_remain_informative(self):
        regrets = np.asarray([0.0] * 60 + [0.1, 0.2, 0.3, 1.0])
        summary = regret_summary(regrets)
        self.assertEqual(summary["failure_count"], 4)
        self.assertAlmostEqual(summary["failure_conditional_mean_regret"], 0.4)
        self.assertAlmostEqual(summary["regret_cvar95"], 0.4)
        self.assertAlmostEqual(empirical_upper_cvar(regrets, level=0.95), 0.4)

    def test_exact_best_rank_uses_cheap_order(self):
        self.assertEqual(exact_best_rank([3.0, 2.0, 1.0], [0.0, 5.0, 1.0]), 2)


if __name__ == "__main__":
    unittest.main()
