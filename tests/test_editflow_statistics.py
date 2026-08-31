import unittest

import numpy as np

from pls.editflow.statistics import (exact_sign_flip_pvalue,
                                     paired_bootstrap_interval,
                                     paired_method_summary)


class EditFlowStatisticsTests(unittest.TestCase):
    def test_exact_sign_flip_detects_consistent_paired_direction(self):
        differences = -np.arange(1, 17, dtype=float)
        self.assertLessEqual(exact_sign_flip_pvalue(differences), 2 / (2**16))

    def test_bootstrap_is_deterministic(self):
        values = np.array([-2.0, -1.0, 0.0, 1.0])
        first = paired_bootstrap_interval(values, samples=1000, seed=7)
        second = paired_bootstrap_interval(values, samples=1000, seed=7)
        self.assertEqual(first, second)

    def test_summary_counts_wins_ties_and_losses(self):
        result = paired_method_summary(
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 2.0],
            bootstrap_samples=100,
            bootstrap_seed=3,
        )
        self.assertEqual(result["path_wins"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["uncertainty_wins"], 1)


if __name__ == "__main__":
    unittest.main()
