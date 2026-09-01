import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from pls.editflow.multifidelity import (
    delta_metrics,
    grouped_oof_predictions,
    selective_hybrid,
)


class EditFlowMultifidelityTests(unittest.TestCase):
    def test_delta_metrics_recovers_exact_field(self):
        truth = np.asarray([1.0, -2.0, 0.5, 3.0])
        metrics = delta_metrics(truth, truth.copy(), np.asarray([0, 0, 1, 1]), top_k=1)
        self.assertAlmostEqual(metrics["edge_spearman"], 1.0)
        self.assertAlmostEqual(metrics["mutation_sign_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["edge_rmse"], 0.0)

    def test_grouped_oof_predictions_hold_out_whole_groups(self):
        features = np.arange(12, dtype=float).reshape(-1, 1)
        target = 2 * features[:, 0] + 1
        groups = np.repeat(np.arange(6), 2)
        prediction = grouped_oof_predictions(
            LinearRegression(), features, target, groups, folds=3
        )
        np.testing.assert_allclose(prediction, target, atol=1e-10)

    def test_selective_hybrid_replaces_only_highest_priority(self):
        approximate = np.zeros(4)
        exact = np.arange(1, 5, dtype=float)
        priority = np.asarray([0.0, 3.0, 1.0, 2.0])
        hybrid, selected = selective_hybrid(approximate, exact, priority, 0.5)
        self.assertEqual(set(selected.tolist()), {1, 3})
        np.testing.assert_array_equal(hybrid, [0.0, 2.0, 0.0, 4.0])


if __name__ == "__main__":
    unittest.main()
