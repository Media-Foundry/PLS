import unittest

import numpy as np

from pls.editflow.cost_aware_gating import calibrate_policies, select_candidates


class CostAwareGatingTest(unittest.TestCase):
    def setUp(self):
        self.model = {
            "length_knots": [50.0, 200.0],
            "seconds_knots": [1.0, 4.0],
            "reference_cost_seconds": 2.0,
        }

    def _row(self, component, length, low, exact):
        size = len(low)
        return {
            "anchor": int(component),
            "component": str(component),
            "length": length,
            "target_digests": [f"{component}-{i}" for i in range(size)],
            "target_node_indices": list(range(size)),
            "edge_indices": list(range(size)),
            "low": np.asarray(low, dtype=float),
            "exact": np.asarray(exact, dtype=float),
        }

    def test_calibration_and_selection_use_same_frozen_scale(self):
        landscapes = [
            self._row(0, 50, [1.0, 0.8], [0.0, 1.0]),
            self._row(1, 200, [1.0, 0.8], [0.0, 1.0]),
        ]
        policy = {"policy_id": "p", "epsilon": 0.0, "gamma": 1.0, "alpha": 0.5}
        calibrated = calibrate_policies(landscapes, [policy], self.model)[0]
        # Cheap gap .2 divided by scale: .1 for the cheap anchor and .4 for
        # the expensive anchor.  The finite-sample median rank selects .4.
        self.assertAlmostEqual(calibrated["quantile"], 0.4)
        selected = select_candidates(landscapes, calibrated, self.model)
        self.assertEqual([row["selected_queries"] for row in selected], [2, 2])


if __name__ == "__main__":
    unittest.main()
