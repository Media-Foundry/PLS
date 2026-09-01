import unittest

import numpy as np

from pls.editflow.acquisition import (conformal_edge_error_envelope,
                                      cost_aware_frontier_node_acquisition,
                                      frontier_node_acquisition,
                                      path_concentration,
                                      path_edge_occupancy,
                                      prequential_frontier_edge_calibration)
from pls.editflow.mutations import (Substitution, apply_substitution,
                                    enumerate_single_substitutions)


class EditFlowAcquisitionTests(unittest.TestCase):
    def test_prequential_calibration_uses_only_newly_purchased_targets(self):
        ensemble = np.array([[0.0, 1.0, 4.0], [0.0, 1.2, 6.0]])
        teacher = np.array([0.0, 2.0, 1000.0])
        edges = np.array([[0, 0], [1, 2]])
        calibration = prequential_frontier_edge_calibration(
            ensemble, teacher, edges, purchased_targets=[1]
        )
        np.testing.assert_array_equal(calibration.edge_index, [[0], [1]])
        self.assertAlmostEqual(calibration.absolute_error[0], 0.9)
        # An unpurchased target's hidden teacher value cannot change the result.
        changed = teacher.copy(); changed[2] = -1000.0
        replay = prequential_frontier_edge_calibration(
            ensemble, changed, edges, purchased_targets=[1]
        )
        np.testing.assert_array_equal(calibration.absolute_error, replay.absolute_error)

    def test_path_concentration_normalizes_occupancy_mass(self):
        concentrated = path_concentration([0.0, 2.0, 0.0])
        self.assertEqual(concentrated.positive_support, 1)
        self.assertEqual(concentrated.effective_support, 1.0)
        self.assertEqual(concentrated.maximum_share, 1.0)
        diffuse = path_concentration([2.0, 2.0, 2.0, 2.0])
        self.assertAlmostEqual(diffuse.normalized_entropy, 1.0)
        self.assertAlmostEqual(diffuse.effective_support, 4.0)

    def test_conformal_envelope_uses_finite_sample_higher_quantile(self):
        calibrated = conformal_edge_error_envelope(
            calibration_uncertainty=[0.1, 0.2, 0.3, 0.4],
            calibration_absolute_error=[0.1, 0.3, 0.5, 0.8],
            target_uncertainty=[0.0, 0.5],
            alpha=0.25,
        )
        # ceil((4 + 1) * .75) / 4 clips to the maximum score: 0.4.
        np.testing.assert_allclose(calibrated.values, [0.4, 0.9])
        self.assertAlmostEqual(calibrated.additive_quantile, 0.4)
        self.assertEqual(calibrated.calibration_count, 4)
        self.assertEqual(calibrated.empirical_calibration_coverage, 1.0)

    def test_path_occupancy_and_frontier_node_budget(self):
        edges = np.array([[0, 0, 1, 2, 1], [1, 2, 3, 3, 4]])
        occupancy = path_edge_occupancy(edges, [[0, 1, 3], [0, 2, 3], [0, 1, 4]])
        np.testing.assert_allclose(occupancy, [2/3, 1/3, 1/3, 1/3, 1/3])
        uncertainty = np.array([0.2, 0.9, 0.5, 0.1, 0.7])
        selected = frontier_node_acquisition(edges, uncertainty, occupancy, {0}, 1)
        self.assertEqual(selected.node_indices.tolist(), [2])
        self.assertAlmostEqual(selected.scores[0], 0.3)
        self.assertEqual(selected.candidate_edges, 2)

    def test_query_cost_is_unique_target_nodes(self):
        edges = np.array([[0, 1, 0], [2, 2, 3]])
        selected = frontier_node_acquisition(edges, [1, 2, 0.5], [1, 1, 1], {0, 1}, 2, reduction="sum")
        self.assertEqual(selected.node_indices.tolist(), [2, 3])
        np.testing.assert_allclose(selected.scores, [3.0, 0.5])

    def test_nonuniform_cost_policy_uses_value_per_cost_and_exact_spend(self):
        edges = np.array([[0, 0, 0], [1, 2, 3]])
        batch = cost_aware_frontier_node_acquisition(
            edges,
            uncertainty=np.array([10.0, 6.0, 4.0]),
            occupancy=np.ones(3),
            queried_nodes={0},
            node_cost=np.array([1.0, 10.0, 2.0, 2.0]),
            cost_budget=4.0,
        )
        # Nodes 2 and 3 have higher value/cost and fit exactly; node 1 does not.
        self.assertEqual(batch.node_indices.tolist(), [2, 3])
        np.testing.assert_allclose(batch.node_costs, [2.0, 2.0])
        self.assertAlmostEqual(batch.total_cost, 4.0)

    def test_substitution_roundtrip_and_enumeration(self):
        edit = Substitution(1, "C", "D")
        self.assertEqual(apply_substitution("AC", edit), "AD")
        mutants = list(enumerate_single_substitutions("AC"))
        self.assertEqual(len(mutants), 38)
        self.assertEqual(len({sequence for _, sequence in mutants}), 38)
        with self.assertRaisesRegex(ValueError, "source"):
            apply_substitution("AC", Substitution(1, "A", "D"))


if __name__ == "__main__":
    unittest.main()
