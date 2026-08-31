import unittest

import numpy as np

from pls.editflow.acquisition import frontier_node_acquisition, path_edge_occupancy
from pls.editflow.mutations import (Substitution, apply_substitution,
                                    enumerate_single_substitutions)


class EditFlowAcquisitionTests(unittest.TestCase):
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

