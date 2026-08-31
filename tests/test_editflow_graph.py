import unittest

import numpy as np
import torch

from pls.editflow.graph import (assert_same_queried_nodes, edge_differences,
                                exact_design_regrets,
                                exact_optimization_regret,
                                graph_sobolev_loss, path_regret_bound,
                                shortest_path_discrepancies)


class EditFlowGraphTests(unittest.TestCase):
    def test_edge_differences_and_graph_sobolev_identity(self):
        edges = torch.tensor([[0, 1, 0], [1, 2, 2]])
        teacher = torch.tensor([1.0, 3.0, 2.0])
        student = torch.tensor([4.0, 5.0, 8.0], requires_grad=True)
        torch.testing.assert_close(edge_differences(teacher, edges), torch.tensor([2.0, -1.0, 1.0]))
        delta = student - teacher
        incidence = torch.tensor([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0]])
        expected = (incidence @ delta).square().sum()
        actual = graph_sobolev_loss(student, teacher, edges, reduction="sum")
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_path_bound_upper_bounds_realized_regret(self):
        # Two routes connect anchor 0 to optimum 3; the minimum-error route is 0-2-3.
        edges = np.array([[0, 1, 0, 2], [1, 3, 2, 3]])
        teacher = np.array([0.0, 1.0, 1.5, 4.0])
        student = np.array([0.0, 2.0, 1.4, 1.8])
        report = exact_optimization_regret(teacher, student)
        self.assertEqual(report["teacher_optimum"], 3)
        self.assertEqual(report["student_choice"], 1)
        bound = path_regret_bound( student, teacher, edges, 0, 3, 1)
        self.assertLessEqual(report["regret"], bound + 1e-12)
        distances = shortest_path_discrepancies(student, teacher, edges, 0)
        self.assertAlmostEqual(bound, distances[3] + distances[1])

    def test_uniform_corollary_and_additive_constant(self):
        edges = np.array([[0, 1, 2], [1, 2, 3]])
        teacher = np.array([0.0, 1.0, 2.0, 3.0])
        student = np.array([9.0, 9.9, 11.0, 11.9])
        edge_error = np.abs(np.diff(student - teacher))
        distances = shortest_path_discrepancies(student, teacher, edges, 0)
        self.assertLessEqual(distances[3], 3 * edge_error.max() + 1e-12)
        shifted = student + 1000
        np.testing.assert_allclose(
            shortest_path_discrepancies(shifted, teacher, edges, 0), distances
        )
        self.assertEqual(
            exact_optimization_regret(teacher, shifted)["student_choice"],
            exact_optimization_regret(teacher, student)["student_choice"],
        )

    def test_query_budget_requires_identical_node_set(self):
        self.assertEqual(assert_same_queried_nodes({1, 2}, [2, 1]), frozenset({1, 2}))
        with self.assertRaisesRegex(ValueError, "identical"):
            assert_same_queried_nodes({1, 2}, {1, 3})

    def test_design_regrets_separate_acquisition_and_generalization(self):
        teacher = np.array([0.0, 4.0, 3.0, 5.0, 2.0])
        # Student selects node 2 among unqueried nodes; node 3 is the true best.
        student = np.array([0.0, -2.0, 8.0, 1.0, 0.0])
        report = exact_design_regrets(
            teacher, student, np.arange(5), queried_nodes={0, 1}
        )
        self.assertEqual(report["acquired"]["choice"], 1)
        self.assertAlmostEqual(report["acquired"]["regret"], 1.0)
        self.assertEqual(report["novel_design"]["teacher_optimum"], 3)
        self.assertEqual(report["novel_design"]["student_choice"], 2)
        self.assertAlmostEqual(report["novel_design"]["regret"], 2.0)
        self.assertEqual(report["campaign"]["choice"], 1)
        self.assertEqual(report["campaign"]["choice_source"], "acquired")
        self.assertAlmostEqual(report["campaign"]["regret"], 1.0)

    def test_design_regrets_mark_exhausted_novel_set_unavailable(self):
        report = exact_design_regrets(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            queried_nodes={0, 1},
        )
        self.assertFalse(report["novel_design"]["available"])
        self.assertIsNone(report["novel_design"]["regret"])
        self.assertAlmostEqual(report["campaign"]["regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
