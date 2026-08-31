import unittest

import numpy as np
import torch

from pls.editflow.acquisition import ensemble_edge_uncertainty
from pls.editflow.metrics import mutation_field_metrics
from pls.editflow.objective import editflow_distillation_loss
from pls.editflow.student import EditPotentialStudent, encode_sequences


class EditFlowObjectiveTests(unittest.TestCase):
    def test_unqueried_teacher_nodes_cannot_enter_edge_loss(self):
        student = torch.tensor([0.0, 1.5, 99.0], requires_grad=True)
        teacher = torch.tensor([0.0, 1.0, float("nan")])
        edges = torch.tensor([[0, 1], [1, 2]])
        report = editflow_distillation_loss(student, teacher, edges, torch.tensor([1, 1, 0], dtype=torch.bool))
        self.assertEqual(report.queried_nodes, 2)
        self.assertEqual(report.closed_edges, 1)
        self.assertTrue(torch.isfinite(report.total))
        report.total.backward()
        self.assertEqual(student.grad[2].item(), 0.0)

    def test_edge_uncertainty_uses_effect_not_absolute_value(self):
        ensemble = np.array([[10.0, 11.0], [20.0, 21.0], [-5.0, -4.0]])
        uncertainty = ensemble_edge_uncertainty(ensemble, np.array([[0], [1]]))
        self.assertAlmostEqual(uncertainty[0], 0.0)

    def test_field_metrics_are_anchor_macro(self):
        teacher = np.array([0., 2., -1., 0., 3., 1.])
        student = np.array([0., 1., -2., 0., 2., 1.5])
        edges = np.array([[0, 0, 3, 3], [1, 2, 4, 5]])
        metrics = mutation_field_metrics(teacher, student, edges, [0, 0, 1, 1], top_k=1)
        self.assertEqual(metrics["mutation_sign_accuracy"], 1.0)
        self.assertEqual(metrics["anchor_macro_top_1_recall"], 1.0)
        self.assertEqual(metrics["anchor_macro_kendall_tau"], 1.0)

    def test_sequence_student_is_padding_invariant_and_differentiable(self):
        torch.manual_seed(37)
        model = EditPotentialStudent(32, 1, 4, 0, 16).eval()
        tokens, mask = encode_sequences(["ACDE", "AC"])
        padded_tokens = torch.nn.functional.pad(tokens[:1], (0, 5))
        padded_mask = torch.nn.functional.pad(mask[:1], (0, 5))
        expected = model(tokens[:1], mask[:1])
        actual = model(padded_tokens, padded_mask)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        model.train(); output = model(tokens, mask); output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.amino_acid.weight.grad).all())


if __name__ == "__main__":
    unittest.main()

