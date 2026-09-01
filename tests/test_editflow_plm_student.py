import unittest

import numpy as np
import torch

from pls.editflow.plm_student import (PLMMutationDeltaHead, PLMPairDeltaHead, PLMPotentialHead,
                                      commuting_cycle_residual)
from pls.training.train_editflow_pls_plm import commuting_cycles


class EditFlowPLMStudentTests(unittest.TestCase):
    def test_potential_and_delta_heads_receive_gradients(self):
        potential = PLMPotentialHead(12, 5, dimension=8, dropout=0.0)
        value = potential(torch.randn(4, 12), torch.randn(4, 5))
        self.assertEqual(value.shape, (4,))
        value.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in potential.parameters()))

        delta = PLMMutationDeltaHead(12, 5, dimension=8, dropout=0.0)
        effect = delta(
            torch.randn(4, 12), torch.randn(4, 5), torch.randn(4, 5),
            torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]),
            torch.rand(4), torch.rand(4),
        )
        self.assertEqual(effect.shape, (4,))
        effect.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in delta.parameters()))

        pair = PLMPairDeltaHead(12, 5, dimension=8, dropout=0.0)
        effect = pair(
            torch.randn(4, 12), torch.randn(4, 12),
            torch.randn(4, 5), torch.randn(4, 5),
            torch.randn(4, 5), torch.randn(4, 5),
            torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4]),
            torch.rand(4), torch.rand(4),
        )
        self.assertEqual(effect.shape, (4,))
        effect.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in pair.parameters()))

    def test_cycle_residual_is_zero_for_commuting_effects(self):
        residual = commuting_cycle_residual(
            torch.tensor([1.0]), torch.tensor([2.0]),
            torch.tensor([0.5]), torch.tensor([2.5]),
        )
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_cycles_use_distinct_positions_within_anchor(self):
        records = {
            "anchor_rank": np.array([0, 0, 0, 1, 1]),
            "position": np.array([3, 3, 4, 1, 2]),
        }
        cycles = commuting_cycles(records)
        self.assertEqual(cycles.tolist(), [[0, 2], [1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
