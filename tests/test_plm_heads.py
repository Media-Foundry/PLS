import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pls.models.plm_heads import PLMDatasetHeads
from pls.training.train_plm_heads import task_loss


class PLMHeadTests(unittest.TestCase):
    def test_all_dataset_specific_heads_train(self):
        model = PLMDatasetHeads(input_dimension=8, hidden_dimension=6, representation_dimension=4, dropout=0)
        features = torch.randn(6, 8)
        tasks = ["uesolds", "pdbsol", "esol", "uesolds", "pdbsol", "esol"]
        targets = torch.tensor([0., 1., .2, 1., 0., .8])
        loss = task_loss(model, features, tasks, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for task in tasks:
            self.assertIsNotNone(model.heads[task].weight.grad)


if __name__ == "__main__":
    unittest.main()
