import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pls.models.plm_heads import PLMDatasetHeads, TASKS
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

    def test_task_adapter_variant(self):
        model = PLMDatasetHeads(input_dimension=8, hidden_dimension=6,
                                representation_dimension=4, dropout=0, task_adapters=True)
        output = model(torch.randn(3, 8), "esol")
        self.assertEqual(output.shape, (3,))
        output.sum().backward()
        self.assertIsNotNone(model.adapters["esol"][1].weight.grad)

    def test_latent_endpoint_mappings_are_monotone(self):
        model = PLMDatasetHeads(input_dimension=8, hidden_dimension=12,
                                representation_dimension=6, dropout=0, latent_endpoint=True)
        latent = torch.tensor([-2., 0., 2.])
        for task in TASKS:
            observed = model.observe_latent(latent, task)
            self.assertTrue(torch.all(observed[1:] > observed[:-1]))
        esol = model.observe_latent(latent, "esol")
        self.assertTrue(torch.all((esol > 0) & (esol < 1)))
        with self.assertRaisesRegex(ValueError, "task-independent"):
            PLMDatasetHeads(input_dimension=8, latent_endpoint=True, task_adapters=True)


if __name__ == "__main__":
    unittest.main()
