import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pls.features.structure_v4_schema import adapt_v4_features
from pls.models.structure_descriptor import StructureDescriptorEncoder


class StructureV4Tests(unittest.TestCase):
    def test_adapter_and_rotation_invariance(self):
        residues = 7
        features = {"physchem_features": torch.randn(residues, 62),
                    "spatial_scalar_features": torch.randn(residues, 89),
                    "spatial_vector_features": torch.randn(residues, 8, 3),
                    "ca_coords": torch.randn(residues, 3), "n_residues": residues}
        adapted = adapt_v4_features(features, torch.linspace(50, 90, residues))
        self.assertEqual(adapted["spatial_scalar_features"].shape, (residues, 90))
        encoder = StructureDescriptorEncoder(hidden_dimension=16, output_dimension=8, dropout=0).eval()
        vectors = adapted["spatial_vector_features"].unsqueeze(0)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated = torch.einsum("bnvc,cd->bnvd", vectors, rotation)
        first, _ = encoder(adapted["physchem_features"].unsqueeze(0),
                           adapted["spatial_scalar_features"].unsqueeze(0), vectors)
        second, _ = encoder(adapted["physchem_features"].unsqueeze(0),
                            adapted["spatial_scalar_features"].unsqueeze(0), rotated)
        self.assertTrue(torch.allclose(first, second, atol=1e-5))


if __name__ == "__main__": unittest.main()
