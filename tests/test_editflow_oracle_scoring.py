import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pls.models.gvp_structure import GVPStructureFusion
from pls.oracles.score_editflow import (load_safe_nodes,
                                        matched_sequence_only_logits)


class EditFlowOracleScoringTests(unittest.TestCase):
    def test_matched_sequence_ablation_uses_only_same_checkpoint_sequence_branch(self):
        model = GVPStructureFusion(
            sequence_dimension=5,
            input_scalar_dimension=10,
            scalar_dimension=4,
            vector_dimension=2,
            representation_dimension=8,
            dropout=0.0,
            layers=1,
            fusion="aligned_moe",
            residue_sequence_dimension=2,
        ).eval()
        sequence = torch.randn(3, 5)
        expected = model.sequence_head(model.sequence(sequence)).squeeze(-1)
        torch.testing.assert_close(matched_sequence_only_logits(model, sequence), expected)

    def test_safe_node_and_entity_alignment(self):
        manifest = {
            "test_evaluated": False,
            "nodes": [{
                "node_index": 0,
                "split": "validation",
                "sequence_sha256": "digest",
                "sequence": "AC",
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            entities_path = root / "entities.csv"
            manifest_path.write_text(json.dumps(manifest))
            with entities_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["entity_id", "sequence_sha256", "sequence", "length"],
                )
                writer.writeheader()
                writer.writerow({
                    "entity_id": "0",
                    "sequence_sha256": "digest",
                    "sequence": "AC",
                    "length": 2,
                })
            nodes, entities = load_safe_nodes(manifest_path, entities_path)
        self.assertEqual(nodes[0]["sequence_sha256"], entities[0]["sequence_sha256"])


if __name__ == "__main__":
    unittest.main()
