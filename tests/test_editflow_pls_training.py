import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pls.training.train_editflow_pls import load_landscape, validation_metrics


class EditFlowPLSTrainingTests(unittest.TestCase):
    def test_load_and_validation_metrics_use_only_safe_groups(self):
        nodes = [
            {"node_index": 0, "anchor_rank": 0, "kind": "anchor", "split": "train", "sequence_sha256": "a", "sequence": "AC"},
            {"node_index": 1, "anchor_rank": 0, "kind": "single_mutant", "split": "train", "sequence_sha256": "b", "sequence": "AD"},
            {"node_index": 2, "anchor_rank": 1, "kind": "anchor", "split": "validation", "sequence_sha256": "c", "sequence": "EC"},
            {"node_index": 3, "anchor_rank": 1, "kind": "single_mutant", "split": "validation", "sequence_sha256": "d", "sequence": "ED"},
            {"node_index": 4, "anchor_rank": 1, "kind": "single_mutant", "split": "validation", "sequence_sha256": "e", "sequence": "EE"},
            {"node_index": 5, "anchor_rank": 2, "kind": "anchor", "split": "validation", "sequence_sha256": "f", "sequence": "FC"},
            {"node_index": 6, "anchor_rank": 2, "kind": "single_mutant", "split": "validation", "sequence_sha256": "g", "sequence": "FD"},
            {"node_index": 7, "anchor_rank": 2, "kind": "single_mutant", "split": "validation", "sequence_sha256": "h", "sequence": "FE"},
        ]
        manifest = {
            "test_evaluated": False,
            "nodes": nodes,
            "edges": [
                {"source_node": 0, "target_node": 1},
                {"source_node": 2, "target_node": 3},
                {"source_node": 2, "target_node": 4},
                {"source_node": 5, "target_node": 6},
                {"source_node": 5, "target_node": 7},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            scores_path = root / "scores.npz"
            report_path = root / "report.json"
            manifest_path.write_text(json.dumps(manifest))
            report_path.write_text(json.dumps({"test_evaluated": False, "output": "raw_logit"}))
            np.savez(
                scores_path,
                node_indices=np.arange(8),
                sequence_sha256=np.asarray(["a", "b", "c", "d", "e", "f", "g", "h"]),
                logits=np.asarray([0.0, 1.0, 2.0, 4.0, 1.0, 0.0, 3.0, -1.0]),
            )
            landscape = load_landscape(manifest_path, scores_path, report_path)
        self.assertEqual(landscape["groups"]["train"], [0])
        self.assertEqual(landscape["groups"]["validation"], [1, 2])
        prediction = np.asarray([np.nan, np.nan, 2.0, 4.0, 1.0, 0.0, 3.0, -1.0])
        metrics = validation_metrics(landscape, prediction, 1)
        self.assertEqual(metrics["edge"]["edges"], 4)
        self.assertAlmostEqual(metrics["value"]["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
