import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pls.editflow.hamming import queried_nodes_sha256
from pls.training.train_editflow_gb1_replay import load_queried_manifest


class EditFlowReplayTests(unittest.TestCase):
    def write_manifest(self, directory: str, nodes, **overrides) -> Path:
        manifest = {
            "schema": "PLS_EditFlow_queried_nodes_v1",
            "node_indices": list(nodes),
            "sha256": queried_nodes_sha256(nodes),
            "oracle_values_included": False,
        }
        manifest.update(overrides)
        path = Path(directory) / "queried_nodes.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_loads_exact_measured_node_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, [1, 3])
            nodes, manifest = load_queried_manifest(
                path, np.array([False, True, False, True]), 2
            )
        self.assertEqual(nodes.tolist(), [1, 3])
        self.assertEqual(manifest["sha256"], queried_nodes_sha256({1, 3}))

    def test_rejects_oracle_values_and_imputed_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(
                directory, [0, 1], oracle_values_included=True
            )
            with self.assertRaisesRegex(ValueError, "must not embed"):
                load_queried_manifest(path, np.ones(2, dtype=bool), 2)
            path = self.write_manifest(directory, [0, 1])
            with self.assertRaisesRegex(ValueError, "imputed"):
                load_queried_manifest(path, np.array([True, False]), 2)

    def test_rejects_duplicates_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(directory, [0, 0])
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_queried_manifest(path, np.ones(2, dtype=bool), 2)
            path = self.write_manifest(directory, [0, 1], sha256="wrong")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_queried_manifest(path, np.ones(2, dtype=bool), 2)


if __name__ == "__main__":
    unittest.main()
