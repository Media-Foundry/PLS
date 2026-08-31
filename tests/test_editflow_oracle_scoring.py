import csv
import json
import tempfile
import unittest
from pathlib import Path

from pls.oracles.score_editflow import load_safe_nodes


class EditFlowOracleScoringTests(unittest.TestCase):
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
