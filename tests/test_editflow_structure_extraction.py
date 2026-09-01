import json
import tempfile
import unittest
from pathlib import Path

from preparation.extract_pls_editflow_structure_v4 import (extraction_status,
                                                            load_safe_records,
                                                            shard_mutant_indices)


class EditFlowStructureExtractionTests(unittest.TestCase):
    def test_shard_plan_selects_only_assigned_mutant_indices(self):
        plan = {
            "test_evaluated": False,
            "shard_count": 2,
            "assignments": [
                {"node_index": 1, "shard": 0},
                {"node_index": 2, "shard": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(shard_mutant_indices(path, 1), {2})
            with self.assertRaisesRegex(ValueError, "outside"):
                shard_mutant_indices(path, 2)

    def test_safe_records_and_dry_status(self):
        manifest = {
            "test_evaluated": False,
            "nodes": [
                {
                    "node_index": 0,
                    "kind": "anchor",
                    "split": "train",
                    "sequence_sha256": "aa",
                    "sequence": "AC",
                },
                {
                    "node_index": 1,
                    "kind": "single_mutant",
                    "split": "validation",
                    "sequence_sha256": "bb",
                    "sequence": "AD",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            records = load_safe_records(path)
            status = extraction_status(records, root / "pdb", root / "parent", root / "out")
        self.assertEqual(status["anchors"], 1)
        self.assertEqual(status["mutants"], 1)
        self.assertEqual(status["mutant_pdb_available"], 0)
        self.assertFalse(status["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
