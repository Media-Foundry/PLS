import json
import tempfile
import unittest
from pathlib import Path

from pls.oracles.fold_editflow import load_shard, shard_status, validate_visible_device


class EditFlowOracleFoldingTests(unittest.TestCase):
    def test_visible_device_contract_supports_local_rocm_and_star_cuda(self):
        self.assertEqual(
            validate_visible_device(0, None, {"HIP_VISIBLE_DEVICES": "0"}),
            ("rocm", 0),
        )
        self.assertEqual(
            validate_visible_device(7, None, {"HIP_VISIBLE_DEVICES": "7"}),
            ("rocm", 7),
        )
        self.assertEqual(
            validate_visible_device(None, 1, {"CUDA_VISIBLE_DEVICES": "1"}),
            ("cuda", 1),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_visible_device(None, None, {})
        with self.assertRaisesRegex(ValueError, "CUDA device mismatch"):
            validate_visible_device(None, 0, {"CUDA_VISIBLE_DEVICES": "1"})

    def test_load_shard_keeps_only_safe_assigned_mutants(self):
        manifest = {
            "test_evaluated": False,
            "nodes": [
                {
                    "node_index": 0,
                    "kind": "single_mutant",
                    "split": "train",
                    "sequence_sha256": "a",
                    "sequence": "AC",
                    "length": 2,
                },
                {
                    "node_index": 1,
                    "kind": "single_mutant",
                    "split": "validation",
                    "sequence_sha256": "b",
                    "sequence": "AD",
                    "length": 2,
                },
            ],
        }
        plan = {
            "test_evaluated": False,
            "shard_count": 2,
            "assignments": [
                {"node_index": 0, "sequence_sha256": "a", "length": 2, "shard": 0},
                {"node_index": 1, "sequence_sha256": "b", "length": 2, "shard": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            plan_path = root / "plan.json"
            manifest_path.write_text(json.dumps(manifest))
            plan_path.write_text(json.dumps(plan))
            records, loaded_plan = load_shard(manifest_path, plan_path, 1)
            status = shard_status(records, root)
        self.assertEqual([row["sequence_sha256"] for row in records], ["b"])
        self.assertEqual(loaded_plan["shard_count"], 2)
        self.assertEqual(status["pending"], 1)
        self.assertFalse(status["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
