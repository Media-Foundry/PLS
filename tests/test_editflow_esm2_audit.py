import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preparation.audit_pls_editflow_esm2 import audit


class EditFlowESM2AuditTests(unittest.TestCase):
    def test_audit_accepts_complete_finite_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entities = root / "entities.csv"
            with entities.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["entity_id", "sequence_sha256", "sequence", "length"])
                writer.writeheader()
                writer.writerow({"entity_id": "0", "sequence_sha256": "a", "sequence": "AC", "length": 2})
            features = root / "features"
            features.mkdir()
            (features / "config.json").write_text(json.dumps({"embedding_dimension": 3}))
            np.save(features / "embeddings.npy", np.ones((1, 3), dtype=np.float32))
            np.save(features / "status.npy", np.ones(1, dtype=np.uint8))
            report = audit(entities, features)
        self.assertEqual(report["complete"], 1)
        self.assertEqual(report["shape"], [1, 3])
        self.assertFalse(report["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
