import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "preparation"))
from si_engine import ScoringConfig, parse_shard_indices, sequence_identity


class SequenceIdentityTests(unittest.TestCase):
    def test_identity_and_gap_denominator(self):
        self.assertEqual(sequence_identity("ACGT", "ACGT"), 1.0)
        self.assertAlmostEqual(sequence_identity("ACGT", "AGT"), 0.75)

    def test_empty_sequences(self):
        self.assertEqual(sequence_identity("", ""), 1.0)
        self.assertEqual(sequence_identity("A", ""), 0.0)

    def test_logical_shard_parser(self):
        self.assertEqual(parse_shard_indices("0-2,5", 8), {0, 1, 2, 5})
        self.assertEqual(parse_shard_indices("all", 3), {0, 1, 2})
        with self.assertRaises(ValueError):
            parse_shard_indices("3", 3)

    def test_resumable_blocks_match_straightforward(self):
        sequences = ["AAAA", "AAAT", "GGGG", "A"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); entities = root / "entities.csv"; output = root / "run"
            import hashlib
            rows = sorted((hashlib.sha256(s.encode()).hexdigest(), s) for s in sequences)
            with entities.open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(["entity_id", "sequence_sha256", "sequence", "length"])
                for i, (digest, sequence) in enumerate(rows): writer.writerow([f"entity:{i:06d}", digest, sequence, len(sequence)])
            command = [sys.executable, str(Path(__file__).parents[1] / "preparation" / "si_engine.py"),
                       "--entities", str(entities), "--output-dir", str(output), "--block-size", "2", "--workers", "2"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            blocks = sorted((output / "blocks").glob("*.npz"))
            mtimes = {path: path.stat().st_mtime_ns for path in blocks}
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in blocks})
            self.assertEqual(json.loads((output / "run_config.json").read_text())["scoring"], ScoringConfig().__dict__)
            observed = []
            for path in blocks:
                with np.load(path) as block: observed.extend(block["similarity"][np.isfinite(block["similarity"])])
            expected = [sequence_identity(a, b) for i, a in enumerate([s for _, s in rows]) for b in [s for _, s in rows][i + 1:]]
            self.assertEqual(sorted(observed), sorted(expected))


if __name__ == "__main__":
    unittest.main()
