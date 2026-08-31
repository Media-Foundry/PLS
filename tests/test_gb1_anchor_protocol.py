import unittest
from pathlib import Path

from preparation.build_gb1_anchor_protocol import build_protocol


class GB1AnchorProtocolTests(unittest.TestCase):
    def test_protocol_does_not_claim_fitness_access(self):
        landscape = Path("benchmark/generated/gb1_wu2016_editflow_v1.npz")
        if not landscape.exists():
            self.skipTest("generated GB1 landscape is unavailable")
        protocol = build_protocol(landscape, 3, "unit-test", ("VDGV",))
        self.assertFalse(protocol["selection"]["fitness_accessed"])
        self.assertEqual(len(protocol["anchors"]), 3)
        self.assertNotIn("VDGV", [row["variant"] for row in protocol["anchors"]])


if __name__ == "__main__":
    unittest.main()
