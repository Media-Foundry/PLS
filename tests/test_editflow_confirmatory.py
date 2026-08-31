import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pls.editflow.hamming import queried_nodes_sha256, variants_from_tokens
from pls.training.train_editflow_gb1_confirmatory import (
    aggregate_anchor_results,
    load_anchor_protocol,
)


class EditFlowConfirmatoryTests(unittest.TestCase):
    def test_anchor_protocol_validates_identity_and_fitness_blindness(self):
        tokens = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
        variants = variants_from_tokens(tokens, "ACDEFGHIKLMNPQRSTVWY")
        protocol = {
            "schema": "PLS_EditFlow_GB1_anchor_protocol_v1",
            "selection": {"count": 2, "fitness_accessed": False},
            "anchors_sha256": queried_nodes_sha256([0, 2]),
            "anchors": [
                {"rank": 0, "node_index": 0, "variant": variants[0]},
                {"rank": 1, "node_index": 2, "variant": variants[2]},
            ],
            "test_evaluated": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol))
            anchors, loaded = load_anchor_protocol(
                path, tokens, np.ones(3, dtype=bool)
            )
        self.assertEqual(len(anchors), 2)
        self.assertFalse(loaded["selection"]["fitness_accessed"])

    def test_aggregate_keeps_anchor_level_regret(self):
        def result(r2, regret):
            return {
                "stages": [{
                    "closed_edges": 10,
                    "value": {"r2": r2},
                    "edge": {
                        "edge_spearman": r2 / 2,
                        "anchor_macro_kendall_tau": r2 / 3,
                    },
                    "regret": {"1": {"regret": regret}},
                }]
            }

        aggregate = aggregate_anchor_results(
            [result(0.2, 0.0), result(0.6, 2.0)], [80], [1]
        )
        self.assertAlmostEqual(aggregate[0]["value_r2"]["mean"], 0.4)
        self.assertAlmostEqual(aggregate[0]["regret"]["1"]["mean"], 1.0)
        self.assertAlmostEqual(
            aggregate[0]["regret"]["1"]["zero_regret_fraction"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
