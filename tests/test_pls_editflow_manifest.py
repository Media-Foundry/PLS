import unittest

from preparation.build_pls_editflow_manifest import (mutation_candidates,
                                                       select_anchors,
                                                       sequence_sha256,
                                                       validate_manifest,
                                                       write_entities)
from preparation.plan_pls_editflow_oracle import lpt_shards


class PLSEditFlowManifestTests(unittest.TestCase):
    def test_anchor_selection_is_deterministic_and_split_stratified(self):
        candidates = {
            "train": [
                {"sequence_sha256": sequence_sha256(value), "value": value}
                for value in ("AAAA", "CCCC", "DDDD")
            ],
            "validation": [
                {"sequence_sha256": sequence_sha256(value), "value": value}
                for value in ("EEEE", "FFFF")
            ],
        }
        first = select_anchors(
            candidates, {"train": 2, "validation": 1}, salt="fixed"
        )
        second = select_anchors(
            candidates, {"train": 2, "validation": 1}, salt="fixed"
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_mutation_candidates_are_canonical_and_deterministic(self):
        first = mutation_candidates("AC", "ACDEFGHIKLMNPQRSTVWY", salt="fixed")
        second = mutation_candidates("AC", "ACDEFGHIKLMNPQRSTVWY", salt="fixed")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 38)
        for row in first:
            self.assertNotEqual(row["source_residue"], row["target_residue"])

    def test_component_unique_selection_prioritizes_cached_components(self):
        candidates = {
            "train": [
                {
                    "sequence_sha256": sequence_sha256(value),
                    "component_root_sha256": component,
                    "value": value,
                }
                for value, component in (
                    ("AAAA", "g1"),
                    ("CCCC", "g1"),
                    ("DDDD", "g2"),
                    ("EEEE", "g3"),
                )
            ],
            "validation": [],
        }
        chosen = select_anchors(
            candidates,
            {"train": 3, "validation": 0},
            salt="fixed",
            unique_components=True,
            priority_hashes_by_split={
                "train": [sequence_sha256("CCCC"), sequence_sha256("AAAA")]
            },
        )
        self.assertEqual(chosen[0]["value"], "CCCC")
        self.assertEqual(len({row["component_root_sha256"] for row in chosen}), 3)

    def test_validator_rejects_forbidden_split(self):
        sequence = "AC"
        manifest = {
            "test_evaluated": False,
            "forbidden_sequences_loaded": False,
            "nodes": [{
                "node_index": 0,
                "anchor_rank": 0,
                "kind": "anchor",
                "split": "forbidden",
                "component_root_sha256": "component",
                "sequence_sha256": sequence_sha256(sequence),
                "sequence": sequence,
                "length": 2,
            }],
            "edges": [],
        }
        report = {
            "nodes_sha256": sequence_sha256(sequence_sha256(sequence)),
            "edges_sha256": sequence_sha256(""),
            "test_sequences_queried": 0,
            "test_evaluated": False,
        }
        with self.assertRaisesRegex(ValueError, "forbidden split"):
            validate_manifest(manifest, report)

    def test_lpt_shards_are_deterministic_and_balanced(self):
        nodes = [
            {"node_index": index, "sequence_sha256": str(index), "length": length}
            for index, length in enumerate((10, 9, 8, 7))
        ]
        first = lpt_shards(nodes, 2)
        second = lpt_shards(nodes, 2)
        self.assertEqual(first, second)
        loads = [
            sum(row["estimated_cost_l2"] for row in first if row["shard"] == shard)
            for shard in range(2)
        ]
        self.assertLess(max(loads) / min(loads), 1.5)


if __name__ == "__main__":
    unittest.main()
