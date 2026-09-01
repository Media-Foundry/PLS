import unittest

from preparation.build_pls_editflow_manifest import sequence_sha256
from preparation.subset_pls_editflow_manifest import build_subset


class PLSEditFlowSubsetTests(unittest.TestCase):
    def test_split_specific_subset_is_label_blind_and_reindexed(self):
        nodes = []
        edges = []
        for anchor_rank, split in enumerate(("train", "validation")):
            parent_sequence = "AAAA" if split == "train" else "DDDD"
            source = len(nodes)
            nodes.append({
                "node_index": source,
                "anchor_rank": anchor_rank,
                "kind": "anchor",
                "split": split,
                "component_root_sha256": f"g{anchor_rank}",
                "sequence_sha256": sequence_sha256(parent_sequence),
                "sequence": parent_sequence,
                "length": 4,
            })
            for position in range(2):
                target = len(nodes)
                sequence = (
                    parent_sequence[:position]
                    + "C"
                    + parent_sequence[position + 1:]
                )
                nodes.append({
                    "node_index": target,
                    "anchor_rank": anchor_rank,
                    "kind": "single_mutant",
                    "split": split,
                    "component_root_sha256": f"g{anchor_rank}",
                    "sequence_sha256": sequence_sha256(sequence),
                    "sequence": sequence,
                    "length": 4,
                    "mutation": {
                        "position_zero_based": position,
                        "position_one_based": position + 1,
                        "source_residue": parent_sequence[position],
                        "target_residue": "C",
                    },
                })
                edges.append({
                    "edge_index": len(edges),
                    "anchor_rank": anchor_rank,
                    "source_node": source,
                    "target_node": target,
                })
        source = {
            "schema": "source",
            "selection": {"component_unique_anchors": True},
            "nodes": nodes,
            "edges": edges,
            "test_evaluated": False,
            "forbidden_sequences_loaded": False,
        }
        subset, report = build_subset(source, {"train": 1, "validation": 2})
        self.assertEqual(report["single_mutation_edges"], 3)
        self.assertEqual(report["unique_sequence_queries"], 5)
        self.assertEqual([row["node_index"] for row in subset["nodes"]], list(range(5)))
        self.assertFalse(report["selection_used_target_values"])

    def test_rejects_non_test_free_source(self):
        with self.assertRaisesRegex(ValueError, "test-free"):
            build_subset({"test_evaluated": True}, {"train": 1, "validation": 1})


if __name__ == "__main__":
    unittest.main()
