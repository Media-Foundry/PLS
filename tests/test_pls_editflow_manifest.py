import unittest

from preparation.build_pls_editflow_manifest import (mutation_candidates,
                                                       select_anchors,
                                                       sequence_sha256,
                                                       validate_manifest)


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


if __name__ == "__main__":
    unittest.main()
