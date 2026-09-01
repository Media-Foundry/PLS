import unittest

from preparation.materialize_fixed_parent_structure import build_mapping


class FixedParentStructureTests(unittest.TestCase):
    def test_mapping_reuses_anchor_without_crossing_split_or_component(self):
        anchor = {
            "node_index": 0,
            "anchor_rank": 0,
            "kind": "anchor",
            "split": "train",
            "component_root_sha256": "component",
            "sequence_sha256": "parent",
            "length": 4,
        }
        mutant = {
            **anchor,
            "node_index": 1,
            "kind": "single_mutant",
            "sequence_sha256": "mutant",
        }
        mapping = build_mapping({
            "nodes": [anchor, mutant],
            "test_evaluated": False,
        })
        self.assertEqual(mapping[1][0]["sequence_sha256"], "mutant")
        self.assertEqual(mapping[1][1]["sequence_sha256"], "parent")

    def test_mapping_rejects_test_manifest(self):
        with self.assertRaisesRegex(ValueError, "test-free"):
            build_mapping({"nodes": [], "test_evaluated": True})


if __name__ == "__main__":
    unittest.main()
