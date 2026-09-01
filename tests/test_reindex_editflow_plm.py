import unittest

from preparation.reindex_editflow_plm_cache import exact_reindex


class ReindexEditFlowPLMTests(unittest.TestCase):
    def test_exact_reindex_preserves_target_order(self):
        source = [
            {"sequence_sha256": "a", "sequence": "AA", "length": "2"},
            {"sequence_sha256": "b", "sequence": "CC", "length": "2"},
        ]
        target = [source[1], source[0]]
        self.assertEqual(exact_reindex(source, target), [1, 0])

    def test_exact_reindex_rejects_metadata_mismatch(self):
        source = [{"sequence_sha256": "a", "sequence": "AA", "length": "2"}]
        target = [{"sequence_sha256": "a", "sequence": "AC", "length": "2"}]
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            exact_reindex(source, target)


if __name__ == "__main__":
    unittest.main()
