import unittest

import numpy as np

from preparation.build_gb1_editflow import column_index
from pls.editflow.hamming import (hamming_distance, hamming_graph_edges,
                                  hash_partition, node_neighbors,
                                  queried_nodes_sha256, variants_from_tokens)
from pls.training.train_editflow_gb1 import closed_local_edges, connected_query_nodes


class GB1EditFlowTests(unittest.TestCase):
    def test_xlsx_column_indices(self):
        self.assertEqual(column_index("A1"), 0)
        self.assertEqual(column_index("Z2"), 25)
        self.assertEqual(column_index("AA19"), 26)
        with self.assertRaisesRegex(ValueError, "invalid"):
            column_index("12")

    def test_complete_hamming_graph_has_expected_edges(self):
        edges = hamming_graph_edges(3, 2)
        self.assertEqual(edges.shape, (2, 18))  # 9*2*(3-1)/2
        self.assertTrue(np.all(edges[0] < edges[1]))
        directed = hamming_graph_edges(3, 2, directed=True)
        self.assertEqual(directed.shape, (2, 36))
        pairs = set(map(tuple, directed.T.tolist()))
        self.assertTrue(all((target, source) in pairs for source, target in pairs))

    def test_value_blind_partition_and_hamming_distance(self):
        tokens = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.uint8)
        variants = variants_from_tokens(tokens, "AC")
        self.assertEqual(variants, ["AA", "AC", "CC"])
        first = hash_partition(variants, salt="fixed")
        second = hash_partition(variants, salt="fixed")
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(hamming_distance(tokens, tokens[0]), [0, 1, 2])

    def test_neighbors_and_connected_query_budget(self):
        neighbors = node_neighbors(0, 3, 2)
        self.assertEqual(set(neighbors.tolist()), {1, 2, 3, 6})
        measured = np.ones(3 ** 2, dtype=bool)
        queried = connected_query_nodes(measured, 0, 5, 17, alphabet_size=3, length=2)
        self.assertEqual(len(set(queried.tolist())), 5)
        edges = closed_local_edges(queried, alphabet_size=3, length=2)
        self.assertGreaterEqual(edges.shape[1], 4)

    def test_graph_and_model_token_spaces_are_distinct(self):
        raw = np.array([[0, 19, 4, 7]], dtype=np.int64)
        model = raw + 1
        self.assertEqual(variants_from_tokens(raw, "ACDEFGHIKLMNPQRSTVWY"), ["AYFI"])
        self.assertTrue(np.all(model >= 1))

    def test_query_manifest_hash_is_order_independent(self):
        self.assertEqual(queried_nodes_sha256([3, 1, 2]), queried_nodes_sha256([2, 3, 1, 1]))
        self.assertEqual(queried_nodes_sha256({1, 2, 3}), queried_nodes_sha256(np.array([3, 2, 1])))


if __name__ == "__main__":
    unittest.main()
