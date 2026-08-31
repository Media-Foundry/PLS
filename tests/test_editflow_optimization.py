import unittest

import numpy as np

from pls.editflow.optimization import (beam_search_paths,
                                       bound_aware_frontier_acquisition,
                                       hybrid_query_budget,
                                       path_aware_frontier_acquisition)
from pls.training.train_editflow_gb1_active import (
    frontier_policy_acquisition,
    uncertainty_acquisition,
)


class EditFlowOptimizationTests(unittest.TestCase):
    def test_beam_search_follows_high_potential_edit_path(self):
        # 2^3 graph: 000 -> 100 -> 110 -> 111 is assigned the high path.
        values = np.zeros(8);values[[4, 6, 7]] = [1, 2, 3]
        paths = beam_search_paths(values, 0, np.ones(8, bool), alphabet_size=2, length=3, steps=3, beam_width=1)
        self.assertEqual(paths[-1], (0, 4, 6, 7))

    def test_path_aware_acquisition_targets_uncertain_visited_frontier(self):
        ensemble = np.array([
            [0, 0, 0, 0, 1.0, 0, 2.0, 3.0],
            [0, 0, 0, 0, 2.0, 0, 2.1, 3.1],
            [0, 0, 0, 0, 0.5, 0, 1.9, 2.9],
        ])
        result = path_aware_frontier_acquisition(
            ensemble, {0}, np.ones(8, bool), 0, 1,
            alphabet_size=2, length=3, steps=3, beam_width=1,
        )
        self.assertEqual(result.batch.node_indices.tolist(), [4])
        self.assertGreater(result.batch.scores[0], 0)
        self.assertGreaterEqual(len(result.paths), 3)

    def test_uncertainty_fill_does_not_expand_through_unpurchased_node(self):
        # In the 20^4 GB1 graph, node 8400 is adjacent to prospective node
        # 8000 but not to queried node 0. Excluding 8000 must not make 8400 a
        # frontier candidate.
        node_count = 20**4
        ensemble = np.array([
            np.zeros(node_count, dtype=float),
            np.arange(node_count, dtype=float),
        ])
        acquired, edges = uncertainty_acquisition(
            ensemble,
            {0},
            np.ones(node_count, dtype=bool),
            3,
            excluded_targets={8000},
        )
        self.assertNotIn(8400, edges[1].tolist())
        self.assertNotIn(8000, acquired.node_indices.tolist())

    def test_bound_aware_acquisition_targets_shortest_uncertainty_routes(self):
        ensemble = np.array([
            [0.0, 0.1, 0.0, 0.2, 1.0, 0.0, 2.0, 3.0],
            [0.0, 0.1, 0.0, 0.2, 2.0, 0.0, 2.1, 3.1],
            [0.0, 0.1, 0.0, 0.2, 0.5, 0.0, 1.9, 2.9],
        ])
        result = bound_aware_frontier_acquisition(
            ensemble,
            {0},
            np.ones(8, dtype=bool),
            0,
            1,
            alphabet_size=2,
            length=3,
            steps=3,
            beam_width=2,
        )
        self.assertGreaterEqual(len(result.candidate_endpoints), 1)
        self.assertEqual(len(result.selected_paths), len(result.candidate_endpoints))
        self.assertTrue(np.all(result.estimated_path_bounds >= 0))
        self.assertEqual(result.batch.node_indices.tolist(), [4])

    def test_hybrid_budget_preserves_targeted_and_exploration_queries(self):
        self.assertEqual(hybrid_query_budget(80, 0.5), 40)
        self.assertEqual(hybrid_query_budget(3, 0.5), 2)
        self.assertEqual(hybrid_query_budget(1, 0.5), 1)
        with self.assertRaises(ValueError):
            hybrid_query_budget(80, 1.0)

    def test_standard_frontier_baselines_share_candidates_and_budget(self):
        ensemble = np.zeros((3, 20**4), dtype=float)
        ensemble[:, 8000] = [3.0, 4.0, 2.0]
        measured = np.ones(20**4, dtype=bool)
        outputs = {}
        for policy in ("random", "greedy", "ucb", "thompson"):
            batch, edges = frontier_policy_acquisition(
                ensemble,
                {0},
                measured,
                2,
                policy,
                np.random.default_rng(17),
                beta=0.5,
            )
            self.assertEqual(len(batch.node_indices), 2)
            self.assertEqual(len(set(batch.node_indices.tolist())), 2)
            self.assertTrue(set(batch.node_indices.tolist()).isdisjoint({0}))
            outputs[policy] = (batch, edges)
        # Greedy target mean is maximal at the deliberately raised neighbor.
        self.assertEqual(outputs["greedy"][0].node_indices[0], 8000)
        # Every baseline operated over the exact same one-hop frontier.
        candidate_edges = {output[1].shape[1] for output in outputs.values()}
        self.assertEqual(len(candidate_edges), 1)

    def test_occupancy_only_does_not_require_ensemble_disagreement(self):
        ensemble = np.zeros((3, 8), dtype=float)
        result = path_aware_frontier_acquisition(
            ensemble,
            {0},
            np.ones(8, dtype=bool),
            0,
            1,
            alphabet_size=2,
            length=3,
            steps=2,
            beam_width=2,
            score_mode="occupancy_only",
        )
        self.assertEqual(len(result.batch.node_indices), 1)
        self.assertGreater(result.batch.scores[0], 0.0)


if __name__ == "__main__":
    unittest.main()
