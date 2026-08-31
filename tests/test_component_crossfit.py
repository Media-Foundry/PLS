import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pls.evaluation.component_crossfit import (
    regression_component_folds,
    si_component_groups,
    stratified_component_folds,
)


class ComponentCrossfitTests(unittest.TestCase):
    def test_entity_indices_map_to_validation_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "entities.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("sequence_sha256",))
                writer.writeheader()
                writer.writerows([
                    {"sequence_sha256": "a"},
                    {"sequence_sha256": "b"},
                    {"sequence_sha256": "c"},
                ])
            fields = ("sequence_sha256", "component_root_sha256", "split")
            with (root / "split.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {"sequence_sha256": "a", "component_root_sha256": "g1", "split": "validation"},
                    {"sequence_sha256": "b", "component_root_sha256": "g1", "split": "validation"},
                    {"sequence_sha256": "c", "component_root_sha256": "g2", "split": "validation"},
                ])
            groups = si_component_groups(
                np.array([2, 0, 1]), root / "entities.csv", root / "split.csv"
            )
        self.assertEqual(groups.tolist(), ["g2", "g1", "g1"])

    def test_binary_and_regression_folds_never_split_components(self):
        groups = np.repeat(np.asarray([f"g{i}" for i in range(10)]), 2)
        targets = np.tile(np.asarray([0, 1]), 10)
        for split in (
            stratified_component_folds(targets, groups, 5, 17),
            regression_component_folds(targets.astype(float), groups, 5),
        ):
            heldout = np.zeros(len(groups), dtype=int)
            for train, validation in split:
                self.assertFalse(set(groups[train]) & set(groups[validation]))
                heldout[validation] += 1
            np.testing.assert_array_equal(heldout, np.ones(len(groups), dtype=int))


if __name__ == "__main__":
    unittest.main()
