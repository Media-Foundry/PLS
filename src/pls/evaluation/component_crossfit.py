"""SI-component-aware folds for validation-only calibration diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def si_component_groups(
    entity_indices,
    entities_path: Path,
    observation_split_path: Path,
    *,
    required_split: str = "validation",
) -> np.ndarray:
    """Map prediction entity indices to immutable SI component root hashes."""
    indices = np.asarray(entity_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("entity_indices must be a nonempty one-dimensional array")
    with Path(entities_path).open(newline="") as handle:
        entities = list(csv.DictReader(handle))
    if indices.min() < 0 or indices.max() >= len(entities):
        raise ValueError("entity index is outside the entity manifest")
    selected_hashes = {entities[int(index)]["sequence_sha256"] for index in indices}
    component_by_hash: dict[str, str] = {}
    with Path(observation_split_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence_hash = row["sequence_sha256"]
            if sequence_hash not in selected_hashes or row["split"] != required_split:
                continue
            component = row["component_root_sha256"]
            previous = component_by_hash.setdefault(sequence_hash, component)
            if previous != component:
                raise ValueError("one sequence maps to multiple SI components")
    missing = sorted(selected_hashes - component_by_hash.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} prediction entities lack a {required_split} SI component"
        )
    return np.asarray(
        [component_by_hash[entities[int(index)]["sequence_sha256"]] for index in indices]
    )


def _validate_folds(folds, groups: np.ndarray, size: int):
    seen = np.zeros(size, dtype=np.int64)
    validated = []
    for train_indices, heldout_indices in folds:
        train_indices = np.asarray(train_indices, dtype=np.int64)
        heldout_indices = np.asarray(heldout_indices, dtype=np.int64)
        if set(groups[train_indices]) & set(groups[heldout_indices]):
            raise RuntimeError("SI component leaked across a calibration fold")
        seen[heldout_indices] += 1
        validated.append((train_indices, heldout_indices))
    if not np.all(seen == 1):
        raise RuntimeError("cross-fit folds must hold out every observation exactly once")
    return validated


def stratified_component_folds(targets, groups, folds: int, seed: int):
    """Binary stratified cross-fitting with SI components as indivisible groups."""
    targets = np.asarray(targets)
    groups = np.asarray(groups)
    if targets.ndim != 1 or targets.shape != groups.shape:
        raise ValueError("targets and groups must be equal one-dimensional arrays")
    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed)
    )
    return _validate_folds(
        splitter.split(np.zeros(len(targets)), targets, groups), groups, len(targets)
    )


def regression_component_folds(targets, groups, folds: int):
    """Regression cross-fitting with SI components as indivisible groups."""
    targets = np.asarray(targets)
    groups = np.asarray(groups)
    if targets.ndim != 1 or targets.shape != groups.shape:
        raise ValueError("targets and groups must be equal one-dimensional arrays")
    splitter = GroupKFold(n_splits=int(folds))
    return _validate_folds(
        splitter.split(np.zeros(len(targets)), targets, groups), groups, len(targets)
    )
