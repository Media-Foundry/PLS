"""Utilities for component-safe multi-fidelity edit-field correction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .metrics import mutation_field_metrics


def delta_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, float | int]:
    """Evaluate directly represented edit effects with existing field metrics."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    groups = np.asarray(groups)
    if truth.ndim != 1 or prediction.shape != truth.shape or groups.shape != truth.shape:
        raise ValueError("truth, prediction, and groups must be equal one-dimensional arrays")
    size = len(truth)
    teacher = np.concatenate([np.zeros(size), truth])
    student = np.concatenate([np.zeros(size), prediction])
    edges = np.stack([np.arange(size), np.arange(size) + size])
    return mutation_field_metrics(teacher, student, edges, groups, top_k=top_k)


def grouped_oof_predictions(
    estimator,
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
) -> np.ndarray:
    """Return out-of-fold predictions with entire groups held out."""
    features = np.asarray(features)
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups)
    if features.ndim != 2 or len(features) != len(target) or groups.shape != target.shape:
        raise ValueError("invalid grouped regression arrays")
    if len(np.unique(groups)) < folds:
        raise ValueError("fewer groups than requested folds")
    result = np.full(len(target), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train, held_out in splitter.split(features, target, groups):
        model = clone(estimator)
        model.fit(features[train], target[train])
        result[held_out] = model.predict(features[held_out])
    if not np.isfinite(result).all():
        raise RuntimeError("grouped OOF prediction left missing values")
    return result


def select_by_grouped_rmse(
    candidates: dict[str, object],
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    compose: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[str, object, np.ndarray, dict[str, float]]:
    """Select a prespecified estimator using grouped OOF RMSE only."""
    records: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, estimator in candidates.items():
        raw = grouped_oof_predictions(estimator, features, target, groups, folds=folds)
        prediction = compose(raw) if compose is not None else raw
        predictions[name] = prediction
        records[name] = float(np.sqrt(np.mean(np.square(prediction - target))))
    selected = min(records, key=lambda name: (records[name], name))
    model = clone(candidates[selected]).fit(features, target)
    return selected, model, predictions[selected], records


def selective_hybrid(
    approximate: np.ndarray,
    exact: np.ndarray,
    priority: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the highest-priority approximate edges with exact queries."""
    approximate = np.asarray(approximate, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    priority = np.asarray(priority, dtype=np.float64)
    if approximate.shape != exact.shape or priority.shape != exact.shape or exact.ndim != 1:
        raise ValueError("selective-refolding arrays must have equal one-dimensional shape")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    count = int(round(fraction * len(exact)))
    selected = np.argsort(-priority, kind="stable")[:count]
    hybrid = approximate.copy()
    hybrid[selected] = exact[selected]
    return hybrid, selected


def correlation_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return {
        "pearson": float(pearsonr(truth, prediction).statistic),
        "spearman": float(spearmanr(truth, prediction).statistic),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - truth)))),
        "mae": float(np.mean(np.abs(prediction - truth))),
    }
