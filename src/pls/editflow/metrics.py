"""Optimization-facing metrics for discrete mutation landscapes."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def _edge_effects(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return values[edges[1]] - values[edges[0]]


def mutation_field_metrics(
    teacher_values,
    student_values,
    edge_index,
    edge_groups=None,
    *,
    top_k: int = 10,
    tie_tolerance: float = 1e-8,
) -> dict[str, float | int]:
    """Evaluate edit effects with edge-level and anchor-macro statistics."""
    teacher = np.asarray(teacher_values, dtype=np.float64)
    student = np.asarray(student_values, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if teacher.ndim != 1 or teacher.shape != student.shape:
        raise ValueError("teacher_values and student_values must be equal 1D arrays")
    if edges.ndim != 2 or edges.shape[0] != 2 or not edges.shape[1]:
        raise ValueError("edge_index must have shape [2, nonzero edges]")
    if edges.min() < 0 or edges.max() >= len(teacher):
        raise ValueError("edge_index references a missing node")
    if top_k <= 0 or tie_tolerance < 0:
        raise ValueError("top_k must be positive and tie_tolerance nonnegative")
    groups = np.zeros(edges.shape[1], dtype=np.int64) if edge_groups is None else np.asarray(edge_groups)
    if groups.shape != (edges.shape[1],):
        raise ValueError("edge_groups must provide one group per edge")
    target = _edge_effects(teacher, edges)
    prediction = _edge_effects(student, edges)
    difference = prediction - target
    informative = np.abs(target) > tie_tolerance
    sign_accuracy = float(np.mean((prediction[informative] > 0) == (target[informative] > 0))) if informative.any() else float("nan")
    recalls = []
    truth_sign = target[informative] > 0
    predicted_sign = prediction[informative] > 0
    for label in (False, True):
        selected = truth_sign == label
        if selected.any():
            recalls.append(float(np.mean(predicted_sign[selected] == label)))
    balanced_sign_accuracy = float(np.mean(recalls)) if len(recalls) == 2 else float("nan")
    taus, recalls_at_k = [], []
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        if len(selected) >= 2 and np.ptp(target[selected]) > 0 and np.ptp(prediction[selected]) > 0:
            value = float(kendalltau(target[selected], prediction[selected]).statistic)
            if np.isfinite(value):
                taus.append(value)
        count = min(top_k, len(selected))
        teacher_top = set(selected[np.argsort(-target[selected], kind="stable")[:count]].tolist())
        student_top = set(selected[np.argsort(-prediction[selected], kind="stable")[:count]].tolist())
        recalls_at_k.append(len(teacher_top & student_top) / count)
    return {
        "nodes": int(len(teacher)), "edges": int(edges.shape[1]),
        "edge_mae": float(np.mean(np.abs(difference))),
        "edge_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "edge_pearson": float(pearsonr(target, prediction).statistic),
        "edge_spearman": float(spearmanr(target, prediction).statistic),
        "mutation_sign_accuracy": sign_accuracy,
        "mutation_sign_balanced_accuracy": balanced_sign_accuracy,
        "anchor_macro_kendall_tau": float(np.mean(taus)) if taus else float("nan"),
        f"anchor_macro_top_{top_k}_recall": float(np.mean(recalls_at_k)),
        "informative_sign_edges": int(informative.sum()),
    }
