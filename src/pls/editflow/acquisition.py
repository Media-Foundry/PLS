"""Path-aware acquisition under a unique-node teacher-query budget."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AcquisitionBatch:
    node_indices: np.ndarray
    scores: np.ndarray
    candidate_edges: int


@dataclass(frozen=True)
class CostAwareAcquisitionBatch:
    node_indices: np.ndarray
    scores: np.ndarray
    node_costs: np.ndarray
    total_cost: float
    candidate_edges: int


@dataclass(frozen=True)
class PathConcentration:
    positive_support: int
    total_mass: float
    entropy: float
    normalized_entropy: float
    effective_support: float
    maximum_share: float


@dataclass(frozen=True)
class EdgeErrorEnvelope:
    values: np.ndarray
    additive_quantile: float
    nominal_coverage: float
    empirical_calibration_coverage: float
    calibration_count: int


def path_concentration(occupancy) -> PathConcentration:
    """Describe concentration after normalizing nonnegative edge occupancy.

    Raw occupancy sums to the mean traversed path length, not one.  Entropy and
    effective support are therefore computed after normalization over edges.
    Empty occupancy has zero support; one positive edge has normalized entropy
    zero and effective support one.
    """
    values = np.asarray(occupancy, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("occupancy must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("occupancy must be finite and nonnegative")
    positive = values[values > 0]
    if not len(positive):
        return PathConcentration(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    probabilities = positive / positive.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = (
        entropy / float(np.log(len(probabilities))) if len(probabilities) > 1 else 0.0
    )
    return PathConcentration(
        positive_support=int(len(probabilities)),
        total_mass=float(positive.sum()),
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        effective_support=float(1.0 / np.sum(probabilities ** 2)),
        maximum_share=float(probabilities.max()),
    )


def conformal_edge_error_envelope(
    calibration_uncertainty,
    calibration_absolute_error,
    target_uncertainty,
    *,
    alpha: float = 0.1,
) -> EdgeErrorEnvelope:
    """Additively calibrate uncertainty using out-of-sample edge errors.

    The caller must supply held-out or cross-fitted absolute edit-field errors;
    in-sample residuals do not justify a conformal coverage claim.  Under the
    usual exchangeability assumption, the finite-sample ``higher`` quantile of
    ``absolute_error - uncertainty`` gives a split-conformal additive envelope.
    The correction is constrained to be nonnegative so calibration never makes
    the raw uncertainty smaller.
    """
    uncertainty = np.asarray(calibration_uncertainty, dtype=np.float64)
    errors = np.asarray(calibration_absolute_error, dtype=np.float64)
    targets = np.asarray(target_uncertainty, dtype=np.float64)
    if uncertainty.ndim != 1 or errors.shape != uncertainty.shape:
        raise ValueError("calibration arrays must be equal one-dimensional arrays")
    if targets.ndim != 1:
        raise ValueError("target_uncertainty must be one-dimensional")
    if not len(uncertainty):
        raise ValueError("at least one calibration edge is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if (
        np.any(~np.isfinite(uncertainty))
        or np.any(~np.isfinite(errors))
        or np.any(~np.isfinite(targets))
        or np.any(uncertainty < 0)
        or np.any(errors < 0)
        or np.any(targets < 0)
    ):
        raise ValueError("uncertainty and absolute errors must be finite and nonnegative")
    scores = errors - uncertainty
    quantile_level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    correction = max(
        0.0,
        float(np.quantile(scores, quantile_level, method="higher")),
    )
    envelope = targets + correction
    empirical = float(np.mean(errors <= uncertainty + correction))
    return EdgeErrorEnvelope(
        values=envelope,
        additive_quantile=correction,
        nominal_coverage=1.0 - alpha,
        empirical_calibration_coverage=empirical,
        calibration_count=int(len(scores)),
    )


def ensemble_edge_uncertainty(ensemble_values, edge_index, *, ddof: int = 1) -> np.ndarray:
    """Standard deviation of edit effects across scalar-potential students."""
    values = np.asarray(ensemble_values, dtype=np.float64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("ensemble_values must have shape [members, nodes]")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if edges.size and (edges.min() < 0 or edges.max() >= values.shape[1]):
        raise ValueError("edge_index references a missing node")
    if not 0 <= ddof < values.shape[0]:
        raise ValueError("ddof must be smaller than the ensemble size")
    effects = values[:, edges[1]] - values[:, edges[0]]
    return effects.std(axis=0, ddof=ddof)


def path_edge_occupancy(edge_index, paths: Sequence[Sequence[int]]) -> np.ndarray:
    """Fraction of optimizer paths traversing each directed edge at least once."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if not paths:
        return np.zeros(edges.shape[1], dtype=np.float64)
    lookup: dict[tuple[int, int], list[int]] = {}
    for index, pair in enumerate(edges.T):
        lookup.setdefault((int(pair[0]), int(pair[1])), []).append(index)
    counts = np.zeros(edges.shape[1], dtype=np.float64)
    for path in paths:
        traversed: set[int] = set()
        for source, target in zip(path[:-1], path[1:]):
            traversed.update(lookup.get((int(source), int(target)), ()))
        if traversed:
            counts[list(traversed)] += 1
    return counts / len(paths)


def frontier_node_acquisition(
    edge_index,
    uncertainty,
    occupancy,
    queried_nodes: Iterable[int],
    budget: int,
    *,
    reduction: str = "max",
) -> AcquisitionBatch:
    """Select unqueried frontier nodes using `occupancy * edge uncertainty`.

    Cost is charged per unique target node. Only edges leaving an already queried
    source and entering an unqueried target are eligible.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    edges = np.asarray(edge_index, dtype=np.int64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if uncertainty.shape != (edges.shape[1],) or occupancy.shape != uncertainty.shape:
        raise ValueError("uncertainty and occupancy need one value per edge")
    if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty < 0):
        raise ValueError("uncertainty must be finite and nonnegative")
    if np.any(~np.isfinite(occupancy)) or np.any(occupancy < 0):
        raise ValueError("occupancy must be finite and nonnegative")
    if reduction not in {"max", "sum"}:
        raise ValueError("reduction must be max or sum")
    queried = frozenset(int(node) for node in queried_nodes)
    node_scores: dict[int, float] = {}
    candidates = 0
    for edge, (source, target) in enumerate(edges.T):
        source, target = int(source), int(target)
        if source not in queried or target in queried:
            continue
        candidates += 1
        score = float(uncertainty[edge] * occupancy[edge])
        if reduction == "sum":
            node_scores[target] = node_scores.get(target, 0.0) + score
        else:
            node_scores[target] = max(node_scores.get(target, -np.inf), score)
    ranked = sorted(node_scores.items(), key=lambda item: (-item[1], item[0]))[:budget]
    return AcquisitionBatch(
        node_indices=np.asarray([node for node, _ in ranked], dtype=np.int64),
        scores=np.asarray([score for _, score in ranked], dtype=np.float64),
        candidate_edges=candidates,
    )


def cost_aware_frontier_node_acquisition(
    edge_index,
    uncertainty,
    occupancy,
    queried_nodes: Iterable[int],
    node_cost,
    cost_budget: float,
    *,
    reduction: str = "max",
) -> CostAwareAcquisitionBatch:
    """Greedily purchase frontier nodes by acquisition value per oracle cost.

    This is a deterministic value-per-cost policy, not an exact knapsack solver
    or a proved expected regret-bound reduction.
    """
    if not np.isfinite(cost_budget) or cost_budget < 0:
        raise ValueError("cost_budget must be finite and nonnegative")
    edges = np.asarray(edge_index, dtype=np.int64)
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)
    costs = np.asarray(node_cost, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if uncertainty.shape != (edges.shape[1],) or occupancy.shape != uncertainty.shape:
        raise ValueError("uncertainty and occupancy need one value per edge")
    if costs.ndim != 1:
        raise ValueError("node_cost must be one-dimensional")
    if edges.size and (edges.min() < 0 or edges.max() >= len(costs)):
        raise ValueError("edge_index references a node without a cost")
    if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty < 0):
        raise ValueError("uncertainty must be finite and nonnegative")
    if np.any(~np.isfinite(occupancy)) or np.any(occupancy < 0):
        raise ValueError("occupancy must be finite and nonnegative")
    if np.any(~np.isfinite(costs)) or np.any(costs <= 0):
        raise ValueError("all node costs must be finite and strictly positive")
    if reduction not in {"max", "sum"}:
        raise ValueError("reduction must be max or sum")

    queried = frozenset(int(node) for node in queried_nodes)
    node_scores: dict[int, float] = {}
    candidates = 0
    for edge, (source, target) in enumerate(edges.T):
        source, target = int(source), int(target)
        if source not in queried or target in queried:
            continue
        candidates += 1
        score = float(uncertainty[edge] * occupancy[edge])
        if reduction == "sum":
            node_scores[target] = node_scores.get(target, 0.0) + score
        else:
            node_scores[target] = max(node_scores.get(target, -np.inf), score)
    ranked = sorted(
        node_scores.items(),
        key=lambda item: (-item[1] / costs[item[0]], -item[1], item[0]),
    )
    selected = []
    spent = 0.0
    tolerance = np.finfo(np.float64).eps * max(1.0, cost_budget) * 8
    for node, score in ranked:
        proposed = spent + float(costs[node])
        if proposed <= cost_budget + tolerance:
            selected.append((node, score, float(costs[node])))
            spent = proposed
    return CostAwareAcquisitionBatch(
        node_indices=np.asarray([node for node, _, _ in selected], dtype=np.int64),
        scores=np.asarray([score for _, score, _ in selected], dtype=np.float64),
        node_costs=np.asarray([cost for _, _, cost in selected], dtype=np.float64),
        total_cost=float(spent),
        candidate_edges=candidates,
    )
