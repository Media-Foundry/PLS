"""Beam trajectories and path-aware frontier acquisition on edit graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import (AcquisitionBatch, ensemble_edge_uncertainty,
                          frontier_node_acquisition, path_edge_occupancy)
from .hamming import node_neighbors


def beam_search_paths(
    values,
    anchor: int,
    available,
    *,
    alphabet_size: int,
    length: int,
    steps: int,
    beam_width: int,
) -> list[tuple[int, ...]]:
    """Return high-potential, cycle-free paths at all depths up to `steps`."""
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    if values.ndim != 1 or available.shape != values.shape:
        raise ValueError("values and available must be equal one-dimensional arrays")
    if not 0 <= anchor < len(values) or not available[anchor]:
        raise ValueError("anchor must be an available node")
    if steps < 1 or beam_width < 1:
        raise ValueError("steps and beam_width must be positive")
    beam: list[tuple[int, ...]] = [(int(anchor),)]
    collected: list[tuple[int, ...]] = []
    for _ in range(steps):
        by_endpoint: dict[int, tuple[int, ...]] = {}
        for path in beam:
            for neighbor in node_neighbors(path[-1], alphabet_size, length):
                neighbor = int(neighbor)
                if not available[neighbor] or neighbor in path:
                    continue
                proposed = path + (neighbor,)
                previous = by_endpoint.get(neighbor)
                if previous is None or len(proposed) < len(previous):
                    by_endpoint[neighbor] = proposed
        if not by_endpoint:
            break
        beam = sorted(
            by_endpoint.values(), key=lambda path: (-values[path[-1]], path)
        )[:beam_width]
        collected.extend(beam)
    return collected


@dataclass(frozen=True)
class PathAwareAcquisition:
    batch: AcquisitionBatch
    paths: tuple[tuple[int, ...], ...]
    path_edges: np.ndarray
    occupancy: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class BoundAwareAcquisition:
    """Frontier batch targeting shortest uncertainty-bound routes."""

    batch: AcquisitionBatch
    candidate_endpoints: np.ndarray
    selected_paths: tuple[tuple[int, ...], ...]
    path_edges: np.ndarray
    occupancy: np.ndarray
    uncertainty: np.ndarray
    estimated_path_bounds: np.ndarray


def hybrid_query_budget(total_budget: int, targeted_fraction: float) -> int:
    """Allocate a fixed query share to path targeting, leaving exploration room."""
    if total_budget < 1:
        raise ValueError("total_budget must be positive")
    if not 0.0 < targeted_fraction < 1.0:
        raise ValueError("targeted_fraction must lie strictly between zero and one")
    if total_budget == 1:
        return 1
    targeted = int(np.floor(total_budget * targeted_fraction + 0.5))
    return min(max(targeted, 1), total_budget - 1)


def path_aware_frontier_acquisition(
    ensemble_values,
    queried_nodes,
    available,
    anchor: int,
    budget: int,
    *,
    alphabet_size: int,
    length: int,
    steps: int,
    beam_width: int,
    conservative_beta: float = 0.0,
) -> PathAwareAcquisition:
    """Acquire frontier nodes prioritized by optimizer occupancy times uncertainty."""
    ensemble = np.asarray(ensemble_values, dtype=np.float64)
    if ensemble.ndim != 2:
        raise ValueError("ensemble_values must have shape [members, nodes]")
    if conservative_beta < 0:
        raise ValueError("conservative_beta must be nonnegative")
    mean = ensemble.mean(0);standard_deviation = ensemble.std(0)
    objectives = [*ensemble, mean - conservative_beta * standard_deviation]
    paths = []
    for objective in objectives:
        paths.extend(beam_search_paths(
            objective, anchor, available, alphabet_size=alphabet_size,
            length=length, steps=steps, beam_width=beam_width,
        ))
    unique_edges = sorted({(source, target) for path in paths for source, target in zip(path[:-1], path[1:])})
    edge_index = np.asarray(unique_edges, dtype=np.int64).T if unique_edges else np.empty((2, 0), dtype=np.int64)
    occupancy = path_edge_occupancy(edge_index, paths)
    uncertainty = ensemble_edge_uncertainty(ensemble, edge_index)
    batch = frontier_node_acquisition(
        edge_index, uncertainty, occupancy, queried_nodes, budget, reduction="max"
    )
    return PathAwareAcquisition(
        batch=batch, paths=tuple(paths), path_edges=edge_index,
        occupancy=occupancy, uncertainty=uncertainty,
    )


def bound_aware_frontier_acquisition(
    ensemble_values,
    queried_nodes,
    available,
    anchor: int,
    budget: int,
    *,
    alphabet_size: int,
    length: int,
    steps: int,
    beam_width: int,
    conservative_beta: float = 0.0,
) -> BoundAwareAcquisition:
    """Acquire edges contributing to plausible optima's uncertainty bounds.

    Each ensemble member plus a conservative ensemble objective proposes an
    optimum. For every unique proposed endpoint, the algorithm retains the
    beam-discovered route with minimum cumulative ensemble edge uncertainty.
    Query priority is the edge's uncertainty times its occupancy among these
    shortest-bound routes.
    """
    ensemble = np.asarray(ensemble_values, dtype=np.float64)
    if ensemble.ndim != 2:
        raise ValueError("ensemble_values must have shape [members, nodes]")
    if conservative_beta < 0:
        raise ValueError("conservative_beta must be nonnegative")
    mean = ensemble.mean(0)
    standard_deviation = ensemble.std(0)
    objectives = [*ensemble, mean - conservative_beta * standard_deviation]
    all_paths: list[tuple[int, ...]] = []
    proposed_endpoints = []
    for objective in objectives:
        paths = beam_search_paths(
            objective,
            anchor,
            available,
            alphabet_size=alphabet_size,
            length=length,
            steps=steps,
            beam_width=beam_width,
        )
        if not paths:
            continue
        all_paths.extend(paths)
        best = min(paths, key=lambda path: (-objective[path[-1]], len(path), path))
        proposed_endpoints.append(int(best[-1]))
    endpoints = np.asarray(sorted(set(proposed_endpoints)), dtype=np.int64)
    unique_edges = sorted({
        (source, target)
        for path in all_paths
        for source, target in zip(path[:-1], path[1:])
    })
    edge_index = (
        np.asarray(unique_edges, dtype=np.int64).T
        if unique_edges
        else np.empty((2, 0), dtype=np.int64)
    )
    uncertainty = ensemble_edge_uncertainty(ensemble, edge_index)
    edge_uncertainty = {
        (int(source), int(target)): float(value)
        for (source, target), value in zip(edge_index.T, uncertainty)
    }
    selected_paths = []
    estimated_bounds = []
    for endpoint in endpoints:
        routes = [path for path in all_paths if path[-1] == int(endpoint)]
        route = min(
            routes,
            key=lambda path: (
                sum(edge_uncertainty[pair] for pair in zip(path[:-1], path[1:])),
                len(path),
                path,
            ),
        )
        selected_paths.append(route)
        estimated_bounds.append(
            sum(edge_uncertainty[pair] for pair in zip(route[:-1], route[1:]))
        )
    occupancy = path_edge_occupancy(edge_index, selected_paths)
    batch = frontier_node_acquisition(
        edge_index,
        uncertainty,
        occupancy,
        queried_nodes,
        budget,
        reduction="max",
    )
    return BoundAwareAcquisition(
        batch=batch,
        candidate_endpoints=endpoints,
        selected_paths=tuple(selected_paths),
        path_edges=edge_index,
        occupancy=occupancy,
        uncertainty=uncertainty,
        estimated_path_bounds=np.asarray(estimated_bounds, dtype=np.float64),
    )
