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
