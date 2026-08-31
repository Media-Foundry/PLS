"""Implicit fixed-length Hamming graphs for combinatorial edit landscapes."""

from __future__ import annotations

import hashlib
import itertools

import numpy as np


def node_neighbors(index: int, alphabet_size: int, length: int) -> np.ndarray:
    """Return every Hamming-distance-one neighbor in base-alphabet node order."""
    nodes = alphabet_size ** length
    if not 0 <= index < nodes:
        raise ValueError("index is outside the Hamming graph")
    values, remainder = [], int(index)
    powers = [alphabet_size ** power for power in range(length - 1, -1, -1)]
    digits = []
    for power in powers:
        digit, remainder = divmod(remainder, power)
        digits.append(digit)
    for position, power in enumerate(powers):
        base = index - digits[position] * power
        for replacement in range(alphabet_size):
            if replacement != digits[position]:
                values.append(base + replacement * power)
    return np.asarray(values, dtype=np.int64)


def hamming_graph_edges(alphabet_size: int, length: int, *, directed: bool = False) -> np.ndarray:
    """Enumerate distance-one edges in base-`alphabet_size` node order.

    Undirected mode emits each edge once. Directed mode emits both orientations.
    """
    if alphabet_size < 2 or length < 1:
        raise ValueError("alphabet_size must be >=2 and length must be >=1")
    nodes = alphabet_size ** length
    digits = np.arange(nodes, dtype=np.int64).reshape((alphabet_size,) * length)
    sources, targets = [], []
    for position in range(length):
        for source_value in range(alphabet_size):
            source = np.take(digits, source_value, axis=position).reshape(-1)
            for target_value in range(source_value + 1, alphabet_size):
                target = np.take(digits, target_value, axis=position).reshape(-1)
                sources.append(source);targets.append(target)
    edge_index = np.stack((np.concatenate(sources), np.concatenate(targets))).astype(np.int32)
    if directed:
        edge_index = np.concatenate((edge_index, edge_index[::-1]), axis=1)
    return edge_index


def variants_from_tokens(tokens, alphabet: str) -> list[str]:
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or np.any(tokens < 0) or np.any(tokens >= len(alphabet)):
        raise ValueError("tokens must be [nodes, length] within the alphabet")
    return ["".join(alphabet[int(value)] for value in row) for row in tokens]


def hash_partition(nodes, *, salt: str, development_fraction: float = 0.2) -> np.ndarray:
    """Value-blind deterministic development/evaluation partition."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must lie strictly between zero and one")
    labels = np.zeros(len(nodes), dtype=np.uint8)
    cutoff = int(development_fraction * (1 << 64))
    for index, node in enumerate(nodes):
        digest = hashlib.sha256(f"{salt}:{node}".encode()).digest()
        labels[index] = int.from_bytes(digest[:8], "big") >= cutoff
    return labels  # 0=development, 1=evaluation


def hamming_distance(tokens, anchor) -> np.ndarray:
    tokens = np.asarray(tokens)
    anchor = np.asarray(anchor)
    if tokens.ndim != 2 or anchor.shape != (tokens.shape[1],):
        raise ValueError("anchor must have one token per sequence position")
    return np.count_nonzero(tokens != anchor, axis=1)


def queried_nodes_sha256(nodes) -> str:
    """Order-independent identity for an exact queried-node set."""
    if isinstance(nodes, np.ndarray):
        values = np.asarray(nodes, dtype=np.int64).reshape(-1)
    else:
        values = np.fromiter((int(node) for node in nodes), dtype=np.int64)
    canonical = np.unique(values).astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def select_hashed_anchors(
    variants,
    eligible,
    count: int,
    *,
    salt: str,
    excluded=(),
) -> np.ndarray:
    """Select anchors by a value-blind SHA-256 priority over variant names."""
    variants = list(variants)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(variants),):
        raise ValueError("eligible must have one value per variant")
    if count < 1:
        raise ValueError("count must be positive")
    excluded = frozenset(map(str, excluded))
    ranked = []
    for node, variant in enumerate(variants):
        if eligible[node] and str(variant) not in excluded:
            digest = hashlib.sha256(f"{salt}:{variant}".encode()).digest()
            ranked.append((digest, node))
    if len(ranked) < count:
        raise ValueError("not enough eligible anchors")
    ranked.sort()
    return np.asarray([node for _, node in ranked[:count]], dtype=np.int64)
