"""Audit SI-threshold connected components and identify giant-component bridges."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.size = np.ones(size, dtype=np.uint32)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = int(self.parent[item])
        return item

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]

    def component_sizes(self) -> Counter:
        return Counter(self.find(index) for index in range(len(self.parent)))


def entropy(sequence: str) -> float:
    counts = Counter(sequence)
    length = len(sequence)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {name: float(np.quantile(array, value)) for name, value in
            (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--si-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    with args.entities.open(newline="", encoding="utf-8") as handle:
        entities = list(csv.DictReader(handle))
    hashes = [row["sequence_sha256"] for row in entities]
    sequences = [row["sequence"] for row in entities]
    index_by_hash = {digest: index for index, digest in enumerate(hashes)}
    with (args.si_dir / "components.csv").open(newline="", encoding="utf-8") as handle:
        component_by_hash = {row["sequence_sha256"]: row["component_root_sha256"]
                             for row in csv.DictReader(handle)}
    component_sizes = Counter(component_by_hash.values())
    giant_root, giant_size = component_sizes.most_common(1)[0]
    giant = np.asarray([component_by_hash[digest] == giant_root for digest in hashes])
    run_config = json.loads((args.si_dir / "run_config.json").read_text())
    block_size = int(run_config["block_size"])

    source_sets = [set() for _ in hashes]
    observation_counts = np.zeros(len(hashes), dtype=np.uint32)
    with args.observations.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = index_by_hash[row["sequence_sha256"]]
            source_sets[index].add(row["source_dataset"])
            observation_counts[index] += 1

    degree = np.zeros(len(hashes), dtype=np.uint32)
    cross_source_degree = np.zeros(len(hashes), dtype=np.uint32)
    min_neighbor_length = np.full(len(hashes), np.iinfo(np.uint16).max, dtype=np.uint16)
    max_neighbor_length = np.zeros(len(hashes), dtype=np.uint16)
    edge_bins = Counter()
    sensitivity_thresholds = (.30, .31, .35, .40, .50, .70, .90)
    sensitivity = {threshold: UnionFind(len(hashes)) for threshold in sensitivity_thresholds}
    motifs = {
        "MBP_core": "KIEEGKLVIWINGDKGYNGLAEVGKKF",
        "polyhistidine_6": "HHHHHH",
        "GFP_core": "FTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL",
    }
    lengths = np.asarray([len(sequence) for sequence in sequences])
    ambiguous = np.asarray([("X" in sequence or "B" in sequence) for sequence in sequences])
    fusion_motif = np.asarray([any(motif in sequence for motif in motifs.values()) for sequence in sequences])
    bridge_filters = {name: UnionFind(len(hashes)) for name in (
        "all", "minimum_length_30", "minimum_length_50", "minimum_length_100",
        "no_ambiguous", "no_known_fusion_motif", "length_ratio_ge_0.5",
        "length_ratio_ge_0.8", "combined_conservative",
    )}
    total_edges = giant_edges = cross_source_edges = 0
    block_paths = sorted((args.si_dir / "blocks").glob("block_*.npz"))
    for number, path in enumerate(block_paths, 1):
        with np.load(path) as block:
            first = block["edge_i"].astype(np.int64, copy=False)
            second = block["edge_j"].astype(np.int64, copy=False)
            if not len(first):
                continue
            matrix = block["similarity"]
            parts = path.stem.split("_")
            start_i, start_j = int(parts[1]) * block_size, int(parts[2]) * block_size
            local_i, local_j = first - start_i, second - start_j
            values = matrix[local_i, local_j]
        total_edges += len(first)
        np.add.at(degree, first, 1); np.add.at(degree, second, 1)
        lengths_first = np.fromiter((len(sequences[i]) for i in first), dtype=np.uint16)
        lengths_second = np.fromiter((len(sequences[i]) for i in second), dtype=np.uint16)
        np.minimum.at(min_neighbor_length, first, lengths_second)
        np.minimum.at(min_neighbor_length, second, lengths_first)
        np.maximum.at(max_neighbor_length, first, lengths_second)
        np.maximum.at(max_neighbor_length, second, lengths_first)
        in_giant = giant[first] & giant[second]
        giant_edges += int(in_giant.sum())
        cross = np.fromiter((source_sets[i].isdisjoint(source_sets[j]) for i, j in zip(first, second)),
                            dtype=bool, count=len(first))
        cross_source_edges += int(cross.sum())
        np.add.at(cross_source_degree, first[cross], 1)
        np.add.at(cross_source_degree, second[cross], 1)
        giant_values = values[in_giant]
        for threshold, union_find in sensitivity.items():
            selected = values >= threshold
            for i, j in zip(first[selected], second[selected]):
                union_find.union(int(i), int(j))
        edge_masks = {
            "all": np.ones(len(first), dtype=bool),
            "minimum_length_30": (lengths[first] >= 30) & (lengths[second] >= 30),
            "minimum_length_50": (lengths[first] >= 50) & (lengths[second] >= 50),
            "minimum_length_100": (lengths[first] >= 100) & (lengths[second] >= 100),
            "no_ambiguous": ~ambiguous[first] & ~ambiguous[second],
            "no_known_fusion_motif": ~fusion_motif[first] & ~fusion_motif[second],
            "length_ratio_ge_0.5": np.minimum(lengths[first], lengths[second]) / np.maximum(lengths[first], lengths[second]) >= .5,
            "length_ratio_ge_0.8": np.minimum(lengths[first], lengths[second]) / np.maximum(lengths[first], lengths[second]) >= .8,
        }
        edge_masks["combined_conservative"] = (edge_masks["minimum_length_50"] &
            edge_masks["no_ambiguous"] & edge_masks["no_known_fusion_motif"] &
            edge_masks["length_ratio_ge_0.5"])
        for name, union_find in bridge_filters.items():
            selected = edge_masks[name]
            for i, j in zip(first[selected], second[selected]):
                union_find.union(int(i), int(j))
        for lower, upper in ((.30, .31), (.31, .35), (.35, .50), (.50, .70), (.70, .90), (.90, 1.000001)):
            edge_bins[f"[{lower:.2f},{upper:.2f})"] += int(((giant_values >= lower) & (giant_values < upper)).sum())
        if number % 5000 == 0:
            print(f"scanned {number:,}/{len(block_paths):,} blocks", flush=True)

    min_neighbor_length[degree == 0] = 0
    candidate_indices = np.flatnonzero(giant)
    candidate_indices = sorted(candidate_indices, key=lambda i: (-int(degree[i]), len(sequences[i]), hashes[i]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["sequence_sha256", "length", "degree", "cross_source_degree", "neighbor_min_length",
              "neighbor_max_length", "entropy_bits", "has_X", "sources", "observation_count", "sequence"]
    with (args.output_dir / "giant_bridge_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index in candidate_indices[:args.top_n]:
            writer.writerow({
                "sequence_sha256": hashes[index], "length": len(sequences[index]),
                "degree": int(degree[index]), "cross_source_degree": int(cross_source_degree[index]),
                "neighbor_min_length": int(min_neighbor_length[index]),
                "neighbor_max_length": int(max_neighbor_length[index]),
                "entropy_bits": f"{entropy(sequences[index]):.6f}", "has_X": "X" in sequences[index],
                "sources": ";".join(sorted(source_sets[index])),
                "observation_count": int(observation_counts[index]), "sequence": sequences[index],
            })

    giant_indices = np.flatnonzero(giant)
    other_indices = np.flatnonzero(~giant)
    def group_summary(indices):
        lengths = [len(sequences[i]) for i in indices]
        entropies = [entropy(sequences[i]) for i in indices]
        return {"entities": len(indices), "length": quantiles(lengths), "entropy": quantiles(entropies),
                "with_X": sum("X" in sequences[i] for i in indices),
                "with_B": sum("B" in sequences[i] for i in indices),
                "source_entities": dict(Counter(source for i in indices for source in source_sets[i]))}
    threshold_report = {}
    for threshold, union_find in sensitivity.items():
        sizes = union_find.component_sizes()
        threshold_report[f"{threshold:.2f}"] = {
            "components": len(sizes), "largest_component": max(sizes.values()),
            "non_singleton_entities": sum(size for size in sizes.values() if size > 1),
        }
    bridge_filter_report = {}
    for name, union_find in bridge_filters.items():
        sizes = union_find.component_sizes()
        bridge_filter_report[name] = {
            "components": len(sizes), "largest_component": max(sizes.values()),
            "non_singleton_entities": sum(size for size in sizes.values() if size > 1),
        }
    motif_report = {
        name: {"giant": sum(motif in sequences[i] for i in giant_indices),
               "non_giant": sum(motif in sequences[i] for i in other_indices)}
        for name, motif in motifs.items()
    }
    report = {
        "schema_version": 1, "giant_component_root": giant_root, "giant_component_size": giant_size,
        "component_count": len(component_sizes), "singleton_components": sum(size == 1 for size in component_sizes.values()),
        "total_threshold_edges": total_edges, "giant_internal_edges": giant_edges,
        "cross_source_edges": cross_source_edges, "giant_edge_identity_bins": dict(edge_bins),
        "threshold_sensitivity": threshold_report, "known_fusion_motifs": motif_report,
        "bridge_filter_sensitivity_at_si_0.30": bridge_filter_report,
        "giant": group_summary(giant_indices), "non_giant": group_summary(other_indices),
    }
    (args.output_dir / "component_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
