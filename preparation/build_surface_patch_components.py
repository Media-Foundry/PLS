"""Build mesh-free connected surface-patch assignments from the frozen PDBSol cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CATEGORIES = {
    "surface": set("ACDEFGHIKLMNPQRSTVWY"),
    "hydrophobic": set("AVILMFWY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "aromatic": set("FWYH"),
}


def components(sequence: str, rsa: np.ndarray, neighbors: np.ndarray, distances: np.ndarray,
               rsa_threshold: float, edge_cutoff: float) -> np.ndarray:
    """Return local component ids [residue, category], with -1 for non-members."""
    length = len(sequence)
    result = np.full((length, len(CATEGORIES)), -1, dtype=np.int32)
    exposed = rsa >= rsa_threshold
    for category, allowed in enumerate(CATEGORIES.values()):
        member = exposed & np.fromiter((aa in allowed for aa in sequence), bool, length)
        parent = np.arange(length, dtype=np.int32)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        for source in np.flatnonzero(member):
            valid = (distances[source] > 1e-6) & (distances[source] <= edge_cutoff)
            for target in neighbors[source, valid]:
                target = int(target)
                if target >= length or not member[target]:
                    continue
                left, right = find(int(source)), find(target)
                if left != right:
                    parent[right] = left
        roots = {}
        for index in np.flatnonzero(member):
            root = find(int(index))
            if root not in roots:
                roots[root] = len(roots)
            result[index, category] = roots[root]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, default=Path("benchmark/generated/sequence_entities.csv"))
    parser.add_argument("--compact", type=Path, default=Path("artifacts/features/pdbsol_structure_v4_compact"))
    parser.add_argument("--geometry", type=Path, default=Path("artifacts/features/pdbsol_structure_v4_geometry"))
    parser.add_argument("--structure-stats", type=Path, default=Path("artifacts/features/pdbsol_structure_v4_train_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features/pdbsol_surface_patch_components_v1"))
    parser.add_argument("--rsa-threshold", type=float, default=.25)
    parser.add_argument("--edge-cutoff", type=float, default=8.)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.entities.open(newline="", encoding="utf-8")))
    offsets = np.load(args.compact / "offsets.npy", mmap_mode="r")
    compact_meta = json.loads((args.compact / "metadata.json").read_text())
    geometry_meta = json.loads((args.geometry / "metadata.json").read_text())
    structure_stats = json.loads(args.structure_stats.read_text())
    shape = tuple(compact_meta["shape"]); neighbor_shape = (geometry_meta["residues"], geometry_meta["neighbors"])
    if len(offsets) != len(rows) + 1 or shape[0] != neighbor_shape[0] or int(offsets[-1]) != shape[0]:
        raise ValueError("entity, compact, and geometry caches are not aligned")
    features = np.memmap(args.compact / "residue_features.f16", mode="r", dtype=np.float16, shape=shape)
    neighbors = np.memmap(args.geometry / "neighbors.i16", mode="r", dtype=np.int16, shape=neighbor_shape)
    distances = np.memmap(args.geometry / "distances.f16", mode="r", dtype=np.float16, shape=neighbor_shape)
    args.output.mkdir(parents=True, exist_ok=False)
    labels = np.memmap(args.output / "component_ids.i32", mode="w+", dtype=np.int32,
                       shape=(shape[0], len(CATEGORIES)))
    labels[:] = -1
    patch_counts = np.zeros((len(rows), len(CATEGORIES)), dtype=np.int32)
    for entity, row in enumerate(rows):
        lo, hi = int(offsets[entity]), int(offsets[entity + 1]); sequence = row["sequence"]
        if hi - lo == 0:
            continue
        if hi - lo != len(sequence):
            raise ValueError(f"sequence/cache length mismatch for entity {entity}")
        rsa = np.clip(np.asarray(features[lo:hi, 63], np.float32) * structure_stats["scalar_stds"][1] + structure_stats["scalar_means"][1], 0, 1)
        current = components(sequence, rsa,
                             np.asarray(neighbors[lo:hi], np.int64), np.asarray(distances[lo:hi], np.float32),
                             args.rsa_threshold, args.edge_cutoff)
        labels[lo:hi] = current
        patch_counts[entity] = [int(current[:, column].max()) + 1 for column in range(current.shape[1])]
    labels.flush(); np.save(args.output / "patch_counts.npy", patch_counts)
    source_digest = hashlib.sha256(args.entities.read_bytes()).hexdigest()
    metadata = {"schema": "PLS_mesh_free_surface_components_v1", "shape": [shape[0], len(CATEGORIES)],
                "entities": len(rows), "categories": list(CATEGORIES), "standardized_rsa_feature_index": 63,
                "spatial_scalar_rsa_index": 1, "rsa_inverse_transform_source": str(args.structure_stats),
                "rsa_threshold": args.rsa_threshold, "edge_cutoff_angstrom": args.edge_cutoff,
                "source_entities_sha256": source_digest, "coordinate_system": "entity-local component ids",
                "non_member_value": -1}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "residues": shape[0], "entities": len(rows),
                      "patches_by_category": dict(zip(CATEGORIES, patch_counts.sum(0).astype(int).tolist()))}, indent=2))


if __name__ == "__main__":
    main()
