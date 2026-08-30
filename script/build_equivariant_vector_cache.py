"""Build an entity-aligned V4 vector/coordinate cache from audited raw features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.entities.open(newline="", encoding="utf-8") as handle:
        entities = list(csv.DictReader(handle))
    status = np.load(args.status, mmap_mode="r")
    offsets = np.load(args.offsets, mmap_mode="r")
    if len(status) != len(entities) or len(offsets) != len(entities) + 1:
        raise ValueError("entity, status, and offset arrays do not align")
    residues = int(offsets[-1])
    args.output.mkdir(parents=True, exist_ok=True)
    vectors = np.memmap(args.output / "vectors.f16", mode="w+", dtype=np.float16, shape=(residues, 8, 3))
    coordinates = np.memmap(args.output / "ca_coords.f32", mode="w+", dtype=np.float32, shape=(residues, 3))
    written = 0
    for entity, row in enumerate(entities):
        lo, hi = int(offsets[entity]), int(offsets[entity + 1])
        if not status[entity]:
            if hi != lo:
                raise ValueError(f"unavailable entity {entity} has non-empty compact offsets")
            continue
        digest = row["sequence_sha256"]
        path = args.raw_root / digest[:2] / f"{digest}.pt"
        features = torch.load(path, map_location="cpu", weights_only=False)
        observed_vectors = features["spatial_vector_features"].detach().cpu().numpy()
        observed_coordinates = features["ca_coords"].detach().cpu().numpy()
        if observed_vectors.shape != (hi - lo, 8, 3) or observed_coordinates.shape != (hi - lo, 3):
            raise ValueError(f"raw/compact residue mismatch for entity {entity}: {digest}")
        if not np.isfinite(observed_vectors).all() or not np.isfinite(observed_coordinates).all():
            raise ValueError(f"non-finite geometry for entity {entity}: {digest}")
        vectors[lo:hi] = observed_vectors.astype(np.float16)
        coordinates[lo:hi] = observed_coordinates.astype(np.float32)
        written += 1
    vectors.flush(); coordinates.flush()
    metadata = {
        "schema": "PLS_V4_equivariant_vector_cache_v1",
        "source": str(args.raw_root),
        "entities": len(entities),
        "available_entities": written,
        "residues": residues,
        "vector_dtype": "float16",
        "vector_shape": [residues, 8, 3],
        "coordinate_dtype": "float32",
        "coordinate_shape": [residues, 3],
        "source_entities_sha256": hashlib.sha256(args.entities.read_bytes()).hexdigest(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
