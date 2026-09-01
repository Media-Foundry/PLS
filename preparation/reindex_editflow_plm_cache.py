"""Reindex completed mean/residue PLM caches onto an exact-sequence subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pls.features.extract_esm2 import sha256


def read_entities(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def exact_reindex(source_rows: list[dict], target_rows: list[dict]) -> list[int]:
    source = {row["sequence_sha256"]: (index, row) for index, row in enumerate(source_rows)}
    if len(source) != len(source_rows):
        raise ValueError("source entities are not exact-sequence unique")
    indices = []
    for row in target_rows:
        record = source.get(row["sequence_sha256"])
        if record is None:
            raise ValueError("target sequence is absent from the source PLM cache")
        index, source_row = record
        if source_row["sequence"] != row["sequence"] or source_row["length"] != row["length"]:
            raise ValueError("target/source exact-sequence metadata mismatch")
        indices.append(index)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-entities", type=Path, required=True)
    parser.add_argument("--target-entities", type=Path, required=True)
    parser.add_argument("--source-offsets", type=Path, required=True)
    parser.add_argument("--target-offsets", type=Path, required=True)
    parser.add_argument("--source-mean", type=Path, required=True)
    parser.add_argument("--source-residue-pca", type=Path, required=True)
    parser.add_argument("--output-mean", type=Path, required=True)
    parser.add_argument("--output-residue-pca", type=Path, required=True)
    arguments = parser.parse_args()

    source_rows = read_entities(arguments.source_entities)
    target_rows = read_entities(arguments.target_entities)
    indices = exact_reindex(source_rows, target_rows)
    source_offsets = np.load(arguments.source_offsets, mmap_mode="r")
    target_offsets = np.load(arguments.target_offsets, mmap_mode="r")
    if len(source_offsets) != len(source_rows) + 1 or len(target_offsets) != len(target_rows) + 1:
        raise ValueError("entity/offset count mismatch")

    source_mean_status = np.load(arguments.source_mean / "status.npy", mmap_mode="r")
    if any(int(source_mean_status[index]) != 1 for index in indices):
        raise ValueError("source mean PLM cache is incomplete for the target subset")
    source_mean = np.load(arguments.source_mean / "embeddings.npy", mmap_mode="r")
    arguments.output_mean.mkdir(parents=True, exist_ok=True)
    target_mean = np.lib.format.open_memmap(
        arguments.output_mean / "embeddings.npy",
        mode="w+",
        dtype=source_mean.dtype,
        shape=(len(target_rows), source_mean.shape[1]),
    )
    target_mean[:] = source_mean[indices]
    target_mean.flush()
    np.save(arguments.output_mean / "status.npy", np.ones(len(target_rows), dtype=np.uint8))
    mean_config = json.loads((arguments.source_mean / "config.json").read_text())
    mean_config.update({
        "entity_count": len(target_rows),
        "entity_manifest_sha256": sha256(arguments.target_entities),
        "reindexed_from": str(arguments.source_mean.resolve()),
        "reindex_exact_sequence_only": True,
    })
    (arguments.output_mean / "config.json").write_text(
        json.dumps(mean_config, indent=2, sort_keys=True) + "\n"
    )

    source_pca_meta = json.loads(
        (arguments.source_residue_pca / "pca_metadata.json").read_text()
    )
    dimension = int(source_pca_meta["shape"][1])
    source_total = int(source_offsets[-1])
    target_total = int(target_offsets[-1])
    source_pca = np.memmap(
        arguments.source_residue_pca / "residue_esm2_pca.f16",
        mode="r",
        dtype=np.float16,
        shape=(source_total, dimension),
    )
    arguments.output_residue_pca.mkdir(parents=True, exist_ok=True)
    target_pca = np.memmap(
        arguments.output_residue_pca / "residue_esm2_pca.f16",
        mode="w+",
        dtype=np.float16,
        shape=(target_total, dimension),
    )
    for target_index, source_index in enumerate(indices):
        source_slice = slice(int(source_offsets[source_index]), int(source_offsets[source_index + 1]))
        target_slice = slice(int(target_offsets[target_index]), int(target_offsets[target_index + 1]))
        if source_slice.stop - source_slice.start != target_slice.stop - target_slice.start:
            raise ValueError("target/source residue span mismatch")
        target_pca[target_slice] = source_pca[source_slice]
    target_pca.flush()
    target_meta = {
        **source_pca_meta,
        "shape": [target_total, dimension],
        "reindexed_from": str(arguments.source_residue_pca.resolve()),
        "reindex_exact_sequence_only": True,
    }
    (arguments.output_residue_pca / "pca_metadata.json").write_text(
        json.dumps(target_meta, indent=2, sort_keys=True) + "\n"
    )
    np.save(
        arguments.output_residue_pca / "pca_status_shard_0.npy",
        np.ones(len(target_rows), dtype=np.uint8),
    )
    print(json.dumps({
        "schema": "PLS_EditFlow_reindexed_PLM_cache_v1",
        "source_entities": len(source_rows),
        "target_entities": len(target_rows),
        "target_residues": target_total,
        "test_evaluated": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
