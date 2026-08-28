"""Validate a strict component split against every cached SI matrix block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--entity-split", type=Path, required=True)
    parser.add_argument("--si-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--neighbors-output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=1000)
    args = parser.parse_args()

    with args.entities.open(newline="", encoding="utf-8") as handle:
        hashes = [row["sequence_sha256"] for row in csv.DictReader(handle)]
    with args.entity_split.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    split_by_hash = {row["sequence_sha256"]: row["split"] for row in split_rows}
    if len(split_by_hash) != len(hashes) or set(split_by_hash) != set(hashes):
        raise ValueError("entity split must contain every canonical entity exactly once")
    split_names = sorted(set(split_by_hash.values()))
    split_code = {name: code for code, name in enumerate(split_names)}
    codes = np.asarray([split_code[split_by_hash[digest]] for digest in hashes], dtype=np.uint8)
    config = json.loads((args.si_dir / "run_config.json").read_text())
    block_size = int(config["block_size"])

    threshold_violations = 0
    cross_pairs = 0
    top: list[tuple[float, int, int]] = []
    blocks = sorted((args.si_dir / "blocks").glob("block_*.npz"))
    for number, path in enumerate(blocks, 1):
        parts = path.stem.split("_")
        start_i, start_j = int(parts[1]) * block_size, int(parts[2]) * block_size
        with np.load(path) as block:
            matrix = block["similarity"]
            edge_i = block["edge_i"].astype(np.int64, copy=False)
            edge_j = block["edge_j"].astype(np.int64, copy=False)
            threshold_violations += int(np.sum(codes[edge_i] != codes[edge_j]))
        row_codes = codes[start_i:start_i + matrix.shape[0]]
        column_codes = codes[start_j:start_j + matrix.shape[1]]
        cross = (row_codes[:, None] != column_codes[None, :]) & np.isfinite(matrix)
        cross_pairs += int(cross.sum())
        if np.any(cross):
            candidate = np.where(cross, matrix, -np.inf)
            flat_count = min(args.top_n, int(cross.sum()))
            flat_indices = np.argpartition(candidate.ravel(), -flat_count)[-flat_count:]
            for flat_index in flat_indices:
                local_i, local_j = np.unravel_index(flat_index, candidate.shape)
                item = (float(candidate[local_i, local_j]), start_i + int(local_i), start_j + int(local_j))
                if len(top) < args.top_n:
                    heapq.heappush(top, item)
                elif item > top[0]:
                    heapq.heapreplace(top, item)
        if number % 10000 == 0:
            print(f"validated {number:,}/{len(blocks):,} blocks", flush=True)

    top.sort(reverse=True)
    args.neighbors_output.parent.mkdir(parents=True, exist_ok=True)
    with args.neighbors_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["identity", "first_sha256", "first_split", "second_sha256", "second_split"])
        for identity, first, second in top:
            writer.writerow([f"{identity:.8f}", hashes[first], split_by_hash[hashes[first]],
                             hashes[second], split_by_hash[hashes[second]]])
    report = {
        "schema_version": 1, "entity_count": len(hashes), "block_count": len(blocks),
        "cross_split_pair_count": cross_pairs, "threshold_edge_cross_split_violations": threshold_violations,
        "maximum_cross_split_identity": top[0][0] if top else None,
        "entity_manifest_sha256": sha256(args.entities), "entity_split_sha256": sha256(args.entity_split),
        "run_config_sha256": sha256(args.si_dir / "run_config.json"),
    }
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if threshold_violations or (top and top[0][0] >= float(config["threshold"])):
        raise SystemExit("strict cross-split SI invariant failed")


if __name__ == "__main__":
    main()
