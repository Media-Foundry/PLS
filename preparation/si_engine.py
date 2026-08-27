"""Resumable exhaustive Biopython sequence-identity engine.

Each upper-triangular block is an independent compressed NumPy artifact. A block
is complete only when its JSON marker exists and its SHA-256 matches, so an
interrupted run can safely resume without trusting partial files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import Bio
import numpy as np
from Bio.Align import PairwiseAligner


@dataclass(frozen=True)
class ScoringConfig:
    mode: str = "global"
    match_score: float = 1.0
    mismatch_score: float = 0.0
    open_gap_score: float = 0.0
    extend_gap_score: float = 0.0
    denominator: str = "alignment_columns_including_gaps"
    optimal_alignment_policy: str = "biopython_first_traceback"


def make_aligner(config: ScoringConfig) -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = config.mode
    aligner.match_score = config.match_score
    aligner.mismatch_score = config.mismatch_score
    aligner.open_gap_score = config.open_gap_score
    aligner.extend_gap_score = config.extend_gap_score
    return aligner


def sequence_identity(first: str, second: str, config: ScoringConfig = ScoringConfig()) -> float:
    """Return identity from Biopython's first deterministic optimal traceback.

    The policy is intentionally explicit: PairwiseAligner can expose many optimal
    alignments. Reproducibility therefore also requires the recorded Biopython
    version. Empty/empty is 1; exactly one empty sequence is 0.
    """
    if not first or not second:
        return float(first == second)
    alignment = make_aligner(config).align(first, second)[0]
    columns = alignment.shape[1]
    return alignment.counts().identities / columns if columns else 1.0


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = int(self.parent[item])
        return item

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _block_paths(output: Path, bi: int, bj: int) -> tuple[Path, Path]:
    stem = f"block_{bi:06d}_{bj:06d}"
    return output / "blocks" / f"{stem}.npz", output / "blocks" / f"{stem}.complete.json"


def _valid_block(output: Path, bi: int, bj: int) -> bool:
    data, marker = _block_paths(output, bi, bj)
    if not data.is_file() or not marker.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text())
        return metadata["sha256"] == _sha256(data)
    except (OSError, KeyError, json.JSONDecodeError):
        return False


_WORKER_SEQUENCES: list[str]
_WORKER_CONFIG: ScoringConfig
_WORKER_THRESHOLD: float
_WORKER_BLOCK_SIZE: int
_WORKER_OUTPUT: Path


def _init_worker(sequences, config, threshold, block_size, output):
    global _WORKER_SEQUENCES, _WORKER_CONFIG, _WORKER_THRESHOLD, _WORKER_BLOCK_SIZE, _WORKER_OUTPUT
    _WORKER_SEQUENCES = sequences
    _WORKER_CONFIG = config
    _WORKER_THRESHOLD = threshold
    _WORKER_BLOCK_SIZE = block_size
    _WORKER_OUTPUT = output


def _compute_block(pair: tuple[int, int]) -> tuple[int, int, int]:
    bi, bj = pair
    start_i, start_j = bi * _WORKER_BLOCK_SIZE, bj * _WORKER_BLOCK_SIZE
    stop_i = min(start_i + _WORKER_BLOCK_SIZE, len(_WORKER_SEQUENCES))
    stop_j = min(start_j + _WORKER_BLOCK_SIZE, len(_WORKER_SEQUENCES))
    similarities = np.full((stop_i - start_i, stop_j - start_j), np.nan, dtype=np.float32)
    edge_i, edge_j = [], []
    aligner = make_aligner(_WORKER_CONFIG)
    for local_i, i in enumerate(range(start_i, stop_i)):
        first = _WORKER_SEQUENCES[i]
        for local_j, j in enumerate(range(start_j, stop_j)):
            if j <= i:
                continue
            alignment = aligner.align(first, _WORKER_SEQUENCES[j])[0]
            value = alignment.counts().identities / alignment.shape[1]
            similarities[local_i, local_j] = value
            if value >= _WORKER_THRESHOLD:
                edge_i.append(i)
                edge_j.append(j)
    data_path, marker_path = _block_paths(_WORKER_OUTPUT, bi, bj)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(f".npz.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, similarity=similarities,
                            edge_i=np.asarray(edge_i, dtype=np.uint32),
                            edge_j=np.asarray(edge_j, dtype=np.uint32))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, data_path)
    checksum = _sha256(data_path)
    _atomic_json(marker_path, {"sha256": checksum, "pairs": int(np.isfinite(similarities).sum()), "edges": len(edge_i)})
    return bi, bj, len(edge_i)


def read_entities(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hashes = [row["sequence_sha256"] for row in rows]
    if hashes != sorted(hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("entities must be uniquely and deterministically ordered by sequence_sha256")
    return hashes, [row["sequence"] for row in rows]


def parse_shard_indices(specification: str, logical_shards: int) -> set[int]:
    """Parse comma-separated indices and inclusive ranges (for example 0-9,12)."""
    if specification == "all":
        return set(range(logical_shards))
    selected: set[int] = set()
    try:
        for token in specification.split(","):
            bounds = token.strip().split("-", 1)
            start = int(bounds[0])
            stop = int(bounds[-1])
            if start > stop:
                raise ValueError
            selected.update(range(start, stop + 1))
    except (ValueError, IndexError) as error:
        raise ValueError(f"invalid shard specification: {specification!r}") from error
    if not selected or min(selected) < 0 or max(selected) >= logical_shards:
        raise ValueError(f"shards must be between 0 and {logical_shards - 1}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--logical-shards", type=int, default=4096)
    parser.add_argument("--shard-indices", default="all",
                        help="comma-separated logical indices/ranges, or 'all'")
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args()
    if args.block_size < 1 or args.workers < 1 or args.logical_shards < 1 or not 0 <= args.threshold <= 1:
        parser.error("block-size/workers must be positive and threshold must be in [0,1]")
    try:
        selected_shards = parse_shard_indices(args.shard_indices, args.logical_shards)
    except ValueError as error:
        parser.error(str(error))

    hashes, sequences = read_entities(args.entities)
    scoring = ScoringConfig()
    run_config = {
        "schema_version": 1, "biopython_version": Bio.__version__,
        "entity_manifest_sha256": _sha256(args.entities), "entity_count": len(sequences),
        "block_size": args.block_size, "threshold": args.threshold,
        "logical_shards": args.logical_shards,
        "scoring": asdict(scoring),
    }
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError("existing run_config.json differs; use a new output directory")
    _atomic_json(config_path, run_config)

    block_count = (len(sequences) + args.block_size - 1) // args.block_size
    all_blocks = [(bi, bj) for bi in range(block_count) for bj in range(bi, block_count)]
    assigned_blocks = [pair for ordinal, pair in enumerate(all_blocks)
                       if ordinal % args.logical_shards in selected_shards]
    pending = [pair for pair in assigned_blocks if not _valid_block(args.output_dir, *pair)]
    print(f"entities={len(sequences):,} blocks={len(all_blocks):,} "
          f"assigned={len(assigned_blocks):,} pending={len(pending):,} "
          f"logical_shards={len(selected_shards):,}/{args.logical_shards:,}")
    if pending and not args.reduce_only:
        with Pool(args.workers, _init_worker, (sequences, scoring, args.threshold, args.block_size, args.output_dir)) as pool:
            for done, _result in enumerate(pool.imap_unordered(_compute_block, pending), 1):
                if done % 10 == 0 or done == len(pending):
                    print(f"completed {done:,}/{len(pending):,} pending blocks", flush=True)

    missing = [pair for pair in all_blocks if not _valid_block(args.output_dir, *pair)]
    if missing:
        print(f"Computation shard complete; global reduction deferred ({len(missing):,} blocks missing)")
        return

    union_find = UnionFind(len(sequences))
    nearest_score = np.full(len(sequences), -1, dtype=np.float32)
    nearest_index = np.full(len(sequences), -1, dtype=np.int64)
    for bi, bj in all_blocks:
        data_path, _ = _block_paths(args.output_dir, bi, bj)
        with np.load(data_path) as block:
            for first, second in zip(block["edge_i"], block["edge_j"]):
                union_find.union(int(first), int(second))
            matrix = block["similarity"]
            start_i, start_j = bi * args.block_size, bj * args.block_size
            for li, lj in np.argwhere(np.isfinite(matrix)):
                value, i, j = matrix[li, lj], start_i + int(li), start_j + int(lj)
                if value > nearest_score[i]: nearest_score[i], nearest_index[i] = value, j
                if value > nearest_score[j]: nearest_score[j], nearest_index[j] = value, i

    with (args.output_dir / "components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["sequence_sha256", "component_root_sha256"])
        for i, digest in enumerate(hashes): writer.writerow([digest, hashes[union_find.find(i)]])
    with (args.output_dir / "nearest_neighbors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["sequence_sha256", "neighbor_sha256", "identity"])
        for i, digest in enumerate(hashes):
            j = nearest_index[i]; writer.writerow([digest, "" if j < 0 else hashes[j], "" if j < 0 else f"{nearest_score[i]:.8f}"])
    print("Wrote components.csv and nearest_neighbors.csv")


if __name__ == "__main__":
    main()
