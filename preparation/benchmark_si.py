"""Benchmark canonical Biopython SI throughput on deterministic entity samples."""

from __future__ import annotations

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import Bio
from Bio.Align import PairwiseAligner


_SEQUENCES: list[str]
_BLOCK_SIZE: int


def _initialize(sequences: list[str], block_size: int) -> None:
    global _SEQUENCES, _BLOCK_SIZE
    _SEQUENCES, _BLOCK_SIZE = sequences, block_size


def _block(pair: tuple[int, int]) -> tuple[int, float]:
    bi, bj = pair
    start_i, start_j = bi * _BLOCK_SIZE, bj * _BLOCK_SIZE
    stop_i = min(start_i + _BLOCK_SIZE, len(_SEQUENCES))
    stop_j = min(start_j + _BLOCK_SIZE, len(_SEQUENCES))
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    count, checksum = 0, 0.0
    for i in range(start_i, stop_i):
        for j in range(max(i + 1, start_j), stop_j):
            alignment = aligner.align(_SEQUENCES[i], _SEQUENCES[j])[0]
            checksum += alignment.counts().identities / alignment.shape[1]
            count += 1
    return count, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--entity-count", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    with args.entities.open(newline="", encoding="utf-8") as handle:
        sequences = [row["sequence"] for _, row in zip(range(args.entity_count), csv.DictReader(handle))]
    blocks = (len(sequences) + args.block_size - 1) // args.block_size
    work = [(bi, bj) for bi in range(blocks) for bj in range(bi, blocks)]
    started = time.monotonic()
    with Pool(args.workers, _initialize, (sequences, args.block_size)) as pool:
        results = list(pool.imap_unordered(_block, work))
    elapsed = time.monotonic() - started
    pairs = sum(item[0] for item in results)
    print(f"biopython={Bio.__version__} entities={len(sequences)} workers={args.workers} "
          f"pairs={pairs} seconds={elapsed:.3f} pairs_per_second={pairs / elapsed:.1f} "
          f"checksum={sum(item[1] for item in results):.6f}")


if __name__ == "__main__":
    main()
