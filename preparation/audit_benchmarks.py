"""Reconstruct and compare public eSOL benchmark cohorts.

This script deliberately uses only the Python standard library. It treats source
repositories as immutable inputs pinned in ``preparation/sources.yaml`` and emits
a normalized manifest suitable for subsequent sequence-identity auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path


EXPECTED = {"train": 2019, "validation": 268, "test": 392}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def names_in_git_tree(repo: Path, prefix: str) -> set[str]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"],
        text=True,
    )
    names = set()
    for item in output.splitlines():
        if item.startswith(prefix) and item.endswith(".fasta"):
            stem = Path(item).stem
            names.add(stem.removesuffix("-model_v4"))
    return names


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgnnsol", type=Path, required=True)
    parser.add_argument("--surfsol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    csv_dir = args.fgnnsol / "dataset" / "csvFile"
    train = read_csv(csv_dir / "eSol_train.csv")
    held_out = read_csv(csv_dir / "eSol_test.csv")
    validation_names = names_in_git_tree(
        args.fgnnsol, "dataset/eval_data/fastaEval/"
    )
    test_names = names_in_git_tree(args.fgnnsol, "dataset/test_data/fastaTest/")

    if validation_names & test_names:
        raise ValueError("FGNNSol validation and test membership overlap")
    held_out_names = {row["gene"] for row in held_out}
    if validation_names | test_names != held_out_names:
        raise ValueError("FGNNSol directory membership does not match eSol_test.csv")

    surf_train = {
        sequence_hash(row["sequence"])
        for row in read_csv(args.surfsol / "data" / "sufsol_train2.csv")
    }
    surf_test = {
        sequence_hash(row["sequence"])
        for row in read_csv(args.surfsol / "data" / "sufsol_test2.csv")
    }

    normalized: list[dict[str, str]] = []
    for source_split, row in [
        *(("train", row) for row in train),
        *(("held_out", row) for row in held_out),
    ]:
        gene = row["gene"]
        split = (
            "train"
            if source_split == "train"
            else "validation"
            if gene in validation_names
            else "test"
        )
        seq = row["sequence"].strip().upper()
        seq_hash = sequence_hash(seq)
        normalized.append(
            {
                "protein_id": gene,
                "sequence": seq,
                "sequence_sha256": seq_hash,
                "solubility": row["solubility"],
                "split": split,
                "present_in_surfsol": str(
                    seq_hash in (surf_train if split == "train" else surf_test)
                ).lower(),
            }
        )

    counts = {
        split: sum(row["split"] == split for row in normalized) for split in EXPECTED
    }
    if counts != EXPECTED:
        raise ValueError(f"unexpected FGNNSol split counts: {counts}")
    hashes = [row["sequence_sha256"] for row in normalized]
    if len(hashes) != len(set(hashes)):
        raise ValueError("exact sequence leakage/duplication exists in FGNNSol cohort")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)

    missing_train = EXPECTED["train"] - len(surf_train)
    missing_held_out = EXPECTED["validation"] + EXPECTED["test"] - len(surf_test)
    print(f"FGNNSol split counts: {counts}")
    print(f"SurfSol preprocessing exclusions: train={missing_train}, held_out={missing_held_out}")
    print(f"Wrote {len(normalized)} records to {args.output}")


if __name__ == "__main__":
    main()
