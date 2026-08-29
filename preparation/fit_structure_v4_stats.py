"""Fit V4 scalar normalization using only strict-train PDBSol structures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-split", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_hashes = set()
    with args.observation_split.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train" and row["source_dataset"] == "PDBSol_ProtSolM":
                train_hashes.add(row["sequence_sha256"])
    total = torch.zeros(89, dtype=torch.float64)
    total_square = torch.zeros(89, dtype=torch.float64)
    residues = proteins = mismatches = 0
    for index, digest in enumerate(sorted(train_hashes), 1):
        path = args.feature_root / digest[:2] / f"{digest}.pt"
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not value["sequence_exact_match"]:
            mismatches += 1
            continue
        raw = value["spatial_scalar_raw_features"].to(torch.float64)
        total += raw.sum(0)
        total_square += raw.square().sum(0)
        residues += len(raw)
        proteins += 1
        if index % 5000 == 0:
            print(json.dumps({"done": index, "total": len(train_hashes), "proteins_used": proteins,
                              "residues": residues, "mismatches_excluded": mismatches}), flush=True)
    mean = total / residues
    variance = (total_square / residues - mean.square()).clamp_min(0)
    std = variance.sqrt()
    near_constant = std < 1e-8
    std[near_constant] = 1.0
    report = {"schema": "PLS_structure_v4_train_scalar_stats_v1", "strict_train_hashes": len(train_hashes),
              "proteins_used": proteins, "mismatches_excluded": mismatches, "residues_used": residues,
              "scalar_means": mean.tolist(), "scalar_stds": std.tolist(),
              "near_constant_indices": torch.where(near_constant)[0].tolist(),
              "observation_split": str(args.observation_split.resolve()),
              "feature_root": str(args.feature_root.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("proteins_used", "mismatches_excluded", "residues_used",
                                                    "near_constant_indices")}, indent=2))


if __name__ == "__main__":
    main()
