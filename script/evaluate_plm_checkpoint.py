"""Evaluate an existing PLM-head checkpoint on strict validation and test sets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pls.models.plm_heads import PLMDatasetHeads
from pls.training.train_plm_heads import (ObservationDataset, SOURCE_TASK, collate,
                                          metrics_from_predictions, predict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--observation-split", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with args.entities.open(newline="", encoding="utf-8") as handle:
        hashes = [row["sequence_sha256"] for row in csv.DictReader(handle)]
    index_by_hash = {digest: index for index, digest in enumerate(hashes)}
    embeddings = np.load(args.embedding_dir / "embeddings.npy", mmap_mode="r")
    records = {split: [] for split in ("validation", "test")}
    with args.observation_split.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] in records:
                records[row["split"]].append((index_by_hash[row["sequence_sha256"]],
                                               SOURCE_TASK[row["source_dataset"]], float(row["target_value"])))
    loaders = {split: DataLoader(ObservationDataset(rows, embeddings), batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate) for split, rows in records.items()}
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PLMDatasetHeads(input_dimension=embeddings.shape[1]).to(device)
    model.load_state_dict(checkpoint["model"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.output_dir / "tensorboard")
    for split, loader in loaders.items():
        predictions, targets = predict(model, loader, device)
        report = metrics_from_predictions(predictions, targets)
        (args.output_dir / f"{split}_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for task, values in report.items():
            for name, value in values.items():
                if name != "n": writer.add_scalar(f"{split}/{task}/{name}", value, checkpoint["epoch"])
        print(split, json.dumps(report, sort_keys=True), flush=True)
    writer.close()


if __name__ == "__main__":
    main()
