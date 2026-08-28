"""Train dataset-specific heads on cached frozen ESM-2 mean embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pls.models.plm_heads import PLMDatasetHeads


SOURCE_TASK = {"UESolDS_PLM_Sol_1.1": "uesolds", "PDBSol_ProtSolM": "pdbsol",
               "eSOL_FGNNSol": "esol"}


class ObservationDataset(Dataset):
    def __init__(self, records, embeddings):
        self.records = records
        self.embeddings = embeddings
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        entity_index, task, target = self.records[index]
        return torch.from_numpy(np.array(self.embeddings[entity_index], copy=True)), task, target


def collate(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch], torch.tensor([item[2] for item in batch])


def task_loss(model, embeddings, tasks, targets):
    losses = []
    for task in SOURCE_TASK.values():
        indices = [index for index, value in enumerate(tasks) if value == task]
        if not indices: continue
        selected = torch.tensor(indices, device=embeddings.device)
        prediction = model(embeddings.index_select(0, selected), task)
        truth = targets.index_select(0, selected)
        losses.append((nn.functional.smooth_l1_loss(prediction, truth) if task == "esol" else
                       nn.functional.binary_cross_entropy_with_logits(prediction, truth)))
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--observation-split", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("HIP_VISIBLE_DEVICES") != "7":
        parser.error("GPU training is pinned to physical device 7; set HIP_VISIBLE_DEVICES=7")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    with args.entities.open(newline="", encoding="utf-8") as handle:
        hashes = [row["sequence_sha256"] for row in csv.DictReader(handle)]
    index_by_hash = {digest: index for index, digest in enumerate(hashes)}
    embeddings = np.load(args.embedding_dir / "embeddings.npy", mmap_mode="r")
    status = np.load(args.embedding_dir / "status.npy", mmap_mode="r")
    if len(embeddings) != len(hashes): raise ValueError("embedding/entity count mismatch")
    if not args.allow_incomplete and np.any(status != 1): raise ValueError("embedding extraction is incomplete")

    records = {split: [] for split in ("train", "validation", "test")}
    with args.observation_split.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = index_by_hash[row["sequence_sha256"]]
            if status[index] != 1: continue
            records[row["split"]].append((index, SOURCE_TASK[row["source_dataset"]], float(row["target_value"])))
    task_counts = Counter(task for _, task, _ in records["train"])
    weights = [1 / task_counts[task] for _, task, _ in records["train"]]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
    train_loader = DataLoader(ObservationDataset(records["train"], embeddings), batch_size=args.batch_size,
                              sampler=sampler, collate_fn=collate, num_workers=0)
    validation_loader = DataLoader(ObservationDataset(records["validation"], embeddings), batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate, num_workers=0)

    device = torch.device(args.device)
    model = PLMDatasetHeads(input_dimension=embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train(); train_sum = train_count = 0
        for features, tasks, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True); loss = task_loss(model, features, tasks, targets)
            loss.backward(); optimizer.step(); train_sum += loss.item() * len(tasks); train_count += len(tasks)
        model.eval(); validation_sum = validation_count = 0
        with torch.inference_mode():
            for features, tasks, targets in validation_loader:
                features, targets = features.to(device), targets.to(device)
                loss = task_loss(model, features, tasks, targets)
                validation_sum += loss.item() * len(tasks); validation_count += len(tasks)
        row = {"epoch": epoch, "train_loss": train_sum / train_count,
               "validation_loss": validation_sum / validation_count}
        history.append(row); print(json.dumps(row), flush=True)
        if row["validation_loss"] < best:
            best, stale = row["validation_loss"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "validation_loss": best,
                        "task_counts": dict(task_counts), "args": vars(args)}, args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
