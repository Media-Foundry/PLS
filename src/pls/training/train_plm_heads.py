"""Config-driven training for dataset-specific frozen-PLM heads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from pls.evaluation.metrics import binary_metrics, regression_metrics
from pls.models.plm_heads import PLMDatasetHeads, TASKS


SOURCE_TASK = {"UESolDS_PLM_Sol_1.1": "uesolds", "PDBSol_ProtSolM": "pdbsol",
               "eSOL_FGNNSol": "esol"}


class ObservationDataset(Dataset):
    def __init__(self, records, embeddings): self.records, self.embeddings = records, embeddings
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        entity_index, task, target = self.records[index]
        return torch.from_numpy(np.array(self.embeddings[entity_index], copy=True)), task, target


def collate(batch):
    return (torch.stack([item[0] for item in batch]), [item[1] for item in batch],
            torch.tensor([item[2] for item in batch], dtype=torch.float32))


def task_loss(model, embeddings, tasks, targets):
    losses = []
    for task in TASKS:
        indices = [index for index, value in enumerate(tasks) if value == task]
        if not indices: continue
        selected = torch.tensor(indices, device=embeddings.device)
        prediction = model(embeddings.index_select(0, selected), task)
        truth = targets.index_select(0, selected)
        losses.append(nn.functional.smooth_l1_loss(prediction, truth) if task == "esol" else
                      nn.functional.binary_cross_entropy_with_logits(prediction, truth))
    return torch.stack(losses).mean()


def predict(model, loader, device):
    model.eval(); predictions, targets = defaultdict(list), defaultdict(list)
    with torch.inference_mode():
        for features, tasks, truth in loader:
            features = features.to(device)
            for task in TASKS:
                indices = [index for index, value in enumerate(tasks) if value == task]
                if not indices: continue
                selected = torch.tensor(indices, device=device)
                output = model(features.index_select(0, selected), task).float().cpu().numpy()
                predictions[task].extend(output.tolist())
                targets[task].extend(truth[indices].numpy().tolist())
    return predictions, targets


def metrics_from_predictions(predictions, targets):
    report = {}
    for task in TASKS:
        if task not in predictions: continue
        report[task] = (regression_metrics(targets[task], predictions[task]) if task == "esol" else
                        binary_metrics(targets[task], predictions[task]))
    return report


def validation_objective(metrics, selection="balanced"):
    # All endpoints contribute equally. Higher correlations/AUROC become a
    # minimization objective while retaining interpretable endpoint metrics.
    if selection == "esol_spearman":
        return float(1 - metrics["esol"]["spearman"])
    if selection == "pdbsol_auroc":
        return float(1 - metrics["pdbsol"]["auroc"])
    if selection == "uesolds_auroc":
        return float(1 - metrics["uesolds"]["auroc"])
    return float(np.mean([1 - metrics["uesolds"]["auroc"], 1 - metrics["pdbsol"]["auroc"],
                          1 - metrics["esol"]["spearman"]]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allow-test-evaluation", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("evaluate_test", False) and not args.allow_test_evaluation:
        parser.error("test evaluation requires explicit --allow-test-evaluation after model freeze")
    data_config, model_config, training = config["data"], config["model"], config["training"]
    device_name = training.get("device", "cuda:0")
    expected_hip = str(training.get("hip_device", 7))
    if device_name.startswith("cuda") and os.environ.get("HIP_VISIBLE_DEVICES") != expected_hip:
        parser.error(f"HIP_VISIBLE_DEVICES must equal configured physical ordinal {expected_hip}")
    seed = int(training["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    entities_path = Path(data_config["entities"])
    split_path = Path(data_config["observation_split"])
    embedding_dir = Path(data_config["embedding_dir"])
    with entities_path.open(newline="", encoding="utf-8") as handle:
        hashes = [row["sequence_sha256"] for row in csv.DictReader(handle)]
    index_by_hash = {digest: index for index, digest in enumerate(hashes)}
    embeddings = np.load(embedding_dir / "embeddings.npy", mmap_mode="r")
    status = np.load(embedding_dir / "status.npy", mmap_mode="r")
    if len(embeddings) != len(hashes) or np.any(status != 1):
        raise ValueError("complete embeddings for every entity are required")
    records = {split: [] for split in ("train", "validation", "test")}
    with split_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records[row["split"]].append((index_by_hash[row["sequence_sha256"]],
                                          SOURCE_TASK[row["source_dataset"]], float(row["target_value"])))
    datasets = {split: ObservationDataset(rows, embeddings) for split, rows in records.items()}
    task_counts = Counter(task for _, task, _ in records["train"])
    weights = [1 / task_counts[task] for _, task, _ in records["train"]]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    batch_size = int(training["batch_size"])
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler,
                              collate_fn=collate, num_workers=0)
    loaders = {split: DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate, num_workers=0)
               for split, dataset in datasets.items() if split != "train"}

    device = torch.device(device_name)
    model = PLMDatasetHeads(input_dimension=embeddings.shape[1], **model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    writer = SummaryWriter(args.run_dir / "tensorboard")
    checkpoint_dir = args.run_dir / "checkpoints"
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); loss_sum = count = 0
        for features, tasks, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True); loss = task_loss(model, features, tasks, targets)
            loss.backward(); optimizer.step(); loss_sum += loss.item() * len(tasks); count += len(tasks)
        validation_predictions, validation_targets = predict(model, loaders["validation"], device)
        validation_metrics = metrics_from_predictions(validation_predictions, validation_targets)
        objective = validation_objective(validation_metrics, training.get("selection_objective", "balanced"))
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               "validation_objective": objective, "validation": validation_metrics}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        writer.add_scalar("loss/train", row["train_loss"], epoch)
        writer.add_scalar("objective/validation", objective, epoch)
        for task, values in validation_metrics.items():
            for name, value in values.items():
                if name != "n": writer.add_scalar(f"validation/{task}/{name}", value, epoch)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "validation_objective": objective, "config": config, "task_counts": dict(task_counts)}
        if epoch % int(training["checkpoint_every"]) == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if objective < best:
            best, stale = objective, 0; torch.save(state, checkpoint_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(training["patience"]): break
    writer.close()
    (args.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    best_state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    evaluation_splits = ("validation", "test") if config.get("evaluate_test", False) else ("validation",)
    for split in evaluation_splits:
        predictions, truth = predict(model, loaders[split], device)
        report = metrics_from_predictions(predictions, truth)
        (args.run_dir / f"{split}_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if split == "validation":
            esol_entities = np.asarray([entity for entity, task, _ in records[split] if task == "esol"], dtype=np.int64)
            np.savez_compressed(args.run_dir / "validation_esol_predictions.npz",
                                entity_indices=esol_entities,
                                targets=np.asarray(truth.get("esol", []), dtype=np.float32),
                                predictions=np.asarray(predictions.get("esol", []), dtype=np.float32))
    print(json.dumps({"best_epoch": best_state["epoch"], "best_objective": best_state["validation_objective"],
                      "test_evaluated": config.get("evaluate_test", False)}), flush=True)


if __name__ == "__main__":
    main()
