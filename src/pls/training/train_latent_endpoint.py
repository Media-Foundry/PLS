"""Joint latent-solubility training with monotonic assay observation heads."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from pls.evaluation.metrics import binary_metrics, regression_metrics
from pls.models.latent_endpoint import LatentEndpointModel


SOURCES = ("PDBSol_ProtSolM", "UESolDS_PLM_Sol_1.1", "eSOL_FGNNSol")
CONTINUOUS_ENDPOINT = 2


class ObservationDataset(Dataset):
    def __init__(self, rows, embeddings, descriptors):
        self.rows, self.embeddings, self.descriptors = rows, embeddings, descriptors

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        entity, endpoint, target = self.rows[index]
        features = np.concatenate((np.asarray(self.embeddings[entity], np.float32), self.descriptors[entity]))
        return torch.from_numpy(features), torch.tensor(endpoint), torch.tensor(target, dtype=torch.float32), torch.tensor(entity)


def source_entity_weights(rows):
    duplicates = Counter((entity, endpoint) for entity, endpoint, _ in rows)
    unique_by_source = Counter(endpoint for entity, endpoint in duplicates)
    return torch.tensor([1 / (len(SOURCES) * unique_by_source[endpoint] * duplicates[(entity, endpoint)]) for entity, endpoint, _ in rows], dtype=torch.double)


def infer(model, loader, device, amp=False):
    model.eval(); endpoints=[]; targets=[]; predictions=[]; entities=[]
    with torch.inference_mode():
        for features, endpoint, target, entity in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                prediction = model(features.to(device, non_blocking=True), endpoint.to(device, non_blocking=True))
            endpoints.extend(endpoint.tolist()); targets.extend(target.tolist()); predictions.extend(prediction.float().cpu().tolist()); entities.extend(entity.tolist())
    return np.asarray(endpoints), np.asarray(targets, np.float32), np.asarray(predictions, np.float32), np.asarray(entities, np.int64)


def endpoint_metrics(endpoints, targets, predictions):
    report = {}
    for endpoint, source in enumerate(SOURCES):
        selected = endpoints == endpoint
        report[source] = regression_metrics(targets[selected], predictions[selected]) if endpoint == CONTINUOUS_ENDPOINT else binary_metrics(targets[selected], predictions[selected])
    return report


def selection_score(metrics):
    return float((metrics[SOURCES[0]]["auroc"] + metrics[SOURCES[1]]["auroc"] + metrics[SOURCES[2]]["spearman"]) / 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); data, model_config, training = config["data"], config["model"], config["training"]
    if config.get("evaluate_test", False): parser.error("test evaluation is permanently disabled")
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(training["hip_device"]): parser.error("HIP device mismatch")
    seed = training["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with open(data["entities"], newline="", encoding="utf-8") as handle: entity_rows = list(csv.DictReader(handle))
    entity_index = {row["sequence_sha256"]: index for index, row in enumerate(entity_rows)}; source_index = {source: index for index, source in enumerate(SOURCES)}
    rows = {split: [] for split in ("train", "validation")}
    with open(data["observation_split"], newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] in rows and row["source_dataset"] in source_index:
                rows[row["split"]].append((entity_index[row["sequence_sha256"]], source_index[row["source_dataset"]], float(row["target_value"])))
    embeddings = np.load(Path(data["embedding_dir"]) / "embeddings.npy", mmap_mode="r")
    raw_descriptors = np.load(Path(data["sequence_descriptor_dir"]) / "descriptors.npy", mmap_mode="r")
    train_entities = np.unique([entity for entity, _, _ in rows["train"]]); mean = np.asarray(raw_descriptors[train_entities], np.float64).mean(0); std = np.maximum(np.asarray(raw_descriptors[train_entities], np.float64).std(0), 1e-6)
    descriptors = ((np.asarray(raw_descriptors, np.float32) - mean) / std).astype(np.float32)
    np.savez(args.run_dir / "descriptor_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32), train_entity_indices=train_entities)
    sets = {split: ObservationDataset(values, embeddings, descriptors) for split, values in rows.items()}
    weights = source_entity_weights(rows["train"]); sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(sets["train"], batch_size=training["batch_size"], sampler=sampler, num_workers=training["workers"], persistent_workers=training["workers"] > 0, pin_memory=True)
    validation_loader = DataLoader(sets["validation"], batch_size=training["batch_size"], num_workers=training["workers"], persistent_workers=training["workers"] > 0, pin_memory=True)
    device = torch.device("cuda:0"); model = LatentEndpointModel(embeddings.shape[1] + descriptors.shape[1], model_config["hidden_dimension"], model_config["dropout"], len(SOURCES)).to(device)
    decay = float(training.get("ema_decay", 0)); ema = copy.deepcopy(model).eval().requires_grad_(False) if decay else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"], fused=training.get("fused_optimizer", False)); writer = SummaryWriter(args.run_dir / "tensorboard")
    best, stale, history = -2., 0, []
    for epoch in range(1, training["epochs"] + 1):
        started = time.monotonic(); model.train(); total = count = 0
        for features, endpoint, target, _ in train_loader:
            features, endpoint, target = features.to(device, non_blocking=True), endpoint.to(device, non_blocking=True), target.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=training.get("amp_bfloat16", True)):
                prediction = model(features, endpoint); continuous = endpoint == CONTINUOUS_ENDPOINT; losses = torch.empty_like(target)
                losses[continuous] = nn.functional.smooth_l1_loss(prediction[continuous], target[continuous], beta=.1, reduction="none")
                losses[~continuous] = nn.functional.binary_cross_entropy_with_logits(prediction[~continuous], target[~continuous], reduction="none"); loss = losses.mean()
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite latent endpoint loss at epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_gradient_norm", 5.))); optimizer.step()
            if ema:
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()): ema_parameter.mul_(decay).add_(parameter, alpha=1-decay)
            total += loss.item() * len(target); count += len(target)
        evaluated = ema or model; endpoints, targets, predictions, _ = infer(evaluated, validation_loader, device, training.get("amp_bfloat16", False)); metrics = endpoint_metrics(endpoints, targets, predictions); score = selection_score(metrics); seconds = time.monotonic() - started
        row = {"epoch": epoch, "train_loss": total / count, "train_seconds": seconds, "train_samples_per_second": count / seconds, "selection_score": score, "validation": metrics, "positive_slopes": evaluated.slopes.detach().float().cpu().tolist()}; history.append(row); print(json.dumps(row), flush=True)
        for source, values in metrics.items(): writer.add_scalar(f"validation/{source}/primary", values["spearman" if source == SOURCES[2] else "auroc"], epoch)
        state = {"model": evaluated.state_dict(), "epoch": epoch, "selection_score": score, "validation": metrics, "config": config}
        if epoch % training["checkpoint_every"] == 0: torch.save(state, args.run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if score > best: best, stale = score, 0; torch.save(state, args.run_dir / "checkpoints" / "best.pt")
        else: stale += 1
        if stale >= training["patience"]: break
    writer.close(); (args.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    state = torch.load(args.run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False); model.load_state_dict(state["model"]); endpoints, targets, predictions, entities = infer(model, validation_loader, device, training.get("amp_bfloat16", False)); metrics = endpoint_metrics(endpoints, targets, predictions)
    (args.run_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n"); np.savez_compressed(args.run_dir / "validation_predictions.npz", endpoints=endpoints, targets=targets, predictions=predictions, entity_indices=entities)
    print(json.dumps({"best_epoch": state["epoch"], "best_selection_score": best, "test_evaluated": False, "validation": metrics}))


if __name__ == "__main__":
    main()
