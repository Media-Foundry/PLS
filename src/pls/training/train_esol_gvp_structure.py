"""Leakage-safe eSOL regression with equivariant GVP residue geometry."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pls.evaluation.metrics import regression_metrics
from pls.models.gvp_structure import GVPStructureFusion
from pls.training.train_esol_structure import LengthSampler, rank_loss
from pls.training.train_gvp_structure import GVPData, collate, infer

SOURCE = "eSOL_FGNNSol"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data, model_config, training = config["data"], config["model"], config["training"]
    if config.get("evaluate_test", False):
        parser.error("test evaluation is permanently disabled")
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(training["hip_device"]):
        parser.error("HIP device mismatch")
    seed = training["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    with open(data["entities"], newline="", encoding="utf-8") as handle:
        entities = list(csv.DictReader(handle))
    entity_index = {row["sequence_sha256"]: index for index, row in enumerate(entities)}
    rows = {split: [] for split in ("train", "validation")}
    with open(data["observation_split"], newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["source_dataset"] == SOURCE and row["split"] in rows:
                rows[row["split"]].append((entity_index[row["sequence_sha256"]], row["sequence_sha256"], float(row["target_value"])))
    status = np.load(data["structure_status"], mmap_mode="r")
    rows = {split: [value for value in values if status[value[0]] == 1] for split, values in rows.items()}
    esm = np.load(Path(data["embedding_dir"]) / "embeddings.npy", mmap_mode="r")
    stats = json.loads(Path(data["structure_stats"]).read_text())
    mean, std = torch.tensor(stats["scalar_means"]), torch.tensor(stats["scalar_stds"])
    compact, geometry = Path(data["compact_structure_dir"]), Path(data["geometry_dir"])
    residue_esm, vector_dir = Path(data["residue_esm_dir"]), Path(data["vector_dir"])
    sets = {split: GVPData(values, esm, Path(data["structure_dir"]), mean, std, compact, geometry, False, model_config["neighbors"], residue_esm, None, None, None, None, vector_dir=vector_dir) for split, values in rows.items()}
    lengths = [int(entities[value[0]]["length"]) for value in rows["train"]]
    sampler = LengthSampler(lengths, training["batch_size"], seed)
    train = DataLoader(sets["train"], batch_sampler=sampler, num_workers=training["workers"], collate_fn=collate, persistent_workers=training["workers"] > 0, pin_memory=True)
    validation = DataLoader(sets["validation"], batch_size=training["batch_size"], num_workers=training["workers"], collate_fn=collate, persistent_workers=training["workers"] > 0, pin_memory=True)

    residue_sequence_dimension = int(json.loads((residue_esm / "pca_metadata.json").read_text())["shape"][1])
    residue_dimension = 152 + residue_sequence_dimension
    device = torch.device("cuda:0")
    model = GVPStructureFusion(esm.shape[1], residue_dimension, model_config["scalar_dimension"], model_config["vector_dimension"], model_config["representation_dimension"], model_config["dropout"], model_config["layers"], model_config.get("fusion", "interaction"), residue_sequence_dimension, stats["scalar_means"][1], stats["scalar_stds"][1]).to(device)
    decay = float(training.get("ema_decay", 0))
    ema = copy.deepcopy(model).eval().requires_grad_(False) if decay else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"], fused=training.get("fused_optimizer", False))
    writer = SummaryWriter(args.run_dir / "tensorboard")
    best, stale, history = -2.0, 0, []
    for epoch in range(1, training["epochs"] + 1):
        started = time.monotonic(); model.train(); total = count = 0
        for sequence, residue, vectors, coordinates, mask, neighbors, distances, patch, components, target in train:
            values = [value.to(device, non_blocking=True) for value in (sequence, residue, vectors, coordinates, mask, neighbors, distances, patch, components, target)]
            sequence, residue, vectors, coordinates, mask, neighbors, distances, patch, components, target = values
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=training.get("amp_bfloat16", True)):
                prediction = model(sequence, residue, vectors, coordinates, mask, neighbors, distances, patch, components)
                loss = nn.functional.smooth_l1_loss(prediction, target, beta=.1) + float(training.get("rank_weight", 0)) * rank_loss(prediction, target)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite GVP regression loss at epoch {epoch}")
            loss.backward(); gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_gradient_norm", 5.0)))
            if not torch.isfinite(gradient_norm): raise FloatingPointError(f"non-finite GVP regression gradient norm at epoch {epoch}")
            optimizer.step()
            if ema:
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()): ema_parameter.mul_(decay).add_(parameter, alpha=1-decay)
            total += loss.item() * len(target); count += len(target)
        seconds = time.monotonic() - started; evaluated = ema or model
        truth, prediction = infer(evaluated, validation, device, training.get("amp_bfloat16", False)); metrics = regression_metrics(truth, prediction)
        row = {"epoch": epoch, "train_loss": total / count, "train_seconds": seconds, "train_samples_per_second": count / seconds, "validation": metrics}; history.append(row); print(json.dumps(row), flush=True)
        writer.add_scalar("validation/spearman", metrics["spearman"], epoch); writer.add_scalar("throughput/train_samples_per_second", row["train_samples_per_second"], epoch)
        state = {"model": evaluated.state_dict(), "epoch": epoch, "validation": metrics, "config": config}
        if epoch % training["checkpoint_every"] == 0: torch.save(state, args.run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if metrics["spearman"] > best: best, stale = metrics["spearman"], 0; torch.save(state, args.run_dir / "checkpoints" / "best.pt")
        else: stale += 1
        if stale >= training["patience"]: break
    writer.close(); (args.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    state = torch.load(args.run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False); model.load_state_dict(state["model"])
    truth, prediction = infer(model, validation, device, training.get("amp_bfloat16", False)); metrics = regression_metrics(truth, prediction)
    (args.run_dir / "validation_metrics.json").write_text(json.dumps({"esol": metrics}, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.run_dir / "validation_predictions.npz", targets=truth, predictions=prediction, entity_indices=np.asarray([value[0] for value in rows["validation"]], np.int64))
    print(json.dumps({"best_epoch": state["epoch"], "best_validation_spearman": best, "test_evaluated": False, "validation_structures": len(rows["validation"])}))


if __name__ == "__main__":
    main()
