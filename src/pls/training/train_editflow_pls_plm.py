"""Train frozen-PLM potential and direct intervention students for PLS EditFlow."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from pls.editflow.mutations import AMINO_ACIDS
from pls.editflow.plm_student import (PLMMutationDeltaHead, PLMPairDeltaHead, PLMPotentialHead,
                                      commuting_cycle_residual)
from pls.training.train_editflow_pls import load_landscape, validation_metrics


AMINO_ACID_INDEX = {value: index for index, value in enumerate(AMINO_ACIDS)}


def load_plm_features(global_path: Path, residue_root: Path, offsets_path: Path) -> dict:
    global_embedding = np.load(global_path, mmap_mode="r")
    offsets = np.load(offsets_path, mmap_mode="r")
    metadata = json.loads((residue_root / "pca_metadata.json").read_text())
    shape = tuple(map(int, metadata["shape"]))
    residue = np.memmap(
        residue_root / "residue_esm2_pca.f16", mode="r", dtype=np.float16, shape=shape
    )
    if global_embedding.ndim != 2 or offsets.shape != (global_embedding.shape[0] + 1,):
        raise ValueError("PLM node features and offsets are inconsistent")
    if int(offsets[-1]) != shape[0]:
        raise ValueError("residue PLM shape does not match offsets")
    pooled = np.stack([
        np.asarray(residue[int(offsets[i]):int(offsets[i + 1])], dtype=np.float32).mean(0)
        for i in range(global_embedding.shape[0])
    ])
    return {
        "global": np.asarray(global_embedding, dtype=np.float32),
        "residue": np.asarray(residue, dtype=np.float32),
        "pooled": pooled,
        "offsets": np.asarray(offsets, dtype=np.int64),
    }


def edge_records(landscape: dict, split: str) -> dict:
    records = []
    for edge_index, (source, target) in enumerate(landscape["edges"].T):
        source = int(source); target = int(target)
        node = landscape["nodes"][target]
        if node["split"] != split:
            continue
        mutation = node["mutation"]
        if landscape["nodes"][source]["kind"] != "anchor":
            raise ValueError("PLS PoC edges must originate from anchors")
        records.append((
            edge_index,
            source,
            target,
            int(mutation["position_zero_based"]),
            AMINO_ACID_INDEX[mutation["source_residue"]],
            AMINO_ACID_INDEX[mutation["target_residue"]],
            int(node["anchor_rank"]),
        ))
    columns = tuple(zip(*records))
    return {
        name: np.asarray(values, dtype=np.int64)
        for name, values in zip(
            ("edge_index", "source", "target", "position", "source_aa", "target_aa", "anchor_rank"),
            columns,
        )
    }


def commuting_cycles(records: dict) -> np.ndarray:
    """Pairs of labelled anchor edits that define an unlabelled commuting square."""
    pairs = []
    for anchor in np.unique(records["anchor_rank"]):
        selected = np.flatnonzero(records["anchor_rank"] == anchor)
        for offset, left in enumerate(selected):
            for right in selected[offset + 1:]:
                if records["position"][left] != records["position"][right]:
                    pairs.append((int(left), int(right)))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def delta_inputs(model, features: dict, parents, positions, source_aa, target_aa, device):
    parents = np.asarray(parents, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    lengths = features["offsets"][parents + 1] - features["offsets"][parents]
    if np.any(positions < 0) or np.any(positions >= lengths):
        raise ValueError("mutation position is outside its parent sequence")
    local_indices = features["offsets"][parents] + positions
    normalized_position = positions / np.maximum(lengths - 1, 1)
    normalized_log_length = np.log1p(lengths) / math.log1p(2048)
    return model(
        torch.from_numpy(features["global"][parents]).to(device),
        torch.from_numpy(features["pooled"][parents]).to(device),
        torch.from_numpy(features["residue"][local_indices]).to(device),
        torch.from_numpy(np.asarray(source_aa, dtype=np.int64)).to(device),
        torch.from_numpy(np.asarray(target_aa, dtype=np.int64)).to(device),
        torch.from_numpy(normalized_position.astype(np.float32)).to(device),
        torch.from_numpy(normalized_log_length.astype(np.float32)).to(device),
    )


def pair_delta_inputs(model, features: dict, parents, targets, positions, source_aa, target_aa, device):
    parents = np.asarray(parents, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    lengths = features["offsets"][parents + 1] - features["offsets"][parents]
    target_lengths = features["offsets"][targets + 1] - features["offsets"][targets]
    if np.any(lengths != target_lengths) or np.any(positions < 0) or np.any(positions >= lengths):
        raise ValueError("pair edit features require equal valid parent/target coordinates")
    parent_local_indices = features["offsets"][parents] + positions
    target_local_indices = features["offsets"][targets] + positions
    normalized_position = positions / np.maximum(lengths - 1, 1)
    normalized_log_length = np.log1p(lengths) / math.log1p(2048)
    return model(
        torch.from_numpy(features["global"][parents]).to(device),
        torch.from_numpy(features["global"][targets]).to(device),
        torch.from_numpy(features["pooled"][parents]).to(device),
        torch.from_numpy(features["pooled"][targets]).to(device),
        torch.from_numpy(features["residue"][parent_local_indices]).to(device),
        torch.from_numpy(features["residue"][target_local_indices]).to(device),
        torch.from_numpy(np.asarray(source_aa, dtype=np.int64)).to(device),
        torch.from_numpy(np.asarray(target_aa, dtype=np.int64)).to(device),
        torch.from_numpy(normalized_position.astype(np.float32)).to(device),
        torch.from_numpy(normalized_log_length.astype(np.float32)).to(device),
    )


def anchored_delta_predictions(
    model, landscape: dict, records: dict, features: dict,
    delta_mean: float, delta_std: float, device, *, pair_mode: bool = False,
) -> np.ndarray:
    prediction = np.full(len(landscape["nodes"]), np.nan, dtype=np.float64)
    with torch.inference_mode():
        if pair_mode:
            normalized = pair_delta_inputs(
                model, features, records["source"], records["target"], records["position"],
                records["source_aa"], records["target_aa"], device,
            )
        else:
            normalized = delta_inputs(
                model, features, records["source"], records["position"],
                records["source_aa"], records["target_aa"], device,
            )
    effects = normalized.float().cpu().numpy() * delta_std + delta_mean
    for source, target, effect in zip(records["source"], records["target"], effects):
        prediction[int(source)] = landscape["teacher"][int(source)]
        prediction[int(target)] = landscape["teacher"][int(source)] + float(effect)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    if config.get("evaluate_test", False):
        parser.error("test evaluation is permanently disabled")
    training = config["training"]
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(training["hip_device"]):
        parser.error("HIP device mismatch")
    seed = int(training["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda:0")
    data = config["data"]
    landscape = load_landscape(
        Path(data["manifest"]), Path(data["oracle_scores"]), Path(data["oracle_report"])
    )
    features = load_plm_features(
        Path(data["global_embeddings"]), Path(data["residue_embeddings"]),
        Path(data["offsets"]),
    )
    if features["global"].shape[0] != len(landscape["nodes"]):
        raise ValueError("PLM features do not cover the oracle nodes exactly")
    train_edges = edge_records(landscape, "train")
    validation_edges = edge_records(landscape, "validation")
    cycles = commuting_cycles(train_edges)
    teacher_delta = (
        landscape["teacher"][train_edges["target"]]
        - landscape["teacher"][train_edges["source"]]
    )
    delta_mean = float(teacher_delta.mean())
    delta_std = max(float(teacher_delta.std()), 1e-6)
    train_nodes = np.asarray([
        index for index, node in enumerate(landscape["nodes"]) if node["split"] == "train"
    ], dtype=np.int64)
    value_mean = float(landscape["teacher"][train_nodes].mean())
    value_std = max(float(landscape["teacher"][train_nodes].std()), 1e-6)
    mode = str(config["model"]["mode"])
    dimension = int(config["model"]["dimension"])
    dropout = float(config["model"]["dropout"])
    if mode == "potential":
        model = PLMPotentialHead(
            features["global"].shape[1], features["pooled"].shape[1], dimension, dropout
        ).to(device)
    elif mode in {"direct_delta", "cycle_delta"}:
        model = PLMMutationDeltaHead(
            features["global"].shape[1], features["pooled"].shape[1], dimension, dropout
        ).to(device)
    elif mode == "pair_delta":
        model = PLMPairDeltaHead(
            features["global"].shape[1], features["pooled"].shape[1], dimension, dropout
        ).to(device)
    else:
        raise ValueError("model mode must be potential, direct_delta, cycle_delta, or pair_delta")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"], fused=training.get("fused_optimizer", False),
    )
    writer = SummaryWriter(arguments.run_dir / "tensorboard")
    best = float("inf"); stale = 0; history = []
    global_tensor = torch.from_numpy(features["global"]).to(device)
    pooled_tensor = torch.from_numpy(features["pooled"]).to(device)
    normalized_values = torch.from_numpy(
        ((landscape["teacher"] - value_mean) / value_std).astype(np.float32)
    ).to(device)
    normalized_delta = torch.from_numpy(
        ((teacher_delta - delta_mean) / delta_std).astype(np.float32)
    ).to(device)
    cycle_weight = float(training.get("cycle_weight", 0.0))
    if mode != "cycle_delta" and cycle_weight != 0.0:
        raise ValueError("cycle_weight is only valid for cycle_delta")
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=training.get("amp_bfloat16", True)):
            if mode == "potential":
                prediction = model(global_tensor[train_nodes], pooled_tensor[train_nodes])
                supervised = F.huber_loss(prediction, normalized_values[train_nodes])
                cycle_loss = prediction.new_zeros(())
            else:
                if mode == "pair_delta":
                    prediction = pair_delta_inputs(
                        model, features, train_edges["source"], train_edges["target"],
                        train_edges["position"], train_edges["source_aa"],
                        train_edges["target_aa"], device,
                    )
                else:
                    prediction = delta_inputs(
                        model, features, train_edges["source"], train_edges["position"],
                        train_edges["source_aa"], train_edges["target_aa"], device,
                    )
                supervised = F.huber_loss(prediction, normalized_delta)
                cycle_loss = prediction.new_zeros(())
                if mode == "cycle_delta" and len(cycles):
                    left = cycles[:, 0]; right = cycles[:, 1]
                    after_left = delta_inputs(
                        model, features, train_edges["target"][left], train_edges["position"][right],
                        train_edges["source_aa"][right], train_edges["target_aa"][right], device,
                    )
                    after_right = delta_inputs(
                        model, features, train_edges["target"][right], train_edges["position"][left],
                        train_edges["source_aa"][left], train_edges["target_aa"][left], device,
                    )
                    residual = commuting_cycle_residual(
                        prediction[left], after_left, prediction[right], after_right
                    )
                    cycle_loss = residual.square().mean()
            objective = supervised + cycle_weight * cycle_loss
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_gradient_norm", 5.0)))
        optimizer.step()
        model.eval()
        if mode == "potential":
            with torch.inference_mode():
                normalized = model(global_tensor, pooled_tensor)
            node_prediction = normalized.float().cpu().numpy() * value_std + value_mean
            metrics = validation_metrics(landscape, node_prediction, int(data.get("top_k", 5)))
            metrics["value"]["anchored"] = False
        else:
            node_prediction = anchored_delta_predictions(
                model, landscape, validation_edges, features, delta_mean, delta_std, device,
                pair_mode=mode == "pair_delta",
            )
            metrics = validation_metrics(landscape, node_prediction, int(data.get("top_k", 5)))
            metrics["value"]["anchored"] = True
            metrics["value"]["anchor_definition"] = "one queried teacher value per validation landscape"
        selection = float(metrics["edge"]["edge_rmse"])
        row = {
            "epoch": epoch,
            "supervised_loss": float(supervised.detach()),
            "cycle_loss": float(cycle_loss.detach()),
            "objective": float(objective.detach()),
            "validation": metrics,
        }
        history.append(row); print(json.dumps(row), flush=True)
        writer.add_scalar("training/supervised", row["supervised_loss"], epoch)
        writer.add_scalar("training/cycle", row["cycle_loss"], epoch)
        writer.add_scalar("validation/edge_rmse", selection, epoch)
        writer.add_scalar("validation/edge_spearman", metrics["edge"]["edge_spearman"], epoch)
        state = {
            "model": model.state_dict(), "epoch": epoch, "validation": metrics,
            "value_mean": value_mean, "value_std": value_std,
            "delta_mean": delta_mean, "delta_std": delta_std, "config": config,
        }
        if epoch % int(training["checkpoint_every"]) == 0:
            torch.save(state, arguments.run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if selection < best:
            best = selection; stale = 0
            torch.save(state, arguments.run_dir / "checkpoints" / "best.pt")
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    writer.close()
    (arguments.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    state = torch.load(arguments.run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"]); model.eval()
    if mode == "potential":
        with torch.inference_mode(): normalized = model(global_tensor, pooled_tensor)
        final_prediction = normalized.float().cpu().numpy() * value_std + value_mean
    else:
        final_prediction = anchored_delta_predictions(
            model, landscape, validation_edges, features, delta_mean, delta_std, device,
            pair_mode=mode == "pair_delta",
        )
    metrics = validation_metrics(landscape, final_prediction, int(data.get("top_k", 5)))
    metrics["value"]["anchored"] = mode != "potential"
    if mode != "potential":
        metrics["value"]["anchor_definition"] = "one queried teacher value per validation landscape"
    metrics["protocol"] = {
        "mode": mode,
        "selection_metric": "validation_edge_rmse",
        "train_oracle_nodes": int(len(train_nodes)),
        "train_oracle_edges": int(len(train_edges["source"])),
        "validation_oracle_nodes": int(sum(node["split"] == "validation" for node in landscape["nodes"])),
        "validation_oracle_edges": int(len(validation_edges["source"])),
        "unlabelled_commuting_cycles": int(len(cycles)) if mode == "cycle_delta" else 0,
        "additional_teacher_queries_for_cycles": 0,
        "test_evaluated": False,
    }
    (arguments.run_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (arguments.run_dir / "query_budget.json").write_text(
        json.dumps(metrics["protocol"], indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(arguments.run_dir / "validation_predictions.npz", predictions=final_prediction)
    print(json.dumps({"best_epoch": state["epoch"], "validation": metrics}))


if __name__ == "__main__":
    main()
