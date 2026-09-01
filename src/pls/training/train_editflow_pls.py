"""Train a sequence-only student on safe PLS oracle values and edit effects."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter

from pls.editflow.metrics import mutation_field_metrics
from pls.editflow.objective import editflow_distillation_loss
from pls.editflow.student import EditPotentialStudent, encode_sequences


def load_landscape(manifest_path: Path, scores_path: Path, report_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    if manifest.get("test_evaluated") is not False or report.get("test_evaluated") is not False:
        raise ValueError("PLS student refuses inputs without a test-free assertion")
    if report.get("output") != "raw_logit":
        raise ValueError("PLS student requires raw teacher logits")
    nodes = manifest["nodes"]
    if any(node["split"] not in {"train", "validation"} for node in nodes):
        raise ValueError("PLS student manifest contains a forbidden split")
    scores = np.load(scores_path)
    indices = scores["node_indices"].astype(np.int64)
    logits = scores["logits"].astype(np.float64)
    hashes = scores["sequence_sha256"].astype(str)
    if not np.array_equal(indices, np.arange(len(nodes), dtype=np.int64)):
        raise ValueError("oracle score node order is not canonical")
    if logits.shape != (len(nodes),) or not np.isfinite(logits).all():
        raise ValueError("oracle logits are incomplete")
    if hashes.tolist() != [node["sequence_sha256"] for node in nodes]:
        raise ValueError("oracle score sequence identity mismatch")
    groups: dict[str, list[int]] = {"train": [], "validation": []}
    grouped_nodes: dict[int, list[int]] = {}
    for node in nodes:
        grouped_nodes.setdefault(int(node["anchor_rank"]), []).append(int(node["node_index"]))
    for anchor_rank, node_indices in sorted(grouped_nodes.items()):
        splits = {nodes[index]["split"] for index in node_indices}
        if len(splits) != 1:
            raise ValueError("an anchor landscape crosses strict splits")
        split = splits.pop()
        anchor_count = sum(nodes[index]["kind"] == "anchor" for index in node_indices)
        if anchor_count != 1:
            raise ValueError("each landscape must contain exactly one anchor")
        groups[split].append(anchor_rank)
    edges = np.asarray(
        [[row["source_node"] for row in manifest["edges"]],
         [row["target_node"] for row in manifest["edges"]]],
        dtype=np.int64,
    )
    return {
        "nodes": nodes,
        "sequences": [node["sequence"] for node in nodes],
        "teacher": logits,
        "groups": groups,
        "grouped_nodes": grouped_nodes,
        "edges": edges,
    }


def group_batch(landscape: dict, anchor_ranks: list[int], teacher_mean: float, teacher_std: float, device):
    node_indices = [
        node
        for anchor_rank in anchor_ranks
        for node in landscape["grouped_nodes"][anchor_rank]
    ]
    local = {node: index for index, node in enumerate(node_indices)}
    selected_edges = [
        (local[int(source)], local[int(target)])
        for source, target in landscape["edges"].T
        if int(source) in local and int(target) in local
    ]
    edge_index = torch.tensor(selected_edges, dtype=torch.long, device=device).T
    tokens, mask = encode_sequences([landscape["sequences"][node] for node in node_indices])
    teacher = torch.tensor(
        (landscape["teacher"][node_indices] - teacher_mean) / teacher_std,
        dtype=torch.float32,
        device=device,
    )
    return (
        np.asarray(node_indices, dtype=np.int64),
        tokens.to(device),
        mask.to(device),
        teacher,
        edge_index,
    )


def predict_groups(model, landscape, anchor_ranks, teacher_mean, teacher_std, device, anchor_batch_size, amp):
    prediction = np.full(len(landscape["nodes"]), np.nan, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(anchor_ranks), anchor_batch_size):
            selected = anchor_ranks[start:start + anchor_batch_size]
            node_indices, tokens, mask, _, _ = group_batch(
                landscape, selected, teacher_mean, teacher_std, device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                normalized = model(tokens, mask)
            prediction[node_indices] = normalized.float().cpu().numpy() * teacher_std + teacher_mean
    return prediction


def validation_metrics(landscape, prediction, top_k):
    validation_nodes = np.asarray([
        index for index, node in enumerate(landscape["nodes"])
        if node["split"] == "validation"
    ], dtype=np.int64)
    local = {node: index for index, node in enumerate(validation_nodes)}
    edges = np.asarray([
        (local[int(source)], local[int(target)])
        for source, target in landscape["edges"].T
        if int(source) in local and int(target) in local
    ], dtype=np.int64).T
    edge_groups = np.asarray([
        landscape["nodes"][validation_nodes[int(source)]]["anchor_rank"]
        for source in edges[0]
    ], dtype=np.int64)
    teacher = landscape["teacher"][validation_nodes]
    student = prediction[validation_nodes]
    value = {
        "nodes": int(len(validation_nodes)),
        "r2": float(r2_score(teacher, student)),
        "pearson": float(pearsonr(teacher, student).statistic),
        "rmse": float(np.sqrt(np.mean(np.square(student - teacher)))),
    }
    field = mutation_field_metrics(
        teacher, student, edges, edge_groups, top_k=top_k
    )
    return {"value": value, "edge": field}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    if config.get("evaluate_test", False):
        parser.error("test evaluation is permanently disabled")
    training = config["training"]
    accelerator_backend = str(training.get("accelerator_backend", "rocm"))
    if accelerator_backend == "rocm":
        if os.environ.get("HIP_VISIBLE_DEVICES") != str(training["hip_device"]):
            parser.error("HIP device mismatch")
    elif accelerator_backend == "cuda_slurm":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not os.environ.get("SLURM_JOB_ID") or not visible or "," in visible:
            parser.error("cuda_slurm requires one Slurm-assigned visible GPU")
    else:
        parser.error("accelerator_backend must be rocm or cuda_slurm")
    seed = int(training["seed"])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    data = config["data"]
    landscape = load_landscape(
        Path(data["manifest"]), Path(data["oracle_scores"]), Path(data["oracle_report"])
    )
    train_nodes = np.asarray([
        index for index, node in enumerate(landscape["nodes"])
        if node["split"] == "train"
    ], dtype=np.int64)
    teacher_mean = float(landscape["teacher"][train_nodes].mean())
    teacher_std = max(float(landscape["teacher"][train_nodes].std()), 1e-6)
    device = torch.device("cuda:0")
    model_config = config["model"]
    model = EditPotentialStudent(
        model_config["dimension"], model_config["layers"], model_config["heads"],
        model_config["dropout"], model_config["max_length"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        fused=training.get("fused_optimizer", False),
    )
    writer = SummaryWriter(arguments.run_dir / "tensorboard")
    best = float("inf");stale = 0;history = []
    train_groups = list(landscape["groups"]["train"])
    validation_groups = list(landscape["groups"]["validation"])
    anchor_batch_size = int(training["anchor_batch_size"])
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train();rng = random.Random(seed + epoch);rng.shuffle(train_groups);total = 0.0;batches = 0
        for start in range(0, len(train_groups), anchor_batch_size):
            selected = train_groups[start:start + anchor_batch_size]
            _, tokens, mask, teacher, edge_index = group_batch(
                landscape, selected, teacher_mean, teacher_std, device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=training.get("amp_bfloat16", True)):
                values = model(tokens, mask)
                report = editflow_distillation_loss(
                    values,
                    teacher,
                    edge_index,
                    torch.ones_like(teacher, dtype=torch.bool),
                    value_weight=float(training["value_weight"]),
                    edge_weight=float(training["edge_weight"]),
                    loss=training.get("loss", "huber"),
                )
            report.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_gradient_norm", 5.0)))
            optimizer.step();total += float(report.total.detach());batches += 1
        prediction = predict_groups(
            model, landscape, validation_groups, teacher_mean, teacher_std, device,
            anchor_batch_size, bool(training.get("amp_bfloat16", True)),
        )
        metrics = validation_metrics(landscape, prediction, int(data.get("top_k", 5)))
        row = {"epoch": epoch, "train_objective": total / batches, "validation": metrics}
        history.append(row);print(json.dumps(row), flush=True)
        writer.add_scalar("validation/edge_rmse", metrics["edge"]["edge_rmse"], epoch)
        writer.add_scalar("validation/mutation_sign_balanced_accuracy", metrics["edge"]["mutation_sign_balanced_accuracy"], epoch)
        state = {
            "model": model.state_dict(), "epoch": epoch, "validation": metrics,
            "teacher_mean": teacher_mean, "teacher_std": teacher_std, "config": config,
        }
        if epoch % int(training["checkpoint_every"]) == 0:
            torch.save(state, arguments.run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        if metrics["edge"]["edge_rmse"] < best:
            best = metrics["edge"]["edge_rmse"];stale = 0
            torch.save(state, arguments.run_dir / "checkpoints" / "best.pt")
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    writer.close()
    (arguments.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    state = torch.load(arguments.run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    prediction = predict_groups(
        model, landscape, validation_groups, teacher_mean, teacher_std, device,
        anchor_batch_size, bool(training.get("amp_bfloat16", True)),
    )
    metrics = validation_metrics(landscape, prediction, int(data.get("top_k", 5)))
    (arguments.run_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(arguments.run_dir / "validation_predictions.npz", predictions=prediction)
    query_budget = {
        "train_oracle_nodes": int(len(train_nodes)),
        "validation_oracle_nodes": int(sum(node["split"] == "validation" for node in landscape["nodes"])),
        "teacher_query_cost_unit": "unique exact sequence",
        "test_evaluated": False,
    }
    (arguments.run_dir / "query_budget.json").write_text(
        json.dumps(query_budget, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"best_epoch": state["epoch"], "validation": metrics, **query_budget}))


if __name__ == "__main__":
    main()
