"""Replay a GB1 distillation objective on an immutable queried-node manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pls.editflow.hamming import hamming_distance, queried_nodes_sha256
from pls.training.train_editflow_gb1 import evaluation_edges
from pls.training.train_editflow_gb1_active import evaluate, fit_ensemble


def load_queried_manifest(
    path: Path,
    measured: np.ndarray,
    expected_budget: int,
) -> tuple[np.ndarray, dict]:
    """Load and validate an exact, value-free queried-node manifest."""
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "PLS_EditFlow_queried_nodes_v1":
        raise ValueError("unsupported queried-node manifest schema")
    if manifest.get("oracle_values_included") is not False:
        raise ValueError("queried-node manifests must not embed oracle values")
    nodes = np.asarray(manifest.get("node_indices", ()), dtype=np.int64)
    if nodes.ndim != 1 or len(nodes) != expected_budget:
        raise ValueError("queried-node manifest has the wrong budget")
    if len(np.unique(nodes)) != len(nodes):
        raise ValueError("queried-node manifest contains duplicates")
    if nodes.size and (nodes.min() < 0 or nodes.max() >= len(measured)):
        raise ValueError("queried-node manifest references a missing node")
    if not np.all(measured[nodes]):
        raise ValueError("queried-node manifest references an imputed node")
    identity = queried_nodes_sha256(nodes)
    if manifest.get("sha256") != identity:
        raise ValueError("queried-node manifest SHA-256 mismatch")
    return nodes, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    data_config = config["data"]
    training = config["training"]
    if config.get("evaluate_test", False):
        parser.error("test evaluation is permanently disabled")
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(training["hip_device"]):
        parser.error("HIP device mismatch")

    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0")

    landscape = np.load(data_config["landscape"])
    raw_tokens = landscape["tokens"].astype(np.int64)
    tokens = torch.from_numpy(raw_tokens + 1)
    fitness = landscape["fitness"].astype(np.float64)
    measured = landscape["is_measured"].astype(bool)
    queried, source_manifest = load_queried_manifest(
        Path(data_config["queried_nodes_manifest"]),
        measured,
        int(data_config["query_budget"]),
    )

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    wild_tokens = np.asarray([alphabet.index(value) for value in "VDGV"])
    distances = hamming_distance(raw_tokens, wild_tokens)
    field_edges, edge_groups = evaluation_edges(
        raw_tokens,
        measured,
        int(data_config["evaluation_anchors"]),
        data_config["evaluation_salt"],
    )
    ensemble, states, training_summary, closed_edges = fit_ensemble(
        tokens,
        fitness,
        set(map(int, queried)),
        config["model"],
        training,
        device,
    )
    value_metrics, edge_metrics, regret_metrics = evaluate(
        ensemble,
        fitness,
        measured,
        distances,
        field_edges,
        edge_groups,
        data_config["edit_radii"],
        int(data_config.get("top_k", 10)),
    )

    budget = int(data_config["query_budget"])
    writer = SummaryWriter(arguments.run_dir / "tensorboard")
    writer.add_scalar("replay/r2", value_metrics["r2"], budget)
    writer.add_scalar("replay/edge_spearman", edge_metrics["edge_spearman"], budget)
    writer.add_scalar("replay/regret_radius_4", regret_metrics["4"]["regret"], budget)
    writer.close()
    torch.save(
        {"members": states, "config": config, "queried_nodes": queried.tolist()},
        arguments.run_dir / "checkpoints" / "best.pt",
    )

    query_budget = {
        "unique_queried_nodes": budget,
        "closed_edges": closed_edges,
        "queried_nodes_sha256": source_manifest["sha256"],
        "source_manifest": data_config["queried_nodes_manifest"],
        "teacher_query_cost_unit": "unique measured node",
        "same_node_budget_required": True,
        "test_evaluated": False,
    }
    history = {
        "training": training_summary,
        "value": value_metrics,
        "edge": edge_metrics,
        "regret": regret_metrics,
    }
    artifacts = {
        "history.json": history,
        "value_metrics.json": value_metrics,
        "edge_metrics.json": edge_metrics,
        "ranking_metrics.json": {
            key: value
            for key, value in edge_metrics.items()
            if "kendall" in key or "recall" in key or "sign" in key
        },
        "regret_metrics.json": regret_metrics,
        "query_budget.json": query_budget,
        "queried_nodes.json": source_manifest,
    }
    for name, value in artifacts.items():
        (arguments.run_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                "value": value_metrics,
                "edge": edge_metrics,
                "regret": regret_metrics,
                "query_budget": query_budget,
                "test_evaluated": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
