"""Re-evaluate final GB1 checkpoints with non-confounded design regrets.

This migration utility is intentionally final-budget only: historical
confirmatory runs retained final ensemble checkpoints but not every intermediate
ensemble.  It never changes the original run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pls.editflow.graph import exact_design_regrets
from pls.editflow.hamming import hamming_distance, queried_nodes_sha256
from pls.editflow.student import EditPotentialStudent
from pls.training.train_editflow_gb1 import batched_predict
from pls.training.train_editflow_gb1_confirmatory import summarize


def load_safe_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    manifests = json.loads((run_dir / "queried_nodes.json").read_text())
    query_budget = json.loads((run_dir / "query_budget.json").read_text())
    if config.get("evaluate_test", False):
        raise ValueError("audit refuses runs with evaluate_test=true")
    if query_budget.get("test_evaluated") is not False:
        raise ValueError("run does not explicitly preserve the test freeze")
    if manifests.get("oracle_values_included") is not False:
        raise ValueError("queried-node manifest unexpectedly embeds oracle values")
    return config, history, manifests


def ensemble_prediction(
    checkpoint: dict,
    model_config: dict,
    tokens: torch.Tensor,
    fitness: np.ndarray,
    queried: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    target_mean = float(fitness[queried].mean())
    target_std = max(float(fitness[queried].std()), 1e-6)
    predictions = []
    for state in checkpoint["members"]:
        model = EditPotentialStudent(
            model_config["dimension"],
            model_config["layers"],
            model_config["heads"],
            model_config["dropout"],
            4,
        ).to(device)
        model.load_state_dict(state)
        normalized = batched_predict(model, tokens, device, batch_size)
        predictions.append(normalized * target_std + target_mean)
    return np.asarray(predictions).mean(0)


def audit_run(run_dir: Path, device_name: str) -> dict:
    config, history, manifest_file = load_safe_run(run_dir)
    data_config = config["data"]
    landscape = np.load(data_config["landscape"])
    raw_tokens = landscape["tokens"].astype(np.int64)
    tokens = torch.from_numpy(raw_tokens + 1)
    fitness = landscape["fitness"].astype(np.float64)
    measured = landscape["is_measured"].astype(bool)
    radii = list(map(int, data_config["edit_radii"]))
    device = torch.device(device_name)
    manifests = manifest_file.get("manifests", [])
    if len(history) != len(manifests):
        raise ValueError("history and queried-node manifest anchor counts differ")

    rows = []
    for result, manifest in zip(history, manifests):
        rank = int(result["anchor"]["rank"])
        if rank != int(manifest["anchor_rank"]):
            raise ValueError("history and queried-node manifest ordering differ")
        queried = np.asarray(manifest["node_indices"], dtype=np.int64)
        if queried_nodes_sha256(queried) != manifest["sha256"]:
            raise ValueError("queried-node manifest SHA-256 mismatch")
        checkpoint_path = run_dir / "checkpoints" / f"anchor_{rank:02d}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if sorted(map(int, checkpoint["queried_nodes"])) != queried.tolist():
            raise ValueError("checkpoint and manifest queried nodes differ")
        prediction = ensemble_prediction(
            checkpoint,
            config["model"],
            tokens,
            fitness,
            queried,
            device,
            int(config["training"].get("inference_batch_size", 8192)),
        )
        anchor = int(result["anchor"]["node_index"])
        distances = hamming_distance(raw_tokens, raw_tokens[anchor])
        regret = {}
        for radius in radii:
            candidates = np.flatnonzero(measured & (distances <= radius))
            regret[str(radius)] = exact_design_regrets(
                fitness, prediction, candidates, queried
            )
        rows.append({"anchor": result["anchor"], "regret": regret})
        print(json.dumps({"anchor_rank": rank, "regret": {
            radius: {name: values[name]["regret"] for name in (
                "acquired", "novel_design", "campaign"
            )} for radius, values in regret.items()
        }}), flush=True)

    aggregate = {}
    for radius in radii:
        aggregate[str(radius)] = {}
        for component in ("acquired", "novel_design", "campaign"):
            values = [
                row["regret"][str(radius)][component]["regret"]
                for row in rows
                if row["regret"][str(radius)][component]["regret"] is not None
            ]
            aggregate[str(radius)][component] = {
                **summarize(values),
                "zero_regret_fraction": (
                    float(np.mean(np.asarray(values) <= 1e-12)) if values else None
                ),
                "unavailable_fraction": float(1.0 - len(values) / len(rows)),
            }
    return {
        "schema": "PLS_EditFlow_GB1_final_regret_decomposition_audit_v1",
        "source_run": str(run_dir),
        "final_budget": int(config["data"]["query_budgets"][-1]),
        "checkpoint_scope": "final budget only",
        "regret_definitions": {
            "acquired": "global feasible optimum minus best queried feasible node",
            "novel_design": "best unqueried feasible node minus teacher value of the student-selected unqueried node",
            "campaign": "global feasible optimum minus the better of best queried and student-selected unqueried node",
        },
        "anchors": rows,
        "aggregate": aggregate,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    result = audit_run(arguments.run_dir, arguments.device)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
