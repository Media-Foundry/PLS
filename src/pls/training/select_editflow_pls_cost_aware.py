"""Apply frozen cost-aware quantiles to the fresh confirmatory components.

This runs before any confirmatory mutant is folded. It reads only the cached-parent
oracle and label-free covariates, then persists the selected candidate set and a
cost-balanced fold plan restricted to that set. Committing this output is the
preregistration that makes selected-stage compute a measured primary endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pls.editflow.cost_aware_gating import load_landscapes, select_candidates
from pls.editflow.runtime_cost import load_runtime_cost_model
from preparation.plan_pls_editflow_oracle import lpt_shards

EXACT_TREES = ("exact", "exact_selected", "exact_full")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refuse_existing_exact(artifact_root: Path) -> None:
    for name in EXACT_TREES:
        if (artifact_root / name).exists():
            raise SystemExit(
                f"refusing to select: {artifact_root / name} already exists. The frozen "
                "protocol requires selection to be persisted before any confirmatory "
                "mutant is folded."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-plan", type=Path, required=True)
    arguments = parser.parse_args()

    protocol = json.loads(arguments.protocol.read_text())
    calibration = json.loads(arguments.calibration.read_text())

    if protocol.get("status") != "frozen_before_fresh_calibration_exact_scores":
        raise ValueError("cost-aware protocol was not frozen before calibration")
    if protocol.get("evaluate_test", False) or protocol.get("test_evaluated") is not False:
        raise ValueError("test evaluation is permanently disabled")
    if calibration.get("status") != "frozen_quantiles_from_fresh_calibration_components":
        raise ValueError("calibration artifact does not carry frozen fresh quantiles")
    if calibration.get("test_evaluated") is not False:
        raise ValueError("calibration artifact is not test-free")
    if calibration.get("protocol_sha256") != _sha256(arguments.protocol):
        raise ValueError("calibration was produced under a different protocol revision")
    if calibration.get("runtime_cost_model_sha256") != protocol["runtime_cost_model_sha256"]:
        raise ValueError("calibration used a different runtime cost model")

    section = protocol["fresh_confirmatory"]
    manifest_path = Path(section["manifest"])
    if _sha256(manifest_path) != section["manifest_sha256"]:
        raise ValueError("confirmatory manifest hash does not match the frozen protocol")

    fixed_scores = Path(section["fixed_scores"])
    artifact_root = fixed_scores.parents[2]
    _refuse_existing_exact(artifact_root)

    runtime_model = load_runtime_cost_model(protocol["runtime_cost_model"])
    landscapes = load_landscapes(manifest_path, fixed_scores, None)
    if len(landscapes) != int(section["components"]):
        raise ValueError("confirmatory component budget mismatch")
    if any(row["exact"] is not None for row in landscapes):
        raise ValueError("confirmatory selection must not read exact scores")

    calibration_manifest = json.loads(Path(protocol["fresh_calibration"]["manifest"]).read_text())
    calibration_components = {
        node["component_root_sha256"]
        for node in calibration_manifest["nodes"]
        if node["kind"] == "anchor"
    }
    confirmatory_components = {row["component"] for row in landscapes}
    overlap = confirmatory_components & calibration_components
    if overlap:
        raise ValueError(f"confirmatory shares {len(overlap)} components with calibration")

    manifest = json.loads(manifest_path.read_text())
    nodes = manifest["nodes"]

    primary_id = protocol["primary_policy_id"]
    policies = []
    selected_nodes_by_policy: dict[str, list[int]] = {}
    for policy in calibration["policies"]:
        selections = select_candidates(landscapes, policy, runtime_model)
        queries = np.asarray([s["selected_queries"] for s in selections], dtype=np.float64)
        predicted = np.asarray(
            [s["predicted_marginal_gpu_seconds"] for s in selections], dtype=np.float64
        )
        exhaustive = np.asarray(
            [s["predicted_exhaustive_gpu_seconds"] for s in selections], dtype=np.float64
        )
        chosen: list[int] = []
        for s in selections:
            chosen.extend(int(i) for i in s["selected_target_node_indices"])
        selected_nodes_by_policy[policy["policy_id"]] = sorted(set(chosen))
        policies.append({
            "policy_id": policy["policy_id"],
            "role": policy["role"],
            "alpha": float(policy["alpha"]),
            "epsilon": float(policy["epsilon"]),
            "gamma": float(policy["gamma"]),
            "quantile": float(policy["quantile"]),
            "selected_queries_total": int(queries.sum()),
            "exhaustive_queries_total": int(sum(len(r["low"]) for r in landscapes)),
            "mean_selected_queries": float(queries.mean()),
            "minimum_selected_queries": int(queries.min()),
            "maximum_selected_queries": int(queries.max()),
            "exact_query_fraction": float(
                queries.sum() / sum(len(r["low"]) for r in landscapes)
            ),
            "predicted_selected_gpu_seconds": float(predicted.sum()),
            "predicted_exhaustive_gpu_seconds": float(exhaustive.sum()),
            "predicted_gpu_cost_fraction": float(predicted.sum() / exhaustive.sum()),
            "selections": selections,
        })

    primary = next(p for p in policies if p["policy_id"] == primary_id)
    primary_nodes = set(selected_nodes_by_policy[primary_id])
    plan_nodes = [
        node for node in nodes
        if node["kind"] == "single_mutant" and int(node["node_index"]) in primary_nodes
    ]
    if len(plan_nodes) != len(primary_nodes):
        raise ValueError("selected node indices do not map onto confirmatory mutants")
    assignments = lpt_shards(
        plan_nodes, int(protocol["compute_contract"]["fold_shards"]), runtime_model
    )
    identity = "\n".join(
        f"{row['node_index']}:{row['sequence_sha256']}:{row['shard']}" for row in assignments
    )
    shard_count = int(protocol["compute_contract"]["fold_shards"])
    fold_plan = {
        "schema": "PLS_EditFlow_ESMFold_query_plan_v1",
        "shard_count": shard_count,
        "cost_proxy": "frozen_monotone_predicted_marginal_esmfold_gpu_seconds",
        "selection_policy_id": primary_id,
        "selection_scope": "gated_selected_candidates_only",
        "assignments": assignments,
        "assignments_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.fold_plan.write_text(json.dumps(fold_plan, indent=2, sort_keys=True) + "\n")

    loads = [
        sum(row["estimated_cost"] for row in assignments if row["shard"] == shard)
        for shard in range(shard_count)
    ]
    result = {
        "schema": "PLS_cost_aware_conformal_confirmatory_selection_v2",
        "status": "selected_before_any_confirmatory_mutant_fold",
        "protocol": str(arguments.protocol),
        "protocol_sha256": _sha256(arguments.protocol),
        "calibration_artifact": str(arguments.calibration),
        "calibration_artifact_sha256": _sha256(arguments.calibration),
        "confirmatory_manifest": str(manifest_path),
        "confirmatory_manifest_sha256": section["manifest_sha256"],
        "fixed_scores": str(fixed_scores),
        "fixed_scores_sha256": _sha256(fixed_scores),
        "runtime_cost_model_sha256": protocol["runtime_cost_model_sha256"],
        "components": len(landscapes),
        "calibration_component_overlap": 0,
        "primary_policy_id": primary_id,
        "fold_plan": str(arguments.fold_plan),
        "fold_plan_assignments_sha256": fold_plan["assignments_sha256"],
        "fold_plan_queries": len(assignments),
        "fold_plan_estimated_cost_by_shard": loads,
        "fold_plan_maximum_to_minimum_cost_ratio": max(loads) / min(loads),
        "exact_scores_read": False,
        "policies": policies,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    summary = {
        key: result[key]
        for key in (
            "status", "components", "primary_policy_id", "fold_plan_queries",
            "fold_plan_maximum_to_minimum_cost_ratio", "exact_scores_read",
        )
    }
    summary["policies"] = [
        {
            "policy_id": p["policy_id"],
            "role": p["role"],
            "quantile": p["quantile"],
            "mean_selected_queries": p["mean_selected_queries"],
            "exact_query_fraction": p["exact_query_fraction"],
            "predicted_gpu_cost_fraction": p["predicted_gpu_cost_fraction"],
        }
        for p in policies
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
