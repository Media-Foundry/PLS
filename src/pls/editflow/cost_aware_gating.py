"""Component-safe calibration and evaluation for cost-aware decision gating."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pls.editflow.decision_gating import (
    epsilon_optimal_nonconformity,
    finite_sample_quantile,
    margin_candidate_indices,
)
from pls.editflow.runtime_cost import predict_runtime_seconds, runtime_cost_scale


def load_score_map(path: str | Path) -> dict[str, float]:
    values = np.load(path)
    return {
        str(key): float(value)
        for key, value in zip(values["sequence_sha256"], values["logits"])
    }


def load_landscapes(
    manifest_path: str | Path,
    fixed_scores_path: str | Path,
    exact_scores_path: str | Path | None = None,
) -> list[dict]:
    """Load train-only mutation landscapes without ever admitting test nodes."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("landscape manifest must explicitly remain test-free")
    if any(row["split"] != "train" for row in manifest["nodes"]):
        raise ValueError("cost-aware gating is restricted to train-only manifests")
    fixed = load_score_map(fixed_scores_path)
    exact = load_score_map(exact_scores_path) if exact_scores_path is not None else None
    by_anchor: dict[int, dict] = {}
    for edge in manifest["edges"]:
        source = manifest["nodes"][int(edge["source_node"])]
        target = manifest["nodes"][int(edge["target_node"])]
        anchor = int(edge["anchor_rank"])
        row = by_anchor.setdefault(anchor, {
            "anchor": anchor,
            "component": str(source["component_root_sha256"]),
            "length": int(source["length"]),
            "source_digest": str(source["sequence_sha256"]),
            "target_digests": [],
            "target_node_indices": [],
            "edge_indices": [],
            "low": [],
            "exact": [],
        })
        if row["component"] != source["component_root_sha256"]:
            raise ValueError("anchor crosses SI30 components")
        source_digest = str(source["sequence_sha256"])
        target_digest = str(target["sequence_sha256"])
        row["target_digests"].append(target_digest)
        row["target_node_indices"].append(int(edge["target_node"]))
        row["edge_indices"].append(int(edge["edge_index"]))
        row["low"].append(fixed[target_digest] - fixed[source_digest])
        if exact is not None:
            row["exact"].append(exact[target_digest] - exact[source_digest])
    result = []
    for row in by_anchor.values():
        row["low"] = np.asarray(row["low"], dtype=np.float64)
        row["exact"] = (
            np.asarray(row["exact"], dtype=np.float64) if exact is not None else None
        )
        result.append(row)
    components = [row["component"] for row in result]
    if len(components) != len(set(components)):
        raise ValueError("fresh calibration/confirmatory protocols require one anchor per SI30 component")
    return sorted(result, key=lambda row: row["anchor"])


def calibrate_policies(
    landscapes: list[dict], policies: list[dict], runtime_model: dict
) -> list[dict]:
    """Fit one component-level split-conformal quantile per frozen policy."""
    calibrated = []
    for policy in policies:
        by_component: dict[str, list[float]] = defaultdict(list)
        for row in landscapes:
            if row["exact"] is None:
                raise ValueError("calibration requires exact high-fidelity scores")
            scale = runtime_cost_scale(
                runtime_model,
                np.full(len(row["low"]), row["length"]),
                gamma=float(policy["gamma"]),
            )
            score = epsilon_optimal_nonconformity(
                row["low"], row["exact"], float(policy["epsilon"]), scale=scale
            )
            by_component[row["component"]].append(score)
        component_scores = [max(values) for values in by_component.values()]
        calibrated.append({
            **policy,
            "quantile": finite_sample_quantile(component_scores, alpha=float(policy["alpha"])),
            "calibration_components": len(component_scores),
            "finite_sample_order_rank": min(
                len(component_scores),
                int(np.ceil((len(component_scores) + 1) * (1 - float(policy["alpha"])))),
            ),
        })
    return calibrated


def select_candidates(
    landscapes: list[dict], calibrated_policy: dict, runtime_model: dict
) -> list[dict]:
    """Apply a frozen policy using only cached-oracle scores and covariates."""
    selections = []
    for row in landscapes:
        scale = runtime_cost_scale(
            runtime_model,
            np.full(len(row["low"]), row["length"]),
            gamma=float(calibrated_policy["gamma"]),
        )
        indices = margin_candidate_indices(
            row["low"], float(calibrated_policy["quantile"]), scale=scale
        )
        predicted_unit_cost = float(predict_runtime_seconds(runtime_model, [row["length"]])[0])
        selections.append({
            "anchor": row["anchor"],
            "component": row["component"],
            "length": row["length"],
            "selected_local_indices": indices.astype(int).tolist(),
            "selected_target_node_indices": [row["target_node_indices"][i] for i in indices],
            "selected_edge_indices": [row["edge_indices"][i] for i in indices],
            "selected_sequence_sha256": [row["target_digests"][i] for i in indices],
            "selected_queries": int(len(indices)),
            "predicted_marginal_gpu_seconds": float(len(indices) * predicted_unit_cost),
            "predicted_exhaustive_gpu_seconds": float(len(row["low"]) * predicted_unit_cost),
        })
    return selections
