"""Train-component-only development of regret- and cost-aware conformal sets."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import GroupKFold

from pls.editflow.decision_gating import (
    epsilon_optimal_nonconformity,
    finite_sample_quantile,
    margin_candidate_indices,
    regret_summary,
)
from pls.editflow.runtime_cost import (
    load_runtime_cost_model,
    predict_runtime_seconds,
    runtime_cost_scale,
)


def _score_map(path: Path) -> dict[str, float]:
    values = np.load(path)
    return {
        str(key): float(value)
        for key, value in zip(values["sequence_sha256"], values["logits"])
    }


def _load_train_anchors(section: dict) -> list[dict]:
    manifest = json.loads(Path(section["manifest"]).read_text())
    if manifest.get("test_evaluated") is not False or section["split"] != "train":
        raise ValueError("epsilon-cost development requires explicitly test-free train data")
    fixed = _score_map(Path(section["fixed_scores"]))
    exact = _score_map(Path(section["exact_scores"]))
    anchors: dict[int, dict] = {}
    for edge in manifest["edges"]:
        source = manifest["nodes"][int(edge["source_node"])]
        target = manifest["nodes"][int(edge["target_node"])]
        if source["split"] != "train":
            continue
        if target["split"] != "train" or source["component_root_sha256"] != target["component_root_sha256"]:
            raise ValueError("edge crosses split or SI30 component")
        anchor = int(edge["anchor_rank"])
        row = anchors.setdefault(anchor, {
            "anchor": anchor,
            "component": str(source["component_root_sha256"]),
            "length": int(source["length"]),
            "low": [],
            "exact": [],
        })
        if row["component"] != source["component_root_sha256"] or row["length"] != int(source["length"]):
            raise ValueError("inconsistent anchor metadata")
        row["low"].append(fixed[str(target["sequence_sha256"])] - fixed[str(source["sequence_sha256"])])
        row["exact"].append(exact[str(target["sequence_sha256"])] - exact[str(source["sequence_sha256"])])
    result = []
    for row in anchors.values():
        row["low"] = np.asarray(row["low"], dtype=np.float64)
        row["exact"] = np.asarray(row["exact"], dtype=np.float64)
        result.append(row)
    if not result:
        raise ValueError("development requires at least one train anchor")
    return sorted(result, key=lambda row: row["anchor"])


def _scale(
    length: int,
    reference_length: float,
    gamma: float,
    size: int,
    runtime_model: dict | None,
) -> np.ndarray:
    if runtime_model is not None:
        value = float(runtime_cost_scale(runtime_model, [length], gamma=gamma)[0])
    else:
        value = (float(length) / reference_length) ** (-gamma)
    return np.full(size, value, dtype=np.float64)


def _unit_cost(length: int, runtime_model: dict | None) -> float:
    if runtime_model is not None:
        return float(predict_runtime_seconds(runtime_model, [length])[0])
    return float(length**2)


def _crossfit(
    anchors: list[dict],
    *,
    folds: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    runtime_model: dict | None,
    frozen_reference_length: float | None,
) -> dict:
    groups = np.asarray([row["component"] for row in anchors])
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(anchors))
    records, quantiles = [], []
    for calibration, held_out in splitter.split(dummy, dummy, groups):
        # Legacy v1 derived the length reference inside each calibration fold.
        # New runtime-cost protocols freeze the score form before calibration.
        reference_length = (
            float(frozen_reference_length)
            if frozen_reference_length is not None
            else float(np.median([anchors[index]["length"] for index in calibration]))
        )
        anchor_scores: dict[str, list[float]] = {}
        for index in calibration:
            row = anchors[index]
            scale = _scale(
                row["length"], reference_length, gamma, len(row["low"]), runtime_model
            )
            anchor_scores.setdefault(row["component"], []).append(
                epsilon_optimal_nonconformity(
                    row["low"], row["exact"], epsilon, scale=scale
                )
            )
        scores = [max(values) for values in anchor_scores.values()]
        quantile = finite_sample_quantile(scores, alpha=alpha)
        quantiles.append(quantile)
        for index in held_out:
            row = anchors[index]
            scale = _scale(
                row["length"], reference_length, gamma, len(row["low"]), runtime_model
            )
            chosen = margin_candidate_indices(row["low"], quantile, scale=scale)
            regret = float(np.max(row["exact"]) - np.max(row["exact"][chosen]))
            unit_cost = _unit_cost(row["length"], runtime_model)
            full_cost = float(len(row["low"]) * unit_cost)
            selected_cost = float(len(chosen) * unit_cost)
            records.append({
                "component": row["component"],
                "queries": int(len(chosen)),
                "cost": selected_cost,
                "full_cost": full_cost,
                "regret": regret,
            })
    regrets = np.asarray([row["regret"] for row in records])
    queries = np.asarray([row["queries"] for row in records])
    cost = float(sum(row["cost"] for row in records))
    full_cost = float(sum(row["full_cost"] for row in records))
    component_epsilon_covered = []
    component_exact_covered = []
    for component in sorted({row["component"] for row in records}):
        values = [row["regret"] for row in records if row["component"] == component]
        component_epsilon_covered.append(all(value <= epsilon + 1e-12 for value in values))
        component_exact_covered.append(all(value <= 1e-12 for value in values))
    return {
        "epsilon": epsilon,
        "length_cost_gamma": gamma,
        "marginal_epsilon_coverage": float(np.mean(regrets <= epsilon + 1e-12)),
        "exact_best_coverage": float(np.mean(regrets <= 1e-12)),
        "component_epsilon_coverage": float(np.mean(component_epsilon_covered)),
        "component_exact_best_coverage": float(np.mean(component_exact_covered)),
        "mean_queries": float(np.mean(queries)),
        "median_queries": float(np.median(queries)),
        "query_range": [int(np.min(queries)), int(np.max(queries))],
        "query_fraction": float(np.sum(queries) / sum(len(row["low"]) for row in anchors)),
        "cost_fraction": cost / full_cost,
        "cost_saving_fraction": 1.0 - cost / full_cost,
        "fold_quantile_range": [float(np.min(quantiles)), float(np.max(quantiles))],
        **regret_summary(regrets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--analysis-md", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    if config.get("evaluate_test", False) or config.get("test_evaluated") is not False:
        raise ValueError("test evaluation is permanently disabled")
    if config.get("status") != "train_only_exploratory_method_development":
        raise ValueError("invalid development protocol status")

    anchors = _load_train_anchors(config["data"])
    runtime_model = (
        load_runtime_cost_model(config["runtime_cost_model"])
        if config.get("runtime_cost_model")
        else None
    )
    frozen_reference_length = config.get("frozen_reference_length")
    if runtime_model is not None and frozen_reference_length is not None:
        raise ValueError("runtime model and length reference are mutually exclusive")
    matrix = []
    for epsilon in config["epsilon_grid"]:
        for gamma in config["length_cost_gamma_grid"]:
            matrix.append(_crossfit(
                anchors,
                folds=int(config["component_folds"]),
                alpha=float(config["alpha"]),
                epsilon=float(epsilon),
                gamma=float(gamma),
                runtime_model=runtime_model,
                frozen_reference_length=(
                    float(frozen_reference_length)
                    if frozen_reference_length is not None
                    else None
                ),
            ))
    result = {
        "schema": "PLS_EditFlow_epsilon_cost_development_report_v1",
        "status": "exploratory_group_crossfit_on_old_train_components_only",
        "anchors": len(anchors),
        "unique_si30_components": len({row["component"] for row in anchors}),
        "alpha": float(config["alpha"]),
        "cost_proxy": config["cost_proxy"],
        "runtime_cost_model": config.get("runtime_cost_model"),
        "score_normalization_frozen_before_calibration": runtime_model is not None
        or frozen_reference_length is not None,
        "matrix": matrix,
        "selection_status": "no method selected; confirmatory v1 was not used for tuning",
        "quantile": "correct direct finite-sample order statistic",
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arguments.config, arguments.run_dir / "config.json")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "test_evaluated": False,
    }
    (arguments.run_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (arguments.run_dir / "validation_metrics.json").write_text(output)
    arguments.analysis_json.write_text(output)

    lines = [
        "# Epsilon-optimal and cost-aware conformal development v1",
        "",
        f"Exploratory five-fold cross-fitting uses only the old 128 train anchors from {result['unique_si30_components']} SI30 components. Calibration uses the maximum anchor nonconformity per component. The held-out confirmatory 64 components and the PLS test split were not used for method tuning.",
        "",
        "| Epsilon | Cost gamma | Anchor epsilon cov. | Component epsilon cov. | Exact coverage | Mean queries | Query fraction | Cost fraction | Mean regret | CVaR95 | Max regret |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in matrix:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['length_cost_gamma']:.2f} | "
            f"{row['marginal_epsilon_coverage']:.4f} | {row['component_epsilon_coverage']:.4f} | "
            f"{row['exact_best_coverage']:.4f} | "
            f"{row['mean_queries']:.2f} | {row['query_fraction']:.4f} | "
            f"{row['cost_fraction']:.4f} | {row['mean_regret']:.4f} | "
            f"{row['regret_cvar95']:.4f} | {row['maximum_regret']:.4f} |"
        )
    lines.extend([
        "",
        "`gamma=0` is the ordinary margin score. Positive gamma tightens candidate sets for more expensive anchors. New runtime-cost protocols use a label-free monotone model and a reference cost frozen before conformal calibration. This matrix is development evidence, not a new confirmatory result.",
        "",
        "PLS test queries/evaluations: **0**.",
        "",
    ])
    report = "\n".join(lines)
    (arguments.run_dir / "report.md").write_text(report)
    arguments.analysis_md.write_text(report)
    print(output, end="")


if __name__ == "__main__":
    main()
