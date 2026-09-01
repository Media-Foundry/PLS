"""Decision-focused cached-oracle analyses using no PLS test entities."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import GroupKFold

from pls.editflow.decision_gating import (
    empirical_bayes_slope_prior,
    exact_best_rank,
    finite_sample_quantile,
    query_until_certified,
    shrinkage_slope,
    top_m_exact_verification,
)
from pls.editflow.multifidelity import delta_metrics


def _score_map(path: Path) -> dict[str, float]:
    values = np.load(path)
    return {str(key): float(value) for key, value in zip(values["sequence_sha256"], values["logits"])}


def load_edges(section: dict, *, split: str) -> dict[str, np.ndarray]:
    manifest = json.loads(Path(section["manifest"]).read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("decision analysis requires an explicitly test-free manifest")
    if split not in {"train", "validation"}:
        raise ValueError("only train or validation may be loaded")
    fixed = _score_map(Path(section["fixed_scores"]))
    exact = _score_map(Path(section["exact_scores"]))
    rows = []
    for edge in manifest["edges"]:
        source = manifest["nodes"][int(edge["source_node"])]
        target = manifest["nodes"][int(edge["target_node"])]
        if source["split"] != split:
            continue
        if source["split"] != target["split"] or source["component_root_sha256"] != target["component_root_sha256"]:
            raise ValueError("edge crosses a split or SI component")
        rows.append((
            fixed[target["sequence_sha256"]] - fixed[source["sequence_sha256"]],
            exact[target["sequence_sha256"]] - exact[source["sequence_sha256"]],
            int(edge["anchor_rank"]),
            source["component_root_sha256"],
            int(edge["edge_index"]),
        ))
    if not rows:
        raise ValueError(f"no {split} edges")
    return {
        "low": np.asarray([row[0] for row in rows], dtype=np.float64),
        "exact": np.asarray([row[1] for row in rows], dtype=np.float64),
        "anchor": np.asarray([row[2] for row in rows], dtype=np.int64),
        "component": np.asarray([row[3] for row in rows]),
        "edge_index": np.asarray([row[4] for row in rows], dtype=np.int64),
    }


def top_m_report(config: dict) -> dict:
    data = load_edges(config["current_exact"], split="validation")
    return {
        str(m): top_m_exact_verification(data["low"], data["exact"], data["anchor"], int(m))
        for m in config["current_exact"]["top_m"]
    }


def probe_calibration_report(data: dict, config: dict) -> dict:
    folds = int(config["exhaustive_train"]["component_folds"])
    splitter = GroupKFold(n_splits=folds)
    probe_counts = [int(value) for value in config["exhaustive_train"]["probe_counts"]]
    predictions = {count: [] for count in probe_counts}
    targets = {count: [] for count in probe_counts}
    anchors = {count: [] for count in probe_counts}
    slopes = {count: [] for count in probe_counts if count > 0}
    split_dummy = np.zeros(len(data["low"]))
    for calibration, held_out in splitter.split(split_dummy, split_dummy, data["component"]):
        global_slope, shrinkage = empirical_bayes_slope_prior(
            data["low"][calibration], data["exact"][calibration], data["anchor"][calibration]
        )
        for anchor in np.unique(data["anchor"][held_out]):
            indices = held_out[data["anchor"][held_out] == anchor]
            indices = indices[np.argsort(data["edge_index"][indices], kind="stable")]
            for count in probe_counts:
                probe_count = min(count, len(indices) - 1) if count else 0
                evaluation = indices[probe_count:]
                if count == 0:
                    slope = 1.0
                else:
                    probe = indices[:probe_count]
                    slope = shrinkage_slope(
                        data["low"][probe], data["exact"][probe], global_slope, shrinkage
                    )
                    slopes[count].append(slope)
                predictions[count].extend((slope * data["low"][evaluation]).tolist())
                targets[count].extend(data["exact"][evaluation].tolist())
                anchors[count].extend([int(anchor)] * len(evaluation))
    result = {}
    for count in probe_counts:
        metric = delta_metrics(
            np.asarray(targets[count]), np.asarray(predictions[count]), np.asarray(anchors[count]), top_k=5
        )
        result[str(count)] = {
            "probe_count": count,
            "held_out_edges": len(targets[count]),
            "metrics": metric,
            "slope_mean": float(np.mean(slopes[count])) if count else 1.0,
            "slope_standard_deviation": float(np.std(slopes[count])) if count else 0.0,
            "note": "positive scalar calibration cannot change within-anchor ranking",
        }
    return result


def _aggregate_certificates(records: list[dict], quantiles: list[float]) -> dict:
    queries = np.asarray([record["queries"] for record in records])
    regrets = np.asarray([record["regret"] for record in records])
    covered = np.asarray([record["simultaneous_coverage"] for record in records])
    correct = np.asarray([record["correct"] for record in records])
    return {
        "anchors": len(records),
        "simultaneous_coverage": float(np.mean(covered)),
        "decision_accuracy": float(np.mean(correct)),
        "decision_accuracy_when_covered": float(np.mean(correct[covered])) if covered.any() else None,
        "mean_exact_queries": float(np.mean(queries)),
        "median_exact_queries": float(np.median(queries)),
        "p90_exact_queries": float(np.quantile(queries, 0.9)),
        "mean_exact_fraction": float(np.mean([record["query_fraction"] for record in records])),
        "zero_regret_fraction": float(np.mean(regrets <= 1e-12)),
        "mean_regret": float(np.mean(regrets)),
        "maximum_regret": float(np.max(regrets)),
        "fold_quantile_mean": float(np.mean(quantiles)),
        "fold_quantile_range": [float(np.min(quantiles)), float(np.max(quantiles))],
    }


def certified_gating_report(data: dict, config: dict) -> dict:
    folds = int(config["exhaustive_train"]["component_folds"])
    alpha = float(config["certification"]["familywise_alpha"])
    family_size = int(config["certification"]["mutations_per_anchor"])
    splitter = GroupKFold(n_splits=folds)
    records = {"component_max": [], "bonferroni_edge": []}
    quantiles = {"component_max": [], "bonferroni_edge": []}
    dummy = np.zeros(len(data["low"]))
    for calibration, held_out in splitter.split(dummy, dummy, data["component"]):
        error = np.abs(data["exact"][calibration] - data["low"][calibration])
        component_scores = []
        for component in np.unique(data["component"][calibration]):
            component_scores.append(float(np.max(error[data["component"][calibration] == component])))
        q_component = finite_sample_quantile(component_scores, alpha=alpha)
        q_bonferroni = finite_sample_quantile(error, alpha=alpha / family_size)
        quantiles["component_max"].append(q_component)
        quantiles["bonferroni_edge"].append(q_bonferroni)
        for anchor in np.unique(data["anchor"][held_out]):
            indices = held_out[data["anchor"][held_out] == anchor]
            for name, radius in (("component_max", q_component), ("bonferroni_edge", q_bonferroni)):
                records[name].append(query_until_certified(
                    data["low"][indices], data["exact"][indices], radius
                ))
    return {
        "familywise_alpha": alpha,
        "calibration_unit": "SI30_component",
        "query_policy": "largest_unqueried_upper_bound_until_unique_interval_winner",
        "envelopes": {
            name: _aggregate_certificates(records[name], quantiles[name]) for name in records
        },
    }


def conformal_candidate_set_report(data: dict, config: dict) -> dict:
    """Cross-fit a component-safe Top-M set that contains the exact maximizer."""
    folds = int(config["exhaustive_train"]["component_folds"])
    alpha = float(config["certification"]["familywise_alpha"])
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(data["low"]))
    records, fold_sizes = [], []
    for calibration, held_out in splitter.split(dummy, dummy, data["component"]):
        anchor_ranks = {}
        for anchor in np.unique(data["anchor"][calibration]):
            indices = calibration[data["anchor"][calibration] == anchor]
            anchor_ranks[int(anchor)] = exact_best_rank(data["low"][indices], data["exact"][indices])
        component_max_rank = []
        for component in np.unique(data["component"][calibration]):
            component_anchors = np.unique(data["anchor"][calibration][data["component"][calibration] == component])
            component_max_rank.append(max(anchor_ranks[int(anchor)] for anchor in component_anchors))
        set_size = int(np.ceil(finite_sample_quantile(component_max_rank, alpha=alpha)))
        fold_sizes.append(set_size)
        for anchor in np.unique(data["anchor"][held_out]):
            indices = held_out[data["anchor"][held_out] == anchor]
            low, exact = data["low"][indices], data["exact"][indices]
            order = np.argsort(-low, kind="stable")
            chosen = order[:min(set_size, len(order))]
            exact_best = float(np.max(exact))
            verified_best = float(np.max(exact[chosen]))
            regret = exact_best - verified_best
            records.append({
                "set_size": len(chosen),
                "covered": bool(regret <= 1e-12),
                "regret": float(regret),
            })
    regrets = np.asarray([record["regret"] for record in records])
    sizes = np.asarray([record["set_size"] for record in records])
    full_component_scores = []
    full_anchor_ranks = {}
    for anchor in np.unique(data["anchor"]):
        indices = np.flatnonzero(data["anchor"] == anchor)
        full_anchor_ranks[int(anchor)] = exact_best_rank(data["low"][indices], data["exact"][indices])
    for component in np.unique(data["component"]):
        component_anchors = np.unique(data["anchor"][data["component"] == component])
        full_component_scores.append(max(full_anchor_ranks[int(anchor)] for anchor in component_anchors))
    frozen_size = int(np.ceil(finite_sample_quantile(full_component_scores, alpha=alpha)))
    return {
        "alpha": alpha,
        "calibration_unit": "SI30_component_maximum_exact_best_rank",
        "crossfit_coverage": float(np.mean([record["covered"] for record in records])),
        "crossfit_mean_exact_queries": float(np.mean(sizes)),
        "crossfit_mean_exact_fraction": float(np.mean(sizes / int(config["certification"]["mutations_per_anchor"]))),
        "crossfit_zero_regret_fraction": float(np.mean(regrets <= 1e-12)),
        "crossfit_mean_regret": float(np.mean(regrets)),
        "crossfit_maximum_regret": float(np.max(regrets)),
        "fold_set_size_range": [int(np.min(fold_sizes)), int(np.max(fold_sizes))],
        "frozen_future_set_size": frozen_size,
        "frozen_future_exact_fraction": frozen_size / int(config["certification"]["mutations_per_anchor"]),
        "interpretation": "marginal component-level risk control, not a deterministic per-anchor certificate",
    }


def conformal_margin_set_report(data: dict, config: dict) -> dict:
    """Cross-fit a variable set using the cheap-score gap to the exact maximizer."""
    folds = int(config["exhaustive_train"]["component_folds"])
    alpha = float(config["certification"]["familywise_alpha"])
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(data["low"]))
    records, fold_quantiles = [], []

    def anchor_score(indices: np.ndarray) -> float:
        low, exact = data["low"][indices], data["exact"][indices]
        return float(np.max(low) - low[int(np.argmax(exact))])

    for calibration, held_out in splitter.split(dummy, dummy, data["component"]):
        component_scores = []
        for component in np.unique(data["component"][calibration]):
            component_indices = calibration[data["component"][calibration] == component]
            component_scores.append(max(
                anchor_score(component_indices[data["anchor"][component_indices] == anchor])
                for anchor in np.unique(data["anchor"][component_indices])
            ))
        quantile = finite_sample_quantile(component_scores, alpha=alpha)
        fold_quantiles.append(quantile)
        for anchor in np.unique(data["anchor"][held_out]):
            indices = held_out[data["anchor"][held_out] == anchor]
            low, exact = data["low"][indices], data["exact"][indices]
            chosen = np.flatnonzero(low >= np.max(low) - quantile - 1e-12)
            regret = float(np.max(exact) - np.max(exact[chosen]))
            records.append({"set_size": len(chosen), "covered": regret <= 1e-12, "regret": regret})

    full_component_scores = []
    for component in np.unique(data["component"]):
        component_indices = np.flatnonzero(data["component"] == component)
        full_component_scores.append(max(
            anchor_score(component_indices[data["anchor"][component_indices] == anchor])
            for anchor in np.unique(data["anchor"][component_indices])
        ))
    frozen_quantile = finite_sample_quantile(full_component_scores, alpha=alpha)
    sizes = np.asarray([record["set_size"] for record in records])
    regrets = np.asarray([record["regret"] for record in records])
    family_size = int(config["certification"]["mutations_per_anchor"])
    return {
        "alpha": alpha,
        "nonconformity": "max_fixed_delta_minus_fixed_delta_of_exact_argmax",
        "calibration_unit": "SI30_component_maximum_anchor_nonconformity",
        "crossfit_coverage": float(np.mean(regrets <= 1e-12)),
        "crossfit_mean_exact_queries": float(np.mean(sizes)),
        "crossfit_median_exact_queries": float(np.median(sizes)),
        "crossfit_query_range": [int(np.min(sizes)), int(np.max(sizes))],
        "crossfit_mean_exact_fraction": float(np.mean(sizes / family_size)),
        "crossfit_mean_regret": float(np.mean(regrets)),
        "crossfit_maximum_regret": float(np.max(regrets)),
        "fold_quantile_range": [float(np.min(fold_quantiles)), float(np.max(fold_quantiles))],
        "frozen_future_margin_quantile": float(frozen_quantile),
        "interpretation": "variable-size marginal component-level risk control, not a deterministic certificate",
    }


def markdown(result: dict) -> str:
    lines = [
        "# Decision-focused cached-oracle development v1",
        "",
        "No PLS test entity was loaded, queried, scored, or evaluated.",
        "",
        "## Top-M exact verification on the frozen 128-component validation report",
        "",
        "| M | Exact fraction | True-best inclusion | Zero regret | Mean regret | P90 regret |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in result["top_m_validation"].items():
        lines.append(
            f"| {key} | {row['mean_exact_fraction']:.4f} | {row['true_best_inclusion']:.4f} | "
            f"{row['zero_regret_fraction']:.4f} | {row['mean_regret']:.4f} | {row['regret_p90']:.4f} |"
        )
    lines.extend([
        "",
        "## Protein-specific shrinkage-slope diagnostic on old exhaustive train anchors",
        "",
        "| Exact probes/anchor | Held-out edges | RMSE | Pearson | Sign |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for key, row in result["probe_calibration_train_only"].items():
        metric = row["metrics"]
        lines.append(
            f"| {key} | {row['held_out_edges']} | {metric['edge_rmse']:.4f} | "
            f"{metric['edge_pearson']:.4f} | {metric['mutation_sign_accuracy']:.4f} |"
        )
    lines.extend([
        "",
        "A positive per-anchor scalar cannot change mutation ordering; this diagnostic tests magnitude attenuation only.",
        "",
        "## Train-only simultaneous certified gating",
        "",
        "| Envelope | Simultaneous coverage | Decision accuracy | Mean exact queries | Exact fraction | Mean regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, row in result["certified_gating_train_only"]["envelopes"].items():
        lines.append(
            f"| {name} | {row['simultaneous_coverage']:.4f} | {row['decision_accuracy']:.4f} | "
            f"{row['mean_exact_queries']:.2f} | {row['mean_exact_fraction']:.4f} | {row['mean_regret']:.4f} |"
        )
    candidate = result["conformal_candidate_set_train_only"]
    margin = result["conformal_margin_set_train_only"]
    lines.extend([
        "",
        "## Train-only conformal exact-best candidate set",
        "",
        f"Component-safe cross-fit coverage is `{candidate['crossfit_coverage']:.4f}` with mean "
        f"`{candidate['crossfit_mean_exact_queries']:.2f}` exact queries per 16-mutation neighborhood. "
        f"The frozen future set size is `{candidate['frozen_future_set_size']}`. This is marginal "
        "risk control, not a deterministic per-anchor certificate.",
        "",
        "## Train-only conformal decision-margin set",
        "",
        f"The variable-size margin set reaches cross-fit coverage `{margin['crossfit_coverage']:.4f}` "
        f"with mean `{margin['crossfit_mean_exact_queries']:.2f}` exact queries. Its frozen future "
        f"margin threshold is `{margin['frozen_future_margin_quantile']:.6f}`. This is marginal "
        "component-level risk control, not a deterministic per-anchor certificate.",
    ])
    lines.extend([
        "",
        "The certified-gating analysis is cross-fitted development evidence on train components only. It does not reuse the current validation set for method selection.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--analysis-md", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    if config.get("evaluate_test", False):
        raise ValueError("test evaluation is permanently disabled")
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
    exhaustive = load_edges(config["exhaustive_train"], split="train")
    result = {
        "schema": "PLS_EditFlow_decision_gating_development_report_v1",
        "top_m_validation": top_m_report(config),
        "probe_calibration_train_only": probe_calibration_report(exhaustive, config),
        "certified_gating_train_only": certified_gating_report(exhaustive, config),
        "conformal_candidate_set_train_only": conformal_candidate_set_report(exhaustive, config),
        "conformal_margin_set_train_only": conformal_margin_set_report(exhaustive, config),
        "protocol": {
            "top_m_is_prespecified_direct_baseline": True,
            "probe_and_certification_development_split": "old_exhaustive_train_only",
            "current_validation_used_for_method_selection": False,
            "test_sequences_queried": 0,
            "test_evaluated": False,
        },
        "test_evaluated": False,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    report = markdown(result)
    (arguments.run_dir / "validation_metrics.json").write_text(output)
    (arguments.run_dir / "report.md").write_text(report)
    arguments.analysis_json.write_text(output)
    arguments.analysis_md.write_text(report)
    print(output, end="")


if __name__ == "__main__":
    main()
