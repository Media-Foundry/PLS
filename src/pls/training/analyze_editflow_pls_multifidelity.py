"""Fit frozen multi-fidelity PLS correction baselines without touching test."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pls.editflow.multifidelity import (
    correlation_metrics,
    delta_metrics,
    grouped_oof_predictions,
    selective_hybrid,
)


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _memmap(path: Path, metadata_path: Path) -> np.memmap:
    metadata = json.loads(metadata_path.read_text())
    dtype = np.dtype(metadata["dtype"])
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(metadata["shape"]))


def _score_map(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    values = np.load(path)
    full = {str(key): float(value) for key, value in zip(values["sequence_sha256"], values["logits"])}
    sequence = {
        str(key): float(value)
        for key, value in zip(values["sequence_sha256"], values["sequence_only_logits"])
    }
    return full, sequence


def build_edge_table(config: dict) -> dict[str, np.ndarray]:
    paths = {name: Path(value) for name, value in config["data"].items()}
    dense = json.loads(paths["dense_manifest"].read_text())
    exact = json.loads(paths["exact_manifest"].read_text())
    if dense.get("test_evaluated") is not False or exact.get("test_evaluated") is not False:
        raise ValueError("multi-fidelity analysis requires explicitly test-free manifests")
    if any(node["split"] not in {"train", "validation"} for node in exact["nodes"]):
        raise ValueError("forbidden split in exact calibration manifest")

    fixed, sequence_only = _score_map(paths["fixed_scores"])
    high, _ = _score_map(paths["exact_scores"])
    dense_index = {node["sequence_sha256"]: int(node["node_index"]) for node in dense["nodes"]}
    embeddings = np.load(paths["mean_embeddings"], mmap_mode="r")
    offsets = np.load(paths["compact_offsets"], mmap_mode="r")
    compact = _memmap(paths["compact_features"], paths["compact_metadata"])
    residue = _memmap(paths["residue_pca"], paths["residue_pca_metadata"])

    basic_rows, local_rows, ridge_rows = [], [], []
    low_delta, exact_delta, seq_delta, groups, splits = [], [], [], [], []
    source_hashes, target_hashes = [], []
    aa_index = {residue_name: index for index, residue_name in enumerate(ALPHABET)}
    for edge in exact["edges"]:
        parent = exact["nodes"][int(edge["source_node"])]
        mutant = exact["nodes"][int(edge["target_node"])]
        source_hash, target_hash = parent["sequence_sha256"], mutant["sequence_sha256"]
        source_index, target_index = dense_index[source_hash], dense_index[target_hash]
        mutation = mutant["mutation"]
        position = int(mutation["position_zero_based"])
        length = int(parent["length"])
        low = fixed[target_hash] - fixed[source_hash]
        truth = high[target_hash] - high[source_hash]
        seq = sequence_only[target_hash] - sequence_only[source_hash]
        source_onehot = np.zeros(20, dtype=np.float32)
        target_onehot = np.zeros(20, dtype=np.float32)
        source_onehot[aa_index[mutation["source_residue"]]] = 1
        target_onehot[aa_index[mutation["target_residue"]]] = 1
        basic = np.concatenate([
            np.asarray([
                low,
                seq,
                abs(low),
                abs(seq),
                low * seq,
                length / 300.0,
                np.log1p(length) / np.log1p(300.0),
                position / max(length - 1, 1),
                min(position, length - 1 - position) / max(length - 1, 1),
            ], dtype=np.float32),
            source_onehot,
            target_onehot,
        ])
        parent_residue = int(offsets[source_index]) + position
        mutant_residue = int(offsets[target_index]) + position
        local_structure = np.asarray(compact[parent_residue], dtype=np.float32)
        parent_token = np.asarray(residue[parent_residue], dtype=np.float32)
        mutant_token = np.asarray(residue[mutant_residue], dtype=np.float32)
        mean_parent = np.asarray(embeddings[source_index], dtype=np.float32)
        mean_mutant = np.asarray(embeddings[target_index], dtype=np.float32)
        local = np.concatenate([basic, local_structure, parent_token, mutant_token - parent_token])
        ridge = np.concatenate([local, mean_parent, mean_mutant - mean_parent])
        basic_rows.append(basic)
        local_rows.append(local)
        ridge_rows.append(ridge)
        low_delta.append(low)
        exact_delta.append(truth)
        seq_delta.append(seq)
        groups.append(parent["component_root_sha256"])
        splits.append(parent["split"])
        source_hashes.append(source_hash)
        target_hashes.append(target_hash)
    return {
        "basic": np.asarray(basic_rows, dtype=np.float32),
        "local": np.asarray(local_rows, dtype=np.float32),
        "ridge": np.asarray(ridge_rows, dtype=np.float32),
        "low_delta": np.asarray(low_delta, dtype=np.float64),
        "exact_delta": np.asarray(exact_delta, dtype=np.float64),
        "sequence_delta": np.asarray(seq_delta, dtype=np.float64),
        "groups": np.asarray(groups),
        "splits": np.asarray(splits),
        "source_hashes": np.asarray(source_hashes),
        "target_hashes": np.asarray(target_hashes),
    }


def _fit_candidates(table: dict[str, np.ndarray], config: dict) -> tuple[dict, dict, str, np.ndarray]:
    train = table["splits"] == "train"
    low = table["low_delta"][train]
    truth = table["exact_delta"][train]
    residual = truth - low
    groups = table["groups"][train]
    folds = int(config["protocol"]["group_folds"])
    top_k = int(config["protocol"]["top_k"])

    oof: dict[str, np.ndarray] = {"fixed_parent": low.copy()}
    models: dict[str, object] = {}
    affine = LinearRegression()
    oof["affine"] = grouped_oof_predictions(affine, low[:, None], truth, groups, folds=folds)
    models["affine"] = affine.fit(low[:, None], truth)

    ridge_records = {}
    ridge_predictions = {}
    ridge_estimators = {}
    for alpha in config["ridge"]["alphas"]:
        name = f"alpha_{float(alpha):g}"
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        residual_prediction = grouped_oof_predictions(
            estimator, table["ridge"][train], residual, groups, folds=folds
        )
        prediction = low + residual_prediction
        ridge_predictions[name] = prediction
        ridge_records[name] = float(np.sqrt(np.mean(np.square(prediction - truth))))
        ridge_estimators[name] = estimator
    ridge_choice = min(ridge_records, key=lambda name: (ridge_records[name], name))
    oof["ridge_residual"] = ridge_predictions[ridge_choice]
    models["ridge_residual"] = ridge_estimators[ridge_choice].fit(table["ridge"][train], residual)

    nonlinear_records = {}
    nonlinear_predictions = {}
    nonlinear_estimators = {}
    nonlinear_config = config["nonlinear"]
    for leaf in nonlinear_config["minimum_leaf_candidates"]:
        name = f"minimum_leaf_{int(leaf)}"
        estimator = ExtraTreesRegressor(
            n_estimators=int(nonlinear_config["estimators"]),
            max_features=float(nonlinear_config["max_features"]),
            min_samples_leaf=int(leaf),
            n_jobs=int(nonlinear_config["workers"]),
            random_state=int(config["protocol"]["random_seed"]),
        )
        residual_prediction = grouped_oof_predictions(
            estimator, table["local"][train], residual, groups, folds=folds
        )
        prediction = low + residual_prediction
        nonlinear_predictions[name] = prediction
        nonlinear_records[name] = float(np.sqrt(np.mean(np.square(prediction - truth))))
        nonlinear_estimators[name] = estimator
    nonlinear_choice = min(nonlinear_records, key=lambda name: (nonlinear_records[name], name))
    oof["nonlinear_residual"] = nonlinear_predictions[nonlinear_choice]
    models["nonlinear_residual"] = nonlinear_estimators[nonlinear_choice].fit(
        table["local"][train], residual
    )

    oof_metrics = {name: delta_metrics(truth, prediction, groups, top_k=top_k) for name, prediction in oof.items()}
    selected = min(oof_metrics, key=lambda name: (oof_metrics[name]["edge_rmse"], name))
    selection = {
        "metric": config["protocol"]["selection_metric"],
        "selected_correction": selected,
        "ridge_selected": ridge_choice,
        "ridge_candidate_oof_rmse": ridge_records,
        "nonlinear_selected": nonlinear_choice,
        "nonlinear_candidate_oof_rmse": nonlinear_records,
    }
    return models, {"metrics": oof_metrics, "selection": selection}, selected, oof[selected]


def _validation_predictions(table: dict[str, np.ndarray], models: dict, train_oof: dict, selected: str) -> tuple[dict, dict]:
    validation = table["splits"] == "validation"
    train = table["splits"] == "train"
    low_val = table["low_delta"][validation]
    residual_train = table["exact_delta"][train] - table["low_delta"][train]
    predictions = {
        "fixed_parent": low_val,
        "affine": models["affine"].predict(low_val[:, None]),
        "ridge_residual": low_val + models["ridge_residual"].predict(table["ridge"][validation]),
        "nonlinear_residual": low_val + models["nonlinear_residual"].predict(table["local"][validation]),
    }
    truth = table["exact_delta"][validation]
    groups = table["groups"][validation]
    metrics = {name: delta_metrics(truth, prediction, groups, top_k=5) for name, prediction in predictions.items()}
    return predictions, metrics


def _random_curve(
    approximate: np.ndarray,
    exact: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    *,
    repetitions: int,
    seed: int,
    top_k: int,
) -> dict:
    count = int(round(fraction * len(exact)))
    records = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(seed + repetition)
        chosen = rng.choice(len(exact), size=count, replace=False)
        hybrid = approximate.copy()
        hybrid[chosen] = exact[chosen]
        records.append(delta_metrics(exact, hybrid, groups, top_k=top_k))
    keys = ["edge_spearman", "mutation_sign_accuracy", "edge_rmse", f"anchor_macro_top_{top_k}_recall"]
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "standard_deviation": float(np.std([record[key] for record in records])),
        }
        for key in keys
    }


def _selective_curves(table: dict, predictions: dict, selected: str, train_oof_prediction: np.ndarray, config: dict) -> dict:
    train = table["splits"] == "train"
    validation = table["splits"] == "validation"
    truth_train = table["exact_delta"][train]
    low_train = table["low_delta"][train]
    truth = table["exact_delta"][validation]
    groups = table["groups"][validation]
    low = table["low_delta"][validation]
    selected_prediction = predictions[selected]
    settings = config["selective_refolding"]

    def discrepancy_model(target: np.ndarray):
        return ExtraTreesRegressor(
            n_estimators=int(settings["estimators"]),
            max_features=float(settings["max_features"]),
            min_samples_leaf=int(settings["minimum_leaf"]),
            n_jobs=int(settings["workers"]),
            random_state=int(config["protocol"]["random_seed"]) + 17,
        ).fit(table["local"][train], target)

    raw_model = discrepancy_model(np.abs(truth_train - low_train))
    corrected_model = discrepancy_model(np.abs(truth_train - train_oof_prediction))
    raw_priority = raw_model.predict(table["local"][validation])
    corrected_priority = corrected_model.predict(table["local"][validation])
    actual_raw_error = np.abs(truth - low)
    actual_corrected_error = np.abs(truth - selected_prediction)
    predictor_fidelity = {
        "raw_fixed_discrepancy": correlation_metrics(actual_raw_error, raw_priority),
        "corrected_discrepancy": correlation_metrics(actual_corrected_error, corrected_priority),
    }
    fractions = [float(value) for value in config["protocol"]["refold_fractions"]]
    curves = {"raw_fixed": [], "selected_correction": []}
    repetitions = int(config["protocol"]["random_refold_repetitions"])
    seed = int(config["protocol"]["random_seed"])
    top_k = int(config["protocol"]["top_k"])
    for curve_name, approximate, priority, actual_error in (
        ("raw_fixed", low, raw_priority, actual_raw_error),
        ("selected_correction", selected_prediction, corrected_priority, actual_corrected_error),
    ):
        for fraction in fractions:
            hybrid, chosen = selective_hybrid(approximate, truth, priority, fraction)
            oracle_hybrid, _ = selective_hybrid(approximate, truth, actual_error, fraction)
            curves[curve_name].append({
                "exact_fold_fraction": fraction,
                "exact_fold_count": int(len(chosen)),
                "selective": delta_metrics(truth, hybrid, groups, top_k=top_k),
                "random": _random_curve(
                    approximate, truth, groups, fraction,
                    repetitions=repetitions, seed=seed, top_k=top_k,
                ),
                "oracle_discrepancy_upper_bound": delta_metrics(
                    truth, oracle_hybrid, groups, top_k=top_k
                ),
            })
    return {"discrepancy_predictor_fidelity": predictor_fidelity, "curves": curves}


def _node_alignment(config: dict, table: dict) -> dict:
    fixed, _ = _score_map(Path(config["data"]["fixed_scores"]))
    exact, _ = _score_map(Path(config["data"]["exact_scores"]))
    manifest = json.loads(Path(config["data"]["exact_manifest"]).read_text())
    result = {}
    for split in ("train", "validation", "all"):
        hashes = [
            node["sequence_sha256"] for node in manifest["nodes"]
            if split == "all" or node["split"] == split
        ]
        truth = np.asarray([exact[key] for key in hashes])
        prediction = np.asarray([fixed[key] for key in hashes])
        result[split] = {"nodes": len(hashes), **correlation_metrics(truth, prediction)}
    return result


def _markdown(result: dict) -> str:
    validation = result["validation_metrics"]
    rows = []
    for name in ("fixed_parent", "affine", "ridge_residual", "nonlinear_residual"):
        metric = validation[name]
        rows.append(
            f"| {name} | {metric['edge_pearson']:.4f} | {metric['edge_spearman']:.4f} | "
            f"{metric['mutation_sign_accuracy']:.4f} | {metric['edge_rmse']:.4f} | "
            f"{metric['anchor_macro_kendall_tau']:.4f} | {metric['anchor_macro_top_5_recall']:.4f} |"
        )
    selected = result["selection"]["selected_correction"]
    lines = [
        "# PLS multi-fidelity correction v1",
        "",
        "This test-free analysis uses 1,024 component-unique train exact edges for grouped-CV correction and evaluates the prespecified candidates once on 2,048 exact validation edges.",
        "",
        "| Method | Edge Pearson | Edge Spearman | Sign | RMSE | Macro Kendall | Top-5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        f"Train grouped-OOF RMSE selected `{selected}` before validation reporting.",
        "",
        "## Selective refolding",
        "",
        "The table uses the selected correction as the unrefolded prediction. Random values are means over the frozen repetitions; oracle discrepancy is non-deployable context.",
        "",
        "| Exact fraction | Selective Spearman | Random Spearman | Oracle Spearman | Selective sign | Selective Top-5 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["selective_refolding"]["curves"]["selected_correction"]:
        selective = row["selective"]
        random = row["random"]
        oracle = row["oracle_discrepancy_upper_bound"]
        lines.append(
            f"| {row['exact_fold_fraction']:.0%} | {selective['edge_spearman']:.4f} | "
            f"{random['edge_spearman']['mean']:.4f} | {oracle['edge_spearman']:.4f} | "
            f"{selective['mutation_sign_accuracy']:.4f} | {selective['anchor_macro_top_5_recall']:.4f} |"
        )
    lines.extend([
        "",
        "The fixed-parent score is conditional on the anchor structure, so these results establish local single-edit fidelity only; they do not define a global scalar potential over multi-step paths.",
        "",
        "Repository test queries/evaluations: **0**.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path)
    parser.add_argument("--analysis-md", type=Path)
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

    table = build_edge_table(config)
    models, train_report, selected, selected_train_oof = _fit_candidates(table, config)
    predictions, validation_metrics = _validation_predictions(table, models, train_report, selected)
    result = {
        "schema": "PLS_EditFlow_multifidelity_correction_report_v1",
        "protocol": {
            "train_exact_edges": int(np.sum(table["splits"] == "train")),
            "validation_exact_edges": int(np.sum(table["splits"] == "validation")),
            "train_unique_components": int(len(np.unique(table["groups"][table["splits"] == "train"]))),
            "validation_unique_components": int(len(np.unique(table["groups"][table["splits"] == "validation"]))),
            "group_folds": int(config["protocol"]["group_folds"]),
            "validation_access": "one_pass_after_train_grouped_cv_selection",
            "test_sequences_queried": 0,
            "test_evaluated": False,
        },
        "node_alignment": _node_alignment(config, table),
        "train_grouped_oof_metrics": train_report["metrics"],
        "selection": train_report["selection"],
        "validation_metrics": validation_metrics,
        "selective_refolding": _selective_curves(
            table, predictions, selected, selected_train_oof, config
        ),
        "test_evaluated": False,
    }
    output_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output_md = _markdown(result)
    (arguments.run_dir / "validation_metrics.json").write_text(output_json)
    (arguments.run_dir / "report.md").write_text(output_md)
    if arguments.analysis_json:
        arguments.analysis_json.write_text(output_json)
    if arguments.analysis_md:
        arguments.analysis_md.write_text(output_md)
    print(output_json)


if __name__ == "__main__":
    main()
