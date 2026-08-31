"""Compare two frozen-protocol GB1 acquisition runs with paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pls.editflow.statistics import paired_method_summary


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_901


def load_run(path: Path) -> tuple[dict, list[dict], dict]:
    config = json.loads((path / "config.json").read_text())
    history = json.loads((path / "history.json").read_text())
    query_budget = json.loads((path / "query_budget.json").read_text())
    if config.get("evaluate_test", False) or query_budget.get("test_evaluated") is not False:
        raise ValueError("comparison refuses any run that does not preserve the test freeze")
    return config, history, query_budget


def validate_pair(
    path_config: dict,
    path_history: list[dict],
    path_budget: dict,
    uncertainty_config: dict,
    uncertainty_history: list[dict],
    uncertainty_budget: dict,
) -> dict:
    if path_config["data"]["acquisition"] != "path_aware":
        raise ValueError("first run is not path-aware")
    if uncertainty_config["data"]["acquisition"] != "uncertainty":
        raise ValueError("second run is not uncertainty-only")
    for key in ("anchor_protocol", "query_budgets", "initial_query_seed", "edit_radii"):
        if path_config["data"][key] != uncertainty_config["data"][key]:
            raise ValueError(f"data protocol mismatch: {key}")
    if path_config["model"] != uncertainty_config["model"]:
        raise ValueError("student model mismatch")
    path_training = dict(path_config["training"]);path_training.pop("hip_device")
    uncertainty_training = dict(uncertainty_config["training"]);uncertainty_training.pop("hip_device")
    if path_training != uncertainty_training:
        raise ValueError("training protocol mismatch")
    if path_budget["anchor_protocol_sha256"] != uncertainty_budget["anchor_protocol_sha256"]:
        raise ValueError("anchor protocol identity mismatch")
    if len(path_history) != len(uncertainty_history):
        raise ValueError("anchor count mismatch")
    shared_initial_sets = True
    for path_anchor, uncertainty_anchor in zip(path_history, uncertainty_history):
        if path_anchor["anchor"] != uncertainty_anchor["anchor"]:
            raise ValueError("anchor ordering mismatch")
        if path_anchor["initial_query_seed"] != uncertainty_anchor["initial_query_seed"]:
            raise ValueError("initial query seed mismatch")
        if path_anchor["ensemble_seeds"] != uncertainty_anchor["ensemble_seeds"]:
            raise ValueError("ensemble seed mismatch")
        path_stages = path_anchor["stages"]
        uncertainty_stages = uncertainty_anchor["stages"]
        if [row["budget"] for row in path_stages] != [row["budget"] for row in uncertainty_stages]:
            raise ValueError("observed budget curve mismatch")
        shared_initial_sets &= (
            path_stages[0]["queried_nodes_sha256"]
            == uncertainty_stages[0]["queried_nodes_sha256"]
        )
    if not shared_initial_sets:
        raise ValueError("paired runs do not share exact initial queried-node sets")
    return {
        "same_anchor_protocol": True,
        "same_model": True,
        "same_training_except_physical_gpu": True,
        "same_initial_queried_node_sets": True,
        "same_unique_node_budgets": True,
        "edge_weight": path_config["training"]["edge_weight"],
        "test_evaluated": False,
    }


def extract_regret(history: list[dict], stage_indices, radii) -> np.ndarray:
    values = []
    for anchor in history:
        cells = [
            anchor["stages"][stage]["regret"][str(radius)]["regret"]
            for stage in stage_indices
            for radius in radii
        ]
        values.append(float(np.mean(cells)))
    return np.asarray(values, dtype=np.float64)


def compare_runs(path_run: Path, uncertainty_run: Path) -> dict:
    path_config, path_history, path_budget = load_run(path_run)
    uncertainty_config, uncertainty_history, uncertainty_budget = load_run(uncertainty_run)
    validation = validate_pair(
        path_config, path_history, path_budget,
        uncertainty_config, uncertainty_history, uncertainty_budget,
    )
    budgets = list(map(int, path_config["data"]["query_budgets"]))
    radii = list(map(int, path_config["data"]["edit_radii"]))
    primary_path = extract_regret(path_history, range(1, len(budgets)), radii)
    primary_uncertainty = extract_regret(
        uncertainty_history, range(1, len(budgets)), radii
    )
    primary = paired_method_summary(
        primary_path,
        primary_uncertainty,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    primary.update({
        "metric": "anchor mean exact regret across budgets 160/320/640 and radii 1/2/3/4",
        "prespecified_before_run_completion": True,
    })
    final_path = extract_regret(path_history, [len(budgets) - 1], radii)
    final_uncertainty = extract_regret(
        uncertainty_history, [len(budgets) - 1], radii
    )
    final = paired_method_summary(
        final_path,
        final_uncertainty,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED + 1,
    )
    final["metric"] = "anchor mean exact regret across radii 1/2/3/4 at 640 queries"

    budget_curve = []
    for stage, budget in enumerate(budgets):
        row = {"budget": budget, "regret": {}}
        for radius in radii:
            path_values = np.asarray([
                anchor["stages"][stage]["regret"][str(radius)]["regret"]
                for anchor in path_history
            ])
            uncertainty_values = np.asarray([
                anchor["stages"][stage]["regret"][str(radius)]["regret"]
                for anchor in uncertainty_history
            ])
            row["regret"][str(radius)] = paired_method_summary(
                path_values,
                uncertainty_values,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED + 10 * stage + radius,
            )
        path_r2 = np.asarray([anchor["stages"][stage]["value"]["r2"] for anchor in path_history])
        uncertainty_r2 = np.asarray([
            anchor["stages"][stage]["value"]["r2"] for anchor in uncertainty_history
        ])
        row["value_r2"] = paired_method_summary(
            path_r2,
            uncertainty_r2,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED + 100 + stage,
        )
        budget_curve.append(row)
    return {
        "schema": "PLS_EditFlow_GB1_paired_comparison_v1",
        "path_run": str(path_run),
        "uncertainty_run": str(uncertainty_run),
        "validation": validation,
        "inference": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_unit": "frozen confirmatory anchor",
            "secondary_cell_pvalues_are_descriptive_and_unadjusted": True,
        },
        "primary": primary,
        "final_budget_secondary": final,
        "budget_curve_secondary": budget_curve,
        "test_evaluated": False,
    }


def markdown_report(result: dict) -> str:
    primary = result["primary"]
    final = result["final_budget_secondary"]
    lines = [
        "# GB1 Path-OLD confirmatory comparison v1",
        "",
        "Negative paired differences favor path-aware acquisition.",
        "",
        "| endpoint | path mean | uncertainty mean | paired difference | bootstrap 95% CI | exact p | W/T/L |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (("Primary query-curve regret", primary), ("Final-640 regret", final)):
        interval = row["bootstrap_95_ci"]
        lines.append(
            f"| {label} | {row['path_mean']:.6f} | {row['uncertainty_mean']:.6f} | "
            f"{row['path_minus_uncertainty_mean']:.6f} | "
            f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{row['exact_two_sided_sign_flip_pvalue']:.6g} | "
            f"{row['path_wins']}/{row['ties']}/{row['uncertainty_wins']} |"
        )
    lines.extend([
        "",
        "The primary endpoint was fixed before run completion. Per-budget and",
        "per-radius cells in the JSON artifact are secondary, descriptive, and",
        "not multiplicity-adjusted. No PLS test split was evaluated.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-run", type=Path, required=True)
    parser.add_argument("--uncertainty-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare_runs(arguments.path_run, arguments.uncertainty_run)
    arguments.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.output_markdown.write_text(markdown_report(result))


if __name__ == "__main__":
    main()
