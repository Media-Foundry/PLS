"""Descriptive comparison for the post-confirmatory bound-aware follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pls.editflow.statistics import paired_method_summary
from compare_gb1_confirmatory import extract_regret, load_run


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_902


def validate_followup(bound_config, bound_history, baseline_config, baseline_history):
    if bound_config["data"]["acquisition"] != "bound_aware":
        raise ValueError("candidate run is not bound-aware")
    if bound_config.get("interpretation", {}).get("status") != "exploratory_followup":
        raise ValueError("bound-aware run is not labelled exploratory")
    for key in ("anchor_protocol", "query_budgets", "initial_query_seed", "edit_radii"):
        if bound_config["data"][key] != baseline_config["data"][key]:
            raise ValueError(f"data protocol mismatch: {key}")
    if bound_config["model"] != baseline_config["model"]:
        raise ValueError("student model mismatch")
    bound_training = dict(bound_config["training"]);bound_training.pop("hip_device")
    baseline_training = dict(baseline_config["training"]);baseline_training.pop("hip_device")
    if bound_training != baseline_training:
        raise ValueError("training protocol mismatch")
    if len(bound_history) != len(baseline_history):
        raise ValueError("anchor count mismatch")
    for candidate, baseline in zip(bound_history, baseline_history):
        if candidate["anchor"] != baseline["anchor"]:
            raise ValueError("anchor ordering mismatch")
        if candidate["stages"][0]["queried_nodes_sha256"] != baseline["stages"][0]["queried_nodes_sha256"]:
            raise ValueError("initial queried-node set mismatch")


def endpoint(history, *, final_only: bool):
    budgets = len(history[0]["stages"])
    radii = [1, 2, 3, 4]
    stages = [budgets - 1] if final_only else range(1, budgets)
    return extract_regret(history, stages, radii)


def compare(bound_run: Path, path_run: Path, uncertainty_run: Path) -> dict:
    bound_config, bound_history, bound_budget = load_run(bound_run)
    path_config, path_history, path_budget = load_run(path_run)
    uncertainty_config, uncertainty_history, uncertainty_budget = load_run(uncertainty_run)
    validate_followup(bound_config, bound_history, path_config, path_history)
    validate_followup(bound_config, bound_history, uncertainty_config, uncertainty_history)
    if len({bound_budget["anchor_protocol_sha256"], path_budget["anchor_protocol_sha256"], uncertainty_budget["anchor_protocol_sha256"]}) != 1:
        raise ValueError("anchor protocol identity mismatch")
    comparisons = {}
    for final_only, endpoint_name in ((False, "query_curve"), (True, "final_640")):
        bound_values = endpoint(bound_history, final_only=final_only)
        comparisons[endpoint_name] = {}
        for name, history in (("path", path_history), ("uncertainty", uncertainty_history)):
            baseline_values = endpoint(history, final_only=final_only)
            comparisons[endpoint_name][f"bound_minus_{name}"] = paired_method_summary(
                bound_values,
                baseline_values,
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED + int(final_only) * 10 + len(name),
            )
    return {
        "schema": "PLS_EditFlow_GB1_bound_followup_comparison_v1",
        "status": "exploratory_posthoc",
        "reason": "The bound-aware arm was developed after partial paired-run results were visible.",
        "bound_run": str(bound_run),
        "path_run": str(path_run),
        "uncertainty_run": str(uncertainty_run),
        "same_anchor_protocol": True,
        "same_initial_queried_node_sets": True,
        "same_training_protocol": True,
        "comparisons": comparisons,
        "pvalues_are_descriptive_not_confirmatory": True,
        "test_evaluated": False,
    }


def markdown(result: dict) -> str:
    lines = [
        "# GB1 bound-aware follow-up v1",
        "",
        "This is an exploratory post-hoc arm, not a confirmatory result. Negative",
        "paired differences favor bound-aware acquisition.",
        "",
        "| endpoint | comparison | bound mean | baseline mean | difference | 95% CI | W/T/L |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for endpoint_name, comparisons in result["comparisons"].items():
        for name, row in comparisons.items():
            interval = row["bootstrap_95_ci"]
            lines.append(
                f"| {endpoint_name} | {name} | {row['path_mean']:.6f} | "
                f"{row['uncertainty_mean']:.6f} | {row['path_minus_uncertainty_mean']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
                f"{row['path_wins']}/{row['ties']}/{row['uncertainty_wins']} |"
            )
    lines.extend(["", "All p-values in the JSON artifact are descriptive. No PLS test split was evaluated.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bound-run", type=Path, required=True)
    parser.add_argument("--path-run", type=Path, required=True)
    parser.add_argument("--uncertainty-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare(arguments.bound_run, arguments.path_run, arguments.uncertainty_run)
    arguments.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.output_markdown.write_text(markdown(result))


if __name__ == "__main__":
    main()
