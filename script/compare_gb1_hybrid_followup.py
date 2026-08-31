"""Descriptive comparison for the post-confirmatory 50/50 hybrid follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pls.editflow.statistics import paired_method_summary
from compare_gb1_confirmatory import extract_regret, load_run


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_903


def validate(candidate_config, candidate_history, baseline_config, baseline_history):
    if candidate_config["data"]["acquisition"] != "hybrid_path":
        raise ValueError("candidate run is not hybrid path acquisition")
    if float(candidate_config["data"]["path_fraction"]) != 0.5:
        raise ValueError("candidate run is not the frozen 50/50 portfolio")
    if candidate_config.get("interpretation", {}).get("status") != "exploratory_followup":
        raise ValueError("candidate run is not labelled exploratory")
    for key in ("anchor_protocol", "query_budgets", "initial_query_seed", "edit_radii"):
        if candidate_config["data"][key] != baseline_config["data"][key]:
            raise ValueError(f"data protocol mismatch: {key}")
    if candidate_config["model"] != baseline_config["model"]:
        raise ValueError("student model mismatch")
    candidate_training = dict(candidate_config["training"]);candidate_training.pop("hip_device")
    baseline_training = dict(baseline_config["training"]);baseline_training.pop("hip_device")
    if candidate_training != baseline_training:
        raise ValueError("training protocol mismatch")
    if len(candidate_history) != len(baseline_history):
        raise ValueError("anchor count mismatch")
    for candidate, baseline in zip(candidate_history, baseline_history):
        if candidate["anchor"] != baseline["anchor"]:
            raise ValueError("anchor ordering mismatch")
        if candidate["stages"][0]["queried_nodes_sha256"] != baseline["stages"][0]["queried_nodes_sha256"]:
            raise ValueError("initial queried-node set mismatch")


def endpoint(history, final_only):
    stages = [len(history[0]["stages"]) - 1] if final_only else range(1, len(history[0]["stages"]))
    return extract_regret(history, stages, [1, 2, 3, 4])


def compare(hybrid_run: Path, path_run: Path, uncertainty_run: Path) -> dict:
    hybrid_config, hybrid_history, hybrid_budget = load_run(hybrid_run)
    path_config, path_history, path_budget = load_run(path_run)
    uncertainty_config, uncertainty_history, uncertainty_budget = load_run(uncertainty_run)
    validate(hybrid_config, hybrid_history, path_config, path_history)
    validate(hybrid_config, hybrid_history, uncertainty_config, uncertainty_history)
    if len({hybrid_budget["anchor_protocol_sha256"], path_budget["anchor_protocol_sha256"], uncertainty_budget["anchor_protocol_sha256"]}) != 1:
        raise ValueError("anchor protocol identity mismatch")
    comparisons = {}
    for final_only, endpoint_name in ((False, "query_curve"), (True, "final_640")):
        hybrid_values = endpoint(hybrid_history, final_only)
        comparisons[endpoint_name] = {}
        for name, history in (("path", path_history), ("uncertainty", uncertainty_history)):
            comparisons[endpoint_name][f"hybrid_minus_{name}"] = paired_method_summary(
                hybrid_values,
                endpoint(history, final_only),
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED + int(final_only) * 10 + len(name),
            )
    return {
        "schema": "PLS_EditFlow_GB1_hybrid_followup_comparison_v1",
        "status": "exploratory_posthoc",
        "reason": "The 50/50 portfolio was selected after diagnosing the paired path-aware result.",
        "hybrid_run": str(hybrid_run),
        "path_run": str(path_run),
        "uncertainty_run": str(uncertainty_run),
        "path_fraction": 0.5,
        "same_anchor_protocol": True,
        "same_initial_queried_node_sets": True,
        "same_training_protocol": True,
        "comparisons": comparisons,
        "pvalues_are_descriptive_not_confirmatory": True,
        "test_evaluated": False,
    }


def markdown(result):
    lines = [
        "# GB1 50/50 hybrid follow-up v1",
        "",
        "This is an exploratory post-hoc arm, not a confirmatory result. Negative",
        "paired differences favor hybrid acquisition.",
        "",
        "| endpoint | comparison | hybrid mean | baseline mean | difference | 95% CI | W/T/L |",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-run", type=Path, required=True)
    parser.add_argument("--path-run", type=Path, required=True)
    parser.add_argument("--uncertainty-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare(arguments.hybrid_run, arguments.path_run, arguments.uncertainty_run)
    arguments.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.output_markdown.write_text(markdown(result))


if __name__ == "__main__":
    main()
