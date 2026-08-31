"""Prespecified paired comparison for unseen-anchor hybrid confirmation v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pls.editflow.statistics import paired_method_summary
from compare_gb1_confirmatory import extract_regret, load_run


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_911


def validate_pair(hybrid_config, hybrid_history, hybrid_budget, path_config, path_history, path_budget):
    if hybrid_config["data"]["acquisition"] != "hybrid_path":
        raise ValueError("first run is not hybrid path acquisition")
    if float(hybrid_config["data"]["path_fraction"]) != 0.5:
        raise ValueError("hybrid path fraction differs from the frozen 50/50 protocol")
    if path_config["data"]["acquisition"] != "path_aware":
        raise ValueError("second run is not pure path acquisition")
    if hybrid_config.get("interpretation", {}).get("status") != "confirmatory_v2":
        raise ValueError("hybrid run is not labelled confirmatory v2")
    if path_config.get("interpretation", {}).get("status") != "confirmatory_v2":
        raise ValueError("path run is not labelled confirmatory v2")
    for key in ("anchor_protocol", "query_budgets", "initial_query_seed", "edit_radii"):
        if hybrid_config["data"][key] != path_config["data"][key]:
            raise ValueError(f"data protocol mismatch: {key}")
    if hybrid_config["model"] != path_config["model"]:
        raise ValueError("student model mismatch")
    hybrid_training = dict(hybrid_config["training"]);hybrid_training.pop("hip_device")
    path_training = dict(path_config["training"]);path_training.pop("hip_device")
    if hybrid_training != path_training:
        raise ValueError("training protocol mismatch")
    if hybrid_budget["anchor_protocol_sha256"] != path_budget["anchor_protocol_sha256"]:
        raise ValueError("anchor protocol identity mismatch")
    if len(hybrid_history) != 16 or len(path_history) != 16:
        raise ValueError("confirmatory v2 requires exactly 16 paired anchors")
    for hybrid, path in zip(hybrid_history, path_history):
        if hybrid["anchor"] != path["anchor"]:
            raise ValueError("anchor order mismatch")
        if hybrid["initial_query_seed"] != path["initial_query_seed"]:
            raise ValueError("initial query seed mismatch")
        if hybrid["ensemble_seeds"] != path["ensemble_seeds"]:
            raise ValueError("ensemble seed mismatch")
        if hybrid["stages"][0]["queried_nodes_sha256"] != path["stages"][0]["queried_nodes_sha256"]:
            raise ValueError("initial queried-node set mismatch")


def compare(hybrid_run: Path, path_run: Path) -> dict:
    hybrid_config, hybrid_history, hybrid_budget = load_run(hybrid_run)
    path_config, path_history, path_budget = load_run(path_run)
    validate_pair(
        hybrid_config, hybrid_history, hybrid_budget,
        path_config, path_history, path_budget,
    )
    radii = [1, 2, 3, 4]
    stage_count = len(hybrid_history[0]["stages"])
    primary = paired_method_summary(
        extract_regret(hybrid_history, range(1, stage_count), radii),
        extract_regret(path_history, range(1, stage_count), radii),
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    primary.update({
        "metric": "anchor mean exact regret across budgets 160/320/640 and radii 1/2/3/4",
        "prespecified_before_run_start": True,
        "comparison": "50/50 hybrid minus pure path",
    })
    final = paired_method_summary(
        extract_regret(hybrid_history, [stage_count - 1], radii),
        extract_regret(path_history, [stage_count - 1], radii),
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED + 1,
    )
    final.update({
        "metric": "anchor mean exact regret across radii 1/2/3/4 at 640 queries",
        "prespecified_secondary": True,
        "comparison": "50/50 hybrid minus pure path",
    })
    return {
        "schema": "PLS_EditFlow_GB1_hybrid_confirmatory_v2",
        "hybrid_run": str(hybrid_run),
        "path_run": str(path_run),
        "anchor_protocol_sha256": hybrid_budget["anchor_protocol_sha256"],
        "validation": {
            "unseen_from_v1": True,
            "same_anchor_protocol": True,
            "same_initial_queried_node_sets": True,
            "same_model_and_seeds": True,
            "same_unique_node_budgets": True,
            "test_evaluated": False,
        },
        "inference": {
            "primary_endpoint_count": 1,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "exact_two_sided_sign_flip_test": True,
        },
        "primary": primary,
        "final_budget_secondary": final,
        "test_evaluated": False,
    }


def markdown(result):
    lines = [
        "# GB1 50/50 hybrid confirmatory v2",
        "",
        "Negative paired differences favor the 50/50 hybrid.",
        "",
        "| endpoint | hybrid mean | path mean | paired difference | bootstrap 95% CI | exact p | W/T/L |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (("Primary query-curve regret", result["primary"]), ("Secondary final-640 regret", result["final_budget_secondary"])):
        interval = row["bootstrap_95_ci"]
        lines.append(
            f"| {label} | {row['path_mean']:.6f} | {row['uncertainty_mean']:.6f} | "
            f"{row['path_minus_uncertainty_mean']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{row['exact_two_sided_sign_flip_pvalue']:.6g} | "
            f"{row['path_wins']}/{row['ties']}/{row['uncertainty_wins']} |"
        )
    lines.extend(["", "The primary endpoint and all protocol identities were committed before either run started. No PLS test split was evaluated.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-run", type=Path, required=True)
    parser.add_argument("--path-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare(arguments.hybrid_run, arguments.path_run)
    arguments.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.output_markdown.write_text(markdown(result))


if __name__ == "__main__":
    main()
