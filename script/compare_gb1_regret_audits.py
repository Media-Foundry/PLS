"""Paired descriptive comparison of final-budget regret decomposition audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pls.editflow.statistics import paired_method_summary


COMPONENTS = ("acquired", "novel_design", "campaign")


def load_audit(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("schema") != "PLS_EditFlow_GB1_final_regret_decomposition_audit_v1":
        raise ValueError("unsupported regret audit schema")
    if report.get("test_evaluated") is not False:
        raise ValueError("comparison refuses an audit without an explicit test freeze")
    return report


def paired_values(first: dict, second: dict, radius: str, component: str):
    first_by_anchor = {
        int(row["anchor"]["node_index"]): row for row in first["anchors"]
    }
    second_by_anchor = {
        int(row["anchor"]["node_index"]): row for row in second["anchors"]
    }
    if first_by_anchor.keys() != second_by_anchor.keys():
        raise ValueError("audits do not share the exact anchor identities")
    first_values = []
    second_values = []
    unavailable = 0
    for anchor in sorted(first_by_anchor):
        first_value = first_by_anchor[anchor]["regret"][radius][component]["regret"]
        second_value = second_by_anchor[anchor]["regret"][radius][component]["regret"]
        if first_value is None or second_value is None:
            unavailable += 1
            continue
        first_values.append(float(first_value))
        second_values.append(float(second_value))
    return np.asarray(first_values), np.asarray(second_values), unavailable


def rename_summary(summary: dict, first_name: str, second_name: str) -> dict:
    return {
        "pairs": summary["pairs"],
        "first_method": first_name,
        "second_method": second_name,
        "first_mean": summary["path_mean"],
        "second_mean": summary["uncertainty_mean"],
        "first_minus_second_mean": summary["path_minus_uncertainty_mean"],
        "first_minus_second_median": summary["path_minus_uncertainty_median"],
        "bootstrap_95_ci": summary["bootstrap_95_ci"],
        "exact_two_sided_sign_flip_pvalue": summary[
            "exact_two_sided_sign_flip_pvalue"
        ],
        "first_wins": summary["path_wins"],
        "ties": summary["ties"],
        "second_wins": summary["uncertainty_wins"],
        "direction": "negative_favors_first_method",
    }


def compare(
    first_path: Path,
    second_path: Path,
    first_name: str,
    second_name: str,
    analysis_status: str,
) -> dict:
    first = load_audit(first_path)
    second = load_audit(second_path)
    if first["final_budget"] != second["final_budget"]:
        raise ValueError("final query budgets differ")
    radii = sorted(first["aggregate"], key=int)
    if radii != sorted(second["aggregate"], key=int):
        raise ValueError("audits use different edit radii")
    cells = {}
    for radius in radii:
        cells[radius] = {}
        for component in COMPONENTS:
            first_values, second_values, unavailable = paired_values(
                first, second, radius, component
            )
            if not len(first_values):
                cells[radius][component] = {
                    "pairs": 0,
                    "unavailable_pairs": unavailable,
                }
                continue
            summary = paired_method_summary(
                first_values,
                second_values,
                bootstrap_samples=20_000,
                bootstrap_seed=20_260_831 + int(radius) * 101 + COMPONENTS.index(component),
            )
            cells[radius][component] = {
                **rename_summary(summary, first_name, second_name),
                "unavailable_pairs": unavailable,
            }
    return {
        "schema": "PLS_EditFlow_GB1_regret_decomposition_paired_audit_v1",
        "first_audit": str(first_path),
        "second_audit": str(second_path),
        "first_method": first_name,
        "second_method": second_name,
        "final_budget": first["final_budget"],
        "analysis_status": analysis_status,
        "multiple_comparisons_adjusted": False,
        "cells": cells,
        "test_evaluated": False,
    }


def markdown(report: dict) -> str:
    lines = [
        "# GB1 final-budget regret decomposition",
        "",
        f"Status: `{report['analysis_status']}`. Negative differences favor "
        f"`{report['first_method']}` over `{report['second_method']}`.",
        "",
        "| radius | regret | first | second | difference | 95% CI | exact p | W/T/L |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for radius, components in report["cells"].items():
        for component, row in components.items():
            if not row.get("pairs"):
                lines.append(f"| {radius} | {component} | NA | NA | NA | NA | NA | NA |")
                continue
            interval = row["bootstrap_95_ci"]
            lines.append(
                f"| {radius} | {component} | {row['first_mean']:.6f} | "
                f"{row['second_mean']:.6f} | {row['first_minus_second_mean']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
                f"{row['exact_two_sided_sign_flip_pvalue']:.6g} | "
                f"{row['first_wins']}/{row['ties']}/{row['second_wins']} |"
            )
    lines.extend([
        "",
        "All cells are post-hoc descriptive, unadjusted for multiple comparisons,",
        "and do not create a confirmatory claim. No PLS test split was evaluated.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-name", required=True)
    parser.add_argument("--second-name", required=True)
    parser.add_argument(
        "--analysis-status",
        choices=("secondary_descriptive", "exploratory_posthoc"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args()
    report = compare(
        arguments.first,
        arguments.second,
        arguments.first_name,
        arguments.second_name,
        arguments.analysis_status,
    )
    arguments.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    arguments.output_markdown.write_text(markdown(report))


if __name__ == "__main__":
    main()
