#!/usr/bin/env python3
"""Summarize same-protocol GB1 development acquisitions without inferential tests.

The GB1 anchors share a four-site landscape and therefore are not independent
biological replicates.  This script deliberately reports descriptive means and
budget-curve integrals only; it does not attach population-level p-values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REGRET_COMPONENTS = ("acquired", "novel_design", "campaign")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=OUTPUT_DIRECTORY")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--run must be NAME=OUTPUT_DIRECTORY")
    return name, Path(path)


def normalized_trapezoid(xs: list[int], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("AUC requires equally sized x/y arrays with at least two points")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for left_x, right_x, left_y, right_y in zip(xs, xs[1:], ys, ys[1:])
    )
    return area / float(xs[-1] - xs[0])


def load_run(name: str, directory: Path) -> dict[str, Any]:
    metrics_path = directory / "aggregate_metrics.json"
    budget_path = directory / "query_budget.json"
    metrics = json.loads(metrics_path.read_text())
    budget = json.loads(budget_path.read_text())
    if budget.get("test_evaluated") is not False:
        raise ValueError(f"{name}: test_evaluated must be explicitly false")
    budgets = [int(row["budget"]) for row in metrics]
    expected = [int(value) for value in budget["budget_curve"]]
    if budgets != expected:
        raise ValueError(f"{name}: aggregate budgets {budgets} != protocol {expected}")
    if int(budget["anchors"]) != int(metrics[0]["anchors"]):
        raise ValueError(f"{name}: anchor-count mismatch")

    curves: dict[str, dict[str, list[float]]] = {}
    auc: dict[str, dict[str, float]] = {}
    final: dict[str, dict[str, dict[str, float | int]]] = {}
    for radius in (1, 2, 3, 4):
        key = str(radius)
        curves[key] = {}
        auc[key] = {}
        final[key] = {}
        for component in REGRET_COMPONENTS:
            values = [float(row["regret"][key][component]["mean"]) for row in metrics]
            curves[key][component] = values
            auc[key][component] = normalized_trapezoid(budgets, values)
            last = metrics[-1]["regret"][key][component]
            final[key][component] = {
                "mean": float(last["mean"]),
                "count": int(last["count"]),
                "unavailable_fraction": float(last["unavailable_fraction"]),
            }

    return {
        "name": name,
        "directory": str(directory),
        "acquisition": budget["acquisition"],
        "anchor_protocol_sha256": budget["anchor_protocol_sha256"],
        "anchors": int(budget["anchors"]),
        "budgets": budgets,
        "teacher_query_cost_unit": budget["teacher_query_cost_unit"],
        "final_value_r2": float(metrics[-1]["value_r2"]["mean"]),
        "final_edge_spearman": float(metrics[-1]["edge_spearman"]["mean"]),
        "curves": curves,
        "normalized_budget_auc": auc,
        "final_regret": final,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    methods = payload["methods"]
    lines = [
        "# GB1 standard-acquisition development matrix v1",
        "",
        "These are descriptive development results on multiple starting points in the",
        "same four-site GB1 landscape. Anchors are not treated as independent biological",
        "replicates, and no population-level significance test is reported. All methods",
        "share the same initial-query protocol, unique-node budgets, candidate universe,",
        "and oracle cost model. No PLS test split was evaluated.",
        "",
        "## Budget-curve summary",
        "",
        "Normalized AUC is the trapezoidal mean regret over budgets 80--640; lower is",
        "better. The local-design column is novel-design regret at radius 2.",
        "",
        "| acquisition | final R2 | final edge Spearman | novel k=2 AUC | acquired k=2 AUC | campaign k=2 AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        auc = method["normalized_budget_auc"]["2"]
        lines.append(
            f"| {method['name']} | {method['final_value_r2']:.6f} | "
            f"{method['final_edge_spearman']:.6f} | {auc['novel_design']:.6f} | "
            f"{auc['acquired']:.6f} | {auc['campaign']:.6f} |"
        )
    lines.extend([
        "",
        "## Final 640-query regret",
        "",
        "| acquisition | radius | acquired | novel design | campaign | novel available |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method in methods:
        for radius in (1, 2, 3, 4):
            row = method["final_regret"][str(radius)]
            novel = row["novel_design"]
            available = int(round(method["anchors"] * (1.0 - novel["unavailable_fraction"])))
            lines.append(
                f"| {method['name']} | {radius} | {row['acquired']['mean']:.6f} | "
                f"{novel['mean']:.6f} | {row['campaign']['mean']:.6f} | "
                f"{available}/{method['anchors']} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Occupancy-only is strongest for very local acquisition/campaign behavior, while",
        "UCB is more competitive at the full radius. This is development evidence for an",
        "exploitation--coverage trade-off, not evidence that an adaptive policy already",
        "wins. Method selection remains confined to GB1 before an untouched landscape is",
        "opened.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    methods = [load_run(name, path) for name, path in args.run]
    protocols = {method["anchor_protocol_sha256"] for method in methods}
    budgets = {tuple(method["budgets"]) for method in methods}
    cost_units = {method["teacher_query_cost_unit"] for method in methods}
    if len(protocols) != 1 or len(budgets) != 1 or len(cost_units) != 1:
        raise ValueError("runs do not share one anchor protocol, budget curve, and cost model")

    payload = {
        "status": "development_only",
        "test_evaluated": False,
        "statistical_unit_warning": (
            "GB1 anchors share one four-site landscape and are not independent biological replicates."
        ),
        "anchor_protocol_sha256": next(iter(protocols)),
        "budget_curve": list(next(iter(budgets))),
        "teacher_query_cost_unit": next(iter(cost_units)),
        "methods": methods,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(render_markdown(payload))


if __name__ == "__main__":
    main()
