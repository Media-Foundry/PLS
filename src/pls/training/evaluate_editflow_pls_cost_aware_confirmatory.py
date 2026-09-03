"""One-pass confirmatory evaluation of cost-aware conformal gating.

Evaluates exactly the preregistered selection against the exhaustive exact
oracle, and charges every policy and baseline its measured ESMFold seconds rather
than its query count. Run once. Do not tune anything from its output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from pls.editflow.decision_gating import empirical_upper_cvar, regret_summary

TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_map(path: Path) -> dict[str, float]:
    values = np.load(path)
    return {str(k): float(v) for k, v in zip(values["sequence_sha256"], values["logits"])}


def coverage_interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def measured_seconds(reports: list[dict]) -> dict[str, float]:
    seconds: dict[str, float] = {}
    for report in reports:
        for row in report["results"]:
            if row["status"] == "ok":
                seconds[row["sequence_sha256"]] = float(row["seconds"])
    return seconds


def summarize(regrets: np.ndarray, covered: np.ndarray, cost: np.ndarray,
              exhaustive_cost: float, queries: np.ndarray, total_candidates: int) -> dict:
    successes = int(covered.sum())
    total = int(covered.size)
    stats = regret_summary(regrets)
    return {
        "components": total,
        "coverage": successes / total,
        "coverage_clopper_pearson_95": coverage_interval(successes, total),
        "misses": total - successes,
        "mean_exact_queries": float(queries.mean()),
        "median_exact_queries": float(np.median(queries)),
        "query_range": [int(queries.min()), int(queries.max())],
        "exact_query_fraction": float(queries.sum() / total_candidates),
        "measured_gpu_seconds": float(cost.sum()),
        "measured_gpu_cost_fraction": float(cost.sum() / exhaustive_cost),
        "mean_regret": stats["mean_regret"],
        "median_regret": stats["median_regret"],
        "maximum_regret": stats["maximum_regret"],
        "regret_cvar95": stats["regret_cvar95"],
        "failure_conditional_mean_regret": stats["failure_conditional_mean_regret"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selected-cost-report", type=Path, required=True)
    parser.add_argument("--exact-scores", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--analysis-md", type=Path, required=True)
    arguments = parser.parse_args()

    protocol = json.loads(arguments.protocol.read_text())
    selection = json.loads(arguments.selection.read_text())
    if protocol.get("test_evaluated") is not False:
        raise ValueError("test evaluation is permanently disabled")
    if selection.get("status") != "selected_before_any_confirmatory_mutant_fold":
        raise ValueError("selection was not persisted before folding")
    if selection.get("protocol_sha256") != sha256(arguments.protocol):
        raise ValueError("selection was produced under a different protocol revision")

    manifest = json.loads(Path(selection["confirmatory_manifest"]).read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("confirmatory manifest is not explicitly test-free")
    if any(node["split"] != "train" for node in manifest["nodes"]):
        raise ValueError("confirmatory manifest must be train-only")
    nodes = manifest["nodes"]

    fixed = score_map(Path(selection["fixed_scores"]))
    exact = score_map(arguments.exact_scores)

    root = Path(selection["fixed_scores"]).parents[2]
    selected_reports = [
        json.loads(p.read_text())
        for p in sorted((root / "exact_selected" / "esmfold").glob("shard_*_report.json"))
    ]
    full_reports = [
        json.loads(p.read_text())
        for p in sorted((root / "exact_full" / "esmfold").glob("shard_*_report.json"))
    ]
    seconds = measured_seconds(selected_reports) | measured_seconds(full_reports)

    by_anchor: dict[int, dict] = {}
    for edge in manifest["edges"]:
        source = nodes[int(edge["source_node"])]
        target = nodes[int(edge["target_node"])]
        row = by_anchor.setdefault(int(edge["anchor_rank"]), {
            "component": source["component_root_sha256"],
            "length": int(source["length"]),
            "targets": [], "low": [], "exact": [], "node_index": [],
        })
        digest = target["sequence_sha256"]
        row["targets"].append(digest)
        row["node_index"].append(int(edge["target_node"]))
        row["low"].append(fixed[digest] - fixed[source["sequence_sha256"]])
        row["exact"].append(exact[digest] - exact[source["sequence_sha256"]])
    anchors = [by_anchor[k] for k in sorted(by_anchor)]
    for row in anchors:
        row["low"] = np.asarray(row["low"], dtype=np.float64)
        row["exact"] = np.asarray(row["exact"], dtype=np.float64)
        row["seconds"] = np.asarray([seconds[d] for d in row["targets"]], dtype=np.float64)

    total_candidates = sum(len(r["low"]) for r in anchors)
    exhaustive_cost = float(sum(r["seconds"].sum() for r in anchors))
    lengths = np.asarray([r["length"] for r in anchors], dtype=np.float64)

    results = {}
    per_anchor_records = {}
    for policy in selection["policies"]:
        epsilon = float(policy["epsilon"])
        regrets, covered, cost, queries, records = [], [], [], [], []
        for row, chosen in zip(anchors, policy["selections"]):
            indices = np.asarray(chosen["selected_local_indices"], dtype=int)
            best = float(row["exact"].max())
            achieved = float(row["exact"][indices].max())
            regret = best - achieved
            regrets.append(regret)
            covered.append(bool(row["exact"][indices].max() >= best - epsilon - TOL))
            cost.append(float(row["seconds"][indices].sum()))
            queries.append(len(indices))
            records.append({
                "component": row["component"],
                "length": row["length"],
                "exact_queries": int(len(indices)),
                "measured_gpu_seconds": float(row["seconds"][indices].sum()),
                "exhaustive_gpu_seconds": float(row["seconds"].sum()),
                "regret": regret,
                "covered": covered[-1],
            })
        results[policy["policy_id"]] = {
            "role": policy["role"],
            "alpha": float(policy["alpha"]),
            "epsilon": epsilon,
            "gamma": float(policy["gamma"]),
            "quantile": float(policy["quantile"]),
            **summarize(
                np.asarray(regrets), np.asarray(covered), np.asarray(cost),
                exhaustive_cost, np.asarray(queries), total_candidates,
            ),
        }
        per_anchor_records[policy["policy_id"]] = records

    # Fixed-budget baselines charged at measured cost, exact-best coverage only.
    for label, budget in (("top_4", 4), ("top_8", 8), ("rank_12", 12)):
        regrets, covered, cost, queries = [], [], [], []
        for row in anchors:
            order = np.argsort(-row["low"], kind="stable")[:budget]
            best = float(row["exact"].max())
            achieved = float(row["exact"][order].max())
            regrets.append(best - achieved)
            covered.append(achieved >= best - TOL)
            cost.append(float(row["seconds"][order].sum()))
            queries.append(len(order))
        results[label] = {
            "role": "fixed_budget_baseline",
            "epsilon": 0.0,
            **summarize(
                np.asarray(regrets), np.asarray(covered), np.asarray(cost),
                exhaustive_cost, np.asarray(queries), total_candidates,
            ),
        }
    results["exhaustive"] = {
        "role": "reference",
        "epsilon": 0.0,
        "components": len(anchors),
        "coverage": 1.0,
        "coverage_clopper_pearson_95": [1.0, 1.0],
        "misses": 0,
        "mean_exact_queries": float(total_candidates / len(anchors)),
        "median_exact_queries": float(total_candidates / len(anchors)),
        "query_range": [16, 16],
        "exact_query_fraction": 1.0,
        "measured_gpu_seconds": exhaustive_cost,
        "measured_gpu_cost_fraction": 1.0,
        "mean_regret": 0.0, "median_regret": 0.0, "maximum_regret": 0.0,
        "regret_cvar95": 0.0, "failure_conditional_mean_regret": 0.0,
    }

    primary_id = protocol["primary_policy_id"]
    strata = []
    for low, high, label in [
        (0.0, 106.5, "<=106"), (106.5, 148.5, "107-148"),
        (148.5, 219.5, "149-219"), (219.5, float("inf"), ">=220"),
    ]:
        mask = (lengths > low) & (lengths <= high)
        if not mask.any():
            continue
        rows = [r for r, keep in zip(per_anchor_records[primary_id], mask) if keep]
        strata.append({
            "length_stratum": label,
            "components": len(rows),
            "coverage": float(np.mean([r["covered"] for r in rows])),
            "mean_exact_queries": float(np.mean([r["exact_queries"] for r in rows])),
            "measured_gpu_cost_fraction": float(
                sum(r["measured_gpu_seconds"] for r in rows)
                / sum(r["exhaustive_gpu_seconds"] for r in rows)
            ),
            "mean_regret": float(np.mean([r["regret"] for r in rows])),
        })

    cost_report = json.loads(arguments.selected_cost_report.read_text())
    output = {
        "schema": "PLS_cost_aware_conformal_confirmatory_result_v2",
        "status": "one_pass_confirmatory_evaluation",
        "protocol_sha256": sha256(arguments.protocol),
        "selection_artifact_sha256": sha256(arguments.selection),
        "exact_scores_sha256": sha256(arguments.exact_scores),
        "primary_policy_id": primary_id,
        "components": len(anchors),
        "candidates_per_component": int(len(anchors[0]["low"])),
        "primary_endpoint": {
            "definition": (
                "measured selected-stage ESMFold GPU-seconds divided by measured "
                "retrospective exhaustive ESMFold GPU-seconds"
            ),
            "measured_selected_gpu_seconds": float(
                cost_report["measured"]["selected_esmfold_gpu_seconds_total"]
            ),
            "measured_exhaustive_gpu_seconds": exhaustive_cost,
            "measured_gpu_cost_fraction": results[primary_id]["measured_gpu_cost_fraction"],
            "risk_target": "component-level marginal probability of zero exact decision regret at least 0.9",
            "observed_coverage": results[primary_id]["coverage"],
            "coverage_clopper_pearson_95": results[primary_id]["coverage_clopper_pearson_95"],
        },
        "policies_and_baselines": results,
        "primary_policy_length_strata": strata,
        "interpretation": (
            "component-level marginal risk control under SI30-component "
            "exchangeability; not a per-protein certificate"
        ),
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.analysis_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    order = [primary_id, "tolerant_runtime_gamma1", "top_4", "top_8", "rank_12", "exhaustive"]
    lines = [
        "# Cost-aware conformal gating: confirmatory v2 result",
        "",
        "One-pass evaluation of the preregistered selection against the exhaustive",
        "exact oracle on 128 fresh SI30 components. Every policy and baseline is",
        "charged its **measured** ESMFold seconds, not its query count.",
        "",
        f"- components: {output['components']}, candidates each: {output['candidates_per_component']}",
        f"- measured exhaustive ESMFold GPU seconds: {exhaustive_cost:.1f}",
        "- test sequences queried: 0",
        "",
        "## Primary endpoint",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| measured selected-stage GPU seconds | {output['primary_endpoint']['measured_selected_gpu_seconds']:.1f} |",
        f"| measured exhaustive GPU seconds | {exhaustive_cost:.1f} |",
        f"| measured GPU cost fraction | {results[primary_id]['measured_gpu_cost_fraction']:.4f} |",
        f"| exact-best coverage | {results[primary_id]['coverage']:.4f} |",
        f"| coverage 95% Clopper-Pearson | [{results[primary_id]['coverage_clopper_pearson_95'][0]:.4f}, {results[primary_id]['coverage_clopper_pearson_95'][1]:.4f}] |",
        "",
        "## Policies and baselines at measured cost",
        "",
        "| Method | Coverage | Mean queries | Query fraction | Measured cost fraction | Mean regret | CVaR95 | Max regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in order:
        r = results[key]
        lines.append(
            f"| `{key}` | {r['coverage']:.4f} | {r['mean_exact_queries']:.2f}/16 | "
            f"{r['exact_query_fraction']:.4f} | {r['measured_gpu_cost_fraction']:.4f} | "
            f"{r['mean_regret']:.4f} | {r['regret_cvar95']:.4f} | {r['maximum_regret']:.4f} |"
        )
    lines += [
        "",
        "Coverage for the tolerant policy is its epsilon-optimal event; every other",
        "row reports exact-best inclusion.",
        "",
        "## Primary policy by predefined length stratum",
        "",
        "Descriptive only. The finite-sample guarantee is marginal over SI30",
        "components and does not hold separately within a bin.",
        "",
        "| Length | Components | Coverage | Mean queries | Measured cost fraction | Mean regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in strata:
        lines.append(
            f"| {s['length_stratum']} | {s['components']} | {s['coverage']:.4f} | "
            f"{s['mean_exact_queries']:.2f} | {s['measured_gpu_cost_fraction']:.4f} | "
            f"{s['mean_regret']:.4f} |"
        )
    lines += [
        "",
        "## Reading this correctly",
        "",
        "The guarantee is component-level marginal risk control under SI30-component",
        "exchangeability. It is not a per-protein certificate, and a miss on any single",
        "protein is compatible with the stated coverage. Report the measured cost",
        "fraction, never the query fraction, as the compute saving.",
        "",
    ]
    arguments.analysis_md.write_text("\n".join(lines))
    print(json.dumps(output["primary_endpoint"], indent=2, sort_keys=True))
    for key in order:
        r = results[key]
        print(
            f"{key:>28}  coverage={r['coverage']:.4f}  queries={r['mean_exact_queries']:.2f}"
            f"  measured_cost={r['measured_gpu_cost_fraction']:.4f}  mean_regret={r['mean_regret']:.4f}"
        )


if __name__ == "__main__":
    main()
