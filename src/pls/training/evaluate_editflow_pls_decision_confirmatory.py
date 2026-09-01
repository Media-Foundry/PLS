"""One-pass evaluation of the frozen cached-oracle decision protocol."""

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
from scipy.stats import beta

from pls.editflow.decision_gating import top_m_exact_verification
from pls.editflow.multifidelity import delta_metrics


def _score_map(path: Path) -> dict[str, float]:
    values = np.load(path)
    return {str(key): float(value) for key, value in zip(values["sequence_sha256"], values["logits"])}


def coverage_interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


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
    if config.get("status") != "frozen_before_confirmatory_exact_scores":
        raise ValueError("confirmatory protocol is not frozen")
    manifest = json.loads(Path(config["landscape_manifest"]).read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("confirmatory manifest is not explicitly test-free")
    if any(node["split"] != "train" for node in manifest["nodes"]):
        raise ValueError("confirmatory manifest must be train-only")
    fixed = _score_map(Path(config["fixed_scores"]))
    exact = _score_map(Path(config["exact_scores"]))
    low_delta, exact_delta, anchors = [], [], []
    by_anchor: dict[int, list[int]] = {}
    for edge in manifest["edges"]:
        source = manifest["nodes"][int(edge["source_node"])]
        target = manifest["nodes"][int(edge["target_node"])]
        low_delta.append(fixed[target["sequence_sha256"]] - fixed[source["sequence_sha256"]])
        exact_delta.append(exact[target["sequence_sha256"]] - exact[source["sequence_sha256"]])
        anchor = int(edge["anchor_rank"])
        anchors.append(anchor)
        by_anchor.setdefault(anchor, []).append(len(low_delta) - 1)
    low_delta = np.asarray(low_delta)
    exact_delta = np.asarray(exact_delta)
    anchors = np.asarray(anchors)

    threshold = float(config["primary_method"]["frozen_margin_quantile"])
    records = []
    for anchor, indices_list in sorted(by_anchor.items()):
        indices = np.asarray(indices_list)
        low, truth = low_delta[indices], exact_delta[indices]
        chosen = np.flatnonzero(low >= np.max(low) - threshold - 1e-12)
        exact_best = float(np.max(truth))
        verified_best = float(np.max(truth[chosen]))
        regret = exact_best - verified_best
        records.append({
            "anchor_rank": anchor,
            "exact_queries": int(len(chosen)),
            "exact_fraction": float(len(chosen) / len(indices)),
            "covered": bool(regret <= 1e-12),
            "regret": float(regret),
        })
    successes = sum(record["covered"] for record in records)
    regrets = np.asarray([record["regret"] for record in records])
    queries = np.asarray([record["exact_queries"] for record in records])
    primary = {
        "anchors": len(records),
        "coverage": successes / len(records),
        "coverage_clopper_pearson_95": coverage_interval(successes, len(records)),
        "mean_exact_queries": float(np.mean(queries)),
        "median_exact_queries": float(np.median(queries)),
        "query_range": [int(np.min(queries)), int(np.max(queries))],
        "mean_exact_fraction": float(np.mean([record["exact_fraction"] for record in records])),
        "zero_regret_fraction": float(np.mean(regrets <= 1e-12)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "regret_p90": float(np.quantile(regrets, 0.9)),
        "maximum_regret": float(np.max(regrets)),
    }
    baselines = {}
    for baseline in config["baselines"]:
        name = f"{baseline['name']}_m{int(baseline['m'])}"
        baselines[name] = top_m_exact_verification(
            low_delta, exact_delta, anchors, int(baseline["m"])
        )
    result = {
        "schema": "PLS_EditFlow_decision_gating_confirmatory_report_v1",
        "protocol_status": "evaluated_once_after_frozen_config",
        "primary": primary,
        "baselines": baselines,
        "fixed_parent_field": delta_metrics(exact_delta, low_delta, anchors, top_k=5),
        "anchor_records": records,
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
    lines = [
        "# Cached-oracle decision gating confirmatory v1",
        "",
        "The protocol, threshold, endpoints, and baselines were frozen before these exact scores existed.",
        "",
        "| Method | Exact fraction | Coverage / zero regret | Mean regret | P90 regret |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| conformal margin | {primary['mean_exact_fraction']:.4f} | {primary['coverage']:.4f} | {primary['mean_regret']:.4f} | {primary['regret_p90']:.4f} |",
    ]
    for name, row in baselines.items():
        lines.append(
            f"| {name} | {row['mean_exact_fraction']:.4f} | {row['zero_regret_fraction']:.4f} | "
            f"{row['mean_regret']:.4f} | {row['regret_p90']:.4f} |"
        )
    lines.extend(["", "PLS test queries/evaluations: **0**.", ""])
    report = "\n".join(lines)
    (arguments.run_dir / "validation_metrics.json").write_text(output)
    (arguments.run_dir / "report.md").write_text(report)
    arguments.analysis_json.write_text(output)
    arguments.analysis_md.write_text(report)
    print(output, end="")


if __name__ == "__main__":
    main()
