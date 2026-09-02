"""Replay measured costs and tail risks for a frozen decision-gating result.

This audit purchases no oracle labels and performs no model inference.  It maps
the already-frozen candidate sets onto per-sequence timings recorded by the
completed ESMFold shards.  Counterfactual wall times are explicitly labelled as
replay estimates because the original execution folded all candidates.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

from pls.editflow.decision_gating import margin_candidate_indices, regret_summary


def _score_map(path: Path) -> dict[str, float]:
    values = np.load(path)
    return {
        str(key): float(value)
        for key, value in zip(values["sequence_sha256"], values["logits"])
    }


def _load_fold_timings(root: Path) -> tuple[dict[str, tuple[int, float]], list[dict]]:
    reports = [json.loads(path.read_text()) for path in sorted(root.glob("shard_*_report.json"))]
    if not reports or any(report.get("failed") for report in reports):
        raise ValueError("complete zero-failure ESMFold shard reports are required")
    timings: dict[str, tuple[int, float]] = {}
    for shard, report in enumerate(reports):
        for row in report["results"]:
            if row["status"] != "ok":
                raise ValueError("cost replay requires newly measured successful folds")
            digest = str(row["sequence_sha256"])
            if digest in timings:
                raise ValueError(f"duplicate fold timing: {digest}")
            timings[digest] = (shard, float(row["seconds"]))
    return timings, reports


def _load_structure_seconds(path: Path) -> dict[str, float]:
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok":
                result[str(row["sequence_sha256"])] = float(row.get("seconds", 0.0))
    return result


def _method_record(
    *,
    chosen: list[str],
    regrets: list[float],
    candidate_lengths: dict[str, int],
    fold_timings: dict[str, tuple[int, float]],
    fold_reports: list[dict],
    structure_seconds: dict[str, float],
    exhaustive_digests: list[str],
) -> dict:
    selected = set(chosen)
    if len(selected) != len(chosen):
        raise ValueError("candidate selections must be unique across decision groups")
    shard_inference = [0.0] * len(fold_reports)
    for digest in chosen:
        shard, seconds = fold_timings[digest]
        shard_inference[shard] += seconds
    full_shard_inference = [
        sum(float(row["seconds"]) for row in report["results"])
        for report in fold_reports
    ]
    startup = [
        float(report["elapsed_seconds"]) - full_shard_inference[index]
        for index, report in enumerate(fold_reports)
    ]
    inference_seconds = float(sum(shard_inference))
    exhaustive_inference = float(sum(full_shard_inference))
    replay_wall = float(max(value + startup[i] for i, value in enumerate(shard_inference)))
    measured_full_wall = float(max(report["elapsed_seconds"] for report in fold_reports))
    selected_residues = int(sum(candidate_lengths[digest] for digest in chosen))
    exhaustive_residues = int(sum(candidate_lengths[digest] for digest in exhaustive_digests))
    selected_structure_seconds = float(sum(structure_seconds.get(digest, 0.0) for digest in chosen))
    exhaustive_structure_seconds = float(
        sum(structure_seconds.get(digest, 0.0) for digest in exhaustive_digests)
    )
    return {
        "exact_queries": len(chosen),
        "exact_query_fraction": len(chosen) / len(exhaustive_digests),
        "selected_residues": selected_residues,
        "residue_fraction": selected_residues / exhaustive_residues,
        "esmfold_inference_gpu_seconds": inference_seconds,
        "esmfold_inference_gpu_fraction": inference_seconds / exhaustive_inference,
        "esmfold_inference_gpu_saving_fraction": 1.0 - inference_seconds / exhaustive_inference,
        "esmfold_replay_four_gpu_wall_seconds_including_recorded_startup": replay_wall,
        "esmfold_replay_wall_fraction": replay_wall / measured_full_wall,
        "esmfold_replay_wall_saving_fraction": 1.0 - replay_wall / measured_full_wall,
        "exact_v4_per_record_cpu_seconds_sum": selected_structure_seconds,
        "exact_v4_per_record_cpu_fraction": (
            selected_structure_seconds / exhaustive_structure_seconds
            if exhaustive_structure_seconds else None
        ),
        **regret_summary(regrets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--analysis-md", type=Path, required=True)
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text())
    if config.get("evaluate_test", False) or config.get("test_evaluated") is not False:
        raise ValueError("test evaluation is permanently disabled")
    manifest = json.loads(Path(config["landscape_manifest"]).read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("manifest must explicitly remain test-free")
    if any(node["split"] != "train" for node in manifest["nodes"]):
        raise ValueError("cost audit accepts train-only confirmatory manifests")

    fixed = _score_map(Path(config["fixed_scores"]))
    exact = _score_map(Path(config["exact_scores"]))
    fold_timings, fold_reports = _load_fold_timings(arguments.fold_root)
    structure_seconds = _load_structure_seconds(arguments.structure_manifest)

    by_anchor: dict[int, list[dict]] = {}
    candidate_lengths: dict[str, int] = {}
    for edge in manifest["edges"]:
        source = manifest["nodes"][int(edge["source_node"])]
        target = manifest["nodes"][int(edge["target_node"])]
        digest = str(target["sequence_sha256"])
        candidate_lengths[digest] = int(target["length"])
        by_anchor.setdefault(int(edge["anchor_rank"]), []).append({
            "digest": digest,
            "low": fixed[digest] - fixed[str(source["sequence_sha256"])],
            "exact": exact[digest] - exact[str(source["sequence_sha256"])],
        })

    exhaustive_digests = [row["digest"] for rows in by_anchor.values() for row in rows]
    if set(exhaustive_digests) != set(fold_timings):
        raise ValueError("fold reports and confirmatory mutant candidates do not match")

    selections: dict[str, tuple[list[str], list[float]]] = {}
    threshold = float(config["primary_method"]["frozen_margin_quantile"])
    chosen, regrets = [], []
    for rows in by_anchor.values():
        low = np.asarray([row["low"] for row in rows])
        indices = margin_candidate_indices(low, threshold)
        chosen.extend(rows[i]["digest"] for i in indices)
        exact_values = np.asarray([row["exact"] for row in rows])
        regrets.append(float(np.max(exact_values) - np.max(exact_values[indices])))
    selections["conformal_margin"] = (chosen, regrets)

    for baseline in config["baselines"]:
        name = f"{baseline['name']}_m{int(baseline['m'])}"
        chosen, regrets = [], []
        for rows in by_anchor.values():
            order = np.argsort(-np.asarray([row["low"] for row in rows]), kind="stable")
            indices = order[: min(int(baseline["m"]), len(rows))]
            chosen.extend(rows[i]["digest"] for i in indices)
            exact_values = np.asarray([row["exact"] for row in rows])
            regrets.append(float(np.max(exact_values) - np.max(exact_values[indices])))
        selections[name] = (chosen, regrets)

    methods = {
        name: _method_record(
            chosen=chosen,
            regrets=regrets,
            candidate_lengths=candidate_lengths,
            fold_timings=fold_timings,
            fold_reports=fold_reports,
            structure_seconds=structure_seconds,
            exhaustive_digests=exhaustive_digests,
        )
        for name, (chosen, regrets) in selections.items()
    }
    result = {
        "schema": "PLS_EditFlow_decision_gating_confirmatory_cost_audit_v1",
        "status": "posthoc_deterministic_replay_of_frozen_candidate_sets",
        "frozen_threshold_unchanged": threshold,
        "exhaustive_measured": {
            "exact_queries": len(exhaustive_digests),
            "residues": int(sum(candidate_lengths.values())),
            "esmfold_inference_gpu_seconds": float(sum(seconds for _, seconds in fold_timings.values())),
            "esmfold_four_gpu_wall_seconds_including_startup": float(
                max(report["elapsed_seconds"] for report in fold_reports)
            ),
            "exact_v4_wall_seconds": float(
                json.loads((arguments.structure_manifest.parent / "extraction_summary.json").read_text())[
                    "elapsed_seconds"
                ]
            ),
        },
        "methods": methods,
        "cost_scope": {
            "measured": [
                "per-sequence ESMFold inference seconds",
                "four-shard startup and full wall seconds",
                "per-record exact V4 extraction CPU seconds",
            ],
            "shared_and_not_saved": [
                "exact-sequence PLM extraction required by the fixed-parent oracle",
                "cached parent structures and parent feature materialization",
                "fixed-parent scoring over all candidate mutations",
            ],
            "not_replayable_from_current_logs": [
                "selected-only exact geometry/vector/patch postprocessing wall time",
                "selected-only exact PLS scoring wall time",
            ],
            "wall_time_warning": (
                "Selected-method wall seconds are counterfactual replay estimates using the original "
                "four shard assignments and recorded per-shard startup overhead; only exhaustive wall "
                "time was directly measured."
            ),
        },
        "quantile_provenance": {
            "confirmatory_v1": "frozen conservative one-order-higher threshold; not recomputed",
            "future_protocols": "direct ceil((n+1)*(1-alpha)) order statistic",
        },
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"

    lines = [
        "# Confirmatory decision-gating cost and tail-risk audit",
        "",
        "This is a deterministic post-hoc replay of the frozen candidate sets; no oracle was queried and the frozen threshold was not recomputed.",
        "",
        "| Method | Queries | ESMFold GPU-s | GPU saving | Replayed 4-GPU wall | Wall saving | Coverage | Mean regret | Failure mean | CVaR95 | Max regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in methods.items():
        lines.append(
            f"| {name} | {row['exact_queries']} | {row['esmfold_inference_gpu_seconds']:.1f} | "
            f"{row['esmfold_inference_gpu_saving_fraction']:.1%} | "
            f"{row['esmfold_replay_four_gpu_wall_seconds_including_recorded_startup']:.1f}s | "
            f"{row['esmfold_replay_wall_saving_fraction']:.1%} | "
            f"{row['zero_regret_fraction']:.4f} | {row['mean_regret']:.4f} | "
            f"{row['failure_conditional_mean_regret']:.4f} | {row['regret_cvar95']:.4f} | "
            f"{row['maximum_regret']:.4f} |"
        )
    lines.extend([
        "",
        "ESMFold GPU-seconds are measured per sequence. Selected-method wall times are counterfactual replays on the original four shard assignments; exhaustive wall time alone was directly measured.",
        "Exact-sequence PLM extraction and fixed-parent scoring are shared upfront work and therefore are not counted as savings. Selected-only exact geometry/patch/scoring wall time was not instrumented in v1.",
        "",
        "PLS test queries/evaluations: **0**.",
        "",
    ])
    report = "\n".join(lines)
    arguments.run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(arguments.config, arguments.run_dir / "config.json")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "test_evaluated": False,
    }
    (arguments.run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    (arguments.run_dir / "validation_metrics.json").write_text(output)
    (arguments.run_dir / "report.md").write_text(report)
    arguments.analysis_json.write_text(output)
    arguments.analysis_md.write_text(report)
    print(output, end="")


if __name__ == "__main__":
    main()
