"""Fit a label-free monotone ESMFold runtime model from historical logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(paths: list[Path]) -> tuple[list[dict], dict]:
    records = []
    signatures = set()
    for report_index, path in enumerate(paths):
        report = json.loads(path.read_text())
        if report.get("failed") != 0:
            raise ValueError(f"runtime report has failures: {path}")
        checkpoint = tuple(
            (row.get("name"), row.get("sha256"))
            for row in report.get("checkpoint_files", [])
        )
        signature = (
            report.get("accelerator_backend"),
            int(report.get("chunk_size", -1)),
            int(report.get("num_recycles", -1)),
            checkpoint,
        )
        signatures.add(signature)
        successful = [row for row in report["results"] if row.get("status") == "ok"]
        # The first sequence in every fresh process includes compilation and
        # model warm-up.  Startup is budgeted separately at deployment; the
        # gating scale models marginal per-candidate inference cost.
        for order, row in enumerate(successful[1:], start=1):
            records.append({
                "report": report_index,
                "path": str(path),
                "order": order,
                "length": int(row["length"]),
                "seconds": float(row["seconds"]),
            })
    if len(signatures) != 1:
        raise ValueError("runtime reports do not share one ESMFold execution signature")
    backend, chunk_size, num_recycles, checkpoint = next(iter(signatures))
    if backend != "rocm" or not records:
        raise ValueError("expected a nonempty homogeneous ROCm runtime corpus")
    return records, {
        "accelerator_backend": backend,
        "chunk_size": chunk_size,
        "num_recycles": num_recycles,
        "checkpoint_files": [
            {"name": name, "sha256": digest} for name, digest in checkpoint
        ],
    }


def _fit(records: list[dict]) -> tuple[IsotonicRegression, np.ndarray, np.ndarray]:
    by_length: dict[int, list[float]] = defaultdict(list)
    for row in records:
        by_length[int(row["length"])].append(float(row["seconds"]))
    lengths = np.asarray(sorted(by_length), dtype=np.float64)
    medians = np.asarray([np.median(by_length[int(value)]) for value in lengths])
    weights = np.asarray([len(by_length[int(value)]) for value in lengths], dtype=np.float64)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(lengths, medians, sample_weight=weights)
    return model, lengths, medians


def _crossfit(records: list[dict], folds: int) -> dict:
    groups = np.asarray([row["report"] for row in records])
    lengths = np.asarray([row["length"] for row in records], dtype=np.float64)
    seconds = np.asarray([row["seconds"] for row in records], dtype=np.float64)
    prediction = np.empty(len(records), dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    dummy = np.zeros(len(records))
    for train, held_out in splitter.split(dummy, dummy, groups):
        model, _, _ = _fit([records[index] for index in train])
        prediction[held_out] = model.predict(lengths[held_out])
    absolute = np.abs(prediction - seconds)
    relative = absolute / np.maximum(seconds, 1e-9)
    return {
        "grouped_by_runtime_report_folds": folds,
        "spearman": float(spearmanr(prediction, seconds).statistic),
        "median_absolute_error_seconds": float(np.median(absolute)),
        "mean_absolute_error_seconds": float(np.mean(absolute)),
        "median_absolute_percentage_error": float(np.median(relative)),
        "p90_absolute_percentage_error": float(np.quantile(relative, 0.9)),
    }


def _reference_lengths(path: Path) -> list[int]:
    manifest = json.loads(path.read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("reference manifest must explicitly remain test-free")
    return [int(row["length"]) for row in manifest["nodes"] if row["kind"] == "anchor"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--analysis-md", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    arguments = parser.parse_args()

    paths = sorted(arguments.report_root.glob("**/esmfold/shard_*_report.json"))
    records, signature = _load_records(paths)
    fitted, observed_lengths, observed_medians = _fit(records)
    reference_lengths = _reference_lengths(arguments.reference_manifest)
    reference_cost = float(np.median(fitted.predict(np.asarray(reference_lengths))))
    model = {
        "schema": "PLS_ESMFold_monotone_runtime_cost_model_v1",
        "status": "frozen_label_free_before_new_calibration",
        "fit_target": "weighted_isotonic_regression_on_per_length_median_marginal_seconds",
        "warmup_policy": "exclude_first_successful_sequence_per_fresh_shard_process",
        "startup_cost_policy": "budget_separately_from_candidate_specific_cost",
        "hardware_signature": signature,
        "source_reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "runtime_records": len(records),
        "runtime_reports": len(paths),
        "observed_length_range": [int(min(row["length"] for row in records)), int(max(row["length"] for row in records))],
        "length_knots": fitted.X_thresholds_.astype(float).tolist(),
        "seconds_knots": fitted.y_thresholds_.astype(float).tolist(),
        "reference_cost_seconds": reference_cost,
        "reference_cost_source": str(arguments.reference_manifest) + ": median predicted anchor cost",
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    diagnostics = {
        "schema": "PLS_ESMFold_runtime_cost_model_report_v1",
        "model": model,
        "crossfit": _crossfit(records, arguments.folds),
        "observed_per_length_points": len(observed_lengths),
        "observed_per_length_medians": {
            str(int(length)): float(value)
            for length, value in zip(observed_lengths, observed_medians)
        },
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }

    arguments.run_dir.mkdir(parents=True, exist_ok=False)
    arguments.output_model.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    output = json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    arguments.analysis_json.write_text(output)
    (arguments.run_dir / "runtime_cost_model.json").write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    (arguments.run_dir / "validation_metrics.json").write_text(output)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "test_evaluated": False,
    }
    (arguments.run_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Frozen ESMFold runtime cost model v1",
        "",
        f"The label-free model uses {len(records):,} marginal timings from {len(paths)} homogeneous ROCm ESMFold shards. The first successful sequence per process is excluded as warm-up; startup remains a separate deployment cost.",
        "",
        "| Diagnostic | Value |",
        "| --- | ---: |",
        f"| Group-report CV Spearman | {diagnostics['crossfit']['spearman']:.4f} |",
        f"| Median absolute error | {diagnostics['crossfit']['median_absolute_error_seconds']:.4f} s |",
        f"| Median absolute percentage error | {diagnostics['crossfit']['median_absolute_percentage_error']:.2%} |",
        f"| P90 absolute percentage error | {diagnostics['crossfit']['p90_absolute_percentage_error']:.2%} |",
        f"| Frozen reference cost | {reference_cost:.4f} s |",
        "",
        "The model is used only to allocate candidate-evaluation cost. It uses no oracle scores and accesses no PLS test entity.",
        "",
    ]
    report = "\n".join(lines)
    arguments.analysis_md.write_text(report)
    (arguments.run_dir / "report.md").write_text(report)
    print(output, end="")


if __name__ == "__main__":
    main()
