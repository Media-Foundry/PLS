"""SI-component-aware OOF affine calibration for a frozen eSOL recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pls.evaluation.component_crossfit import (
    regression_component_folds,
    si_component_groups,
)
from pls.evaluation.metrics import regression_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_esol_partial_candidates import frozen_prediction


def affine_fit(prediction: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(prediction, targets, 1)
    return float(slope), float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--entities",
        type=Path,
        default=Path("benchmark/generated/sequence_entities.csv"),
    )
    parser.add_argument(
        "--observation-split",
        type=Path,
        default=Path("benchmark/generated/strict_si30_observation_split.csv"),
    )
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    if source.get("test_evaluated") is not False:
        raise ValueError("source report does not explicitly preserve the test freeze")
    targets, prediction, entity_indices = frozen_prediction(args.report)
    groups = si_component_groups(
        entity_indices, args.entities, args.observation_split
    )
    crossfit = np.zeros_like(prediction, dtype=np.float64)
    fold_reports = []
    for fold, (train_indices, heldout_indices) in enumerate(
        regression_component_folds(targets, groups, args.folds)
    ):
        slope, intercept = affine_fit(
            prediction[train_indices], targets[train_indices]
        )
        heldout = slope * prediction[heldout_indices] + intercept
        crossfit[heldout_indices] = heldout
        fold_reports.append({
            "fold": fold,
            "train_entities": int(len(train_indices)),
            "heldout_entities": int(len(heldout_indices)),
            "train_components": int(len(np.unique(groups[train_indices]))),
            "heldout_components": int(len(np.unique(groups[heldout_indices]))),
            "slope": slope,
            "intercept": intercept,
            "heldout": regression_metrics(targets[heldout_indices], heldout),
        })
    slope, intercept = affine_fit(prediction, targets)
    calibrated = slope * prediction + intercept
    report = {
        "schema": "PLS_eSOL_component_crossfit_affine_v1",
        "selection_data": "strict-validation only",
        "test_evaluated": False,
        "source_report": str(args.report),
        "entity_count": int(len(entity_indices)),
        "si_component_count": int(len(np.unique(groups))),
        "crossfit_grouping": "strict_si30_component_root_sha256",
        "uncalibrated": regression_metrics(targets, prediction),
        "deployment_affine_slope": slope,
        "deployment_affine_intercept": intercept,
        "validation_fitted_affine": regression_metrics(targets, calibrated),
        "crossfit_folds": fold_reports,
        "crossfit_affine": regression_metrics(targets, crossfit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
