"""Validation-only non-negative weighting of aligned UESolDS models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from pls.evaluation.metrics import binary_metrics


def load(run: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(Path(run) / "validation_predictions.npz")
    return data["targets"], data["logits"], data["entity_indices"]


def initial_weights(report: dict) -> dict[str, float]:
    """Read either a weighted selection report or a legacy ensemble sweep."""
    if "run_weights" in report:
        return {str(run): float(weight) for run, weight in report["run_weights"].items()}
    if "runs" in report and "weights" in report:
        return dict(zip(map(str, report["runs"]), map(float, report["weights"])))
    entries = report["reports"]
    base = max(entries, key=lambda row: row["uncalibrated"]["auroc"])
    return {str(run): 1.0 / len(base["runs"]) for run in base["runs"]}


def objective_score(targets: np.ndarray, logits: np.ndarray, objective: str) -> float:
    if objective == "auroc": return float(roc_auc_score(targets, logits))
    if objective == "auprc": return float(average_precision_score(targets, logits))
    if objective == "brier":
        probability = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        return -float(np.mean((probability - targets) ** 2))
    raise ValueError(objective)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("outputs/validation_selection/uesolds_capacity_weighted_auc.json"),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=[],
        help="additional aligned validation run (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=("auroc", "auprc", "brier"), default="auroc")
    args = parser.parse_args()

    source = json.loads(args.source_report.read_text())
    prior = initial_weights(source)
    runs = sorted(set(prior) | {str(run) for run in args.candidate})
    targets, _, entities = load(runs[0])
    predictions = []
    for run in runs:
        observed, prediction, observed_entities = load(run)
        if not np.array_equal(targets, observed) or not np.array_equal(
            entities, observed_entities
        ):
            raise ValueError(f"validation alignment mismatch: {run}")
        predictions.append(prediction)
    predictions = np.asarray(predictions)

    weights = np.asarray([prior.get(run, 0.0) for run in runs], dtype=float)
    weights /= weights.sum()
    best = objective_score(targets, weights @ predictions, args.objective)
    for _ in range(8):
        changed = False
        for index in range(len(weights)):
            winner = (best, weights.copy())
            for value in np.linspace(0, 0.4, 81):
                proposed = weights.copy()
                other = proposed.sum() - proposed[index]
                if other > 0:
                    proposed *= (1 - value) / other
                    proposed[index] = value
                else:
                    proposed[:] = 0
                    proposed[index] = 1
                score = objective_score(targets, proposed @ predictions, args.objective)
                if score > winner[0] + 1e-12:
                    winner = (score, proposed.copy())
            if winner[0] > best:
                best, weights = winner
                changed = True
        if not changed:
            break

    logits = weights @ predictions

    def nll(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0, scaled) - targets * scaled))

    temperature = float(
        np.exp(minimize_scalar(nll, bounds=(-2, 2), method="bounded").x)
    )
    calibrated = logits / temperature
    probability = 1 / (1 + np.exp(-np.clip(calibrated, -50, 50)))
    thresholds = np.unique(np.quantile(probability, np.linspace(0, 1, 2001)))
    threshold, mcc = max(
        (
            (float(value), float(matthews_corrcoef(targets, probability >= value)))
            for value in thresholds
        ),
        key=lambda row: row[1],
    )
    report = {
        "selection_data": "strict-validation only",
        "test_evaluated": False,
        "objective": args.objective,
        "source_report": str(args.source_report),
        "candidate_runs": [str(run) for run in args.candidate],
        "runs": runs,
        "weights": weights.tolist(),
        "run_weights": dict(zip(runs, weights.tolist())),
        "uncalibrated": binary_metrics(targets, logits),
        "temperature": temperature,
        "calibrated": binary_metrics(targets, calibrated),
        "mcc_threshold": threshold,
        "mcc_at_selected_threshold": mcc,
        "entity_count": int(len(entities)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
