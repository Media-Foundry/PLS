"""Endpoint-specific solubility metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, roc_auc_score)


def binary_metrics(targets, logits, bins: int = 10, sample_weight=None):
    targets = np.asarray(targets, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    predictions = probabilities >= .5
    sample_weight = np.ones(len(targets), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    sample_weight = sample_weight / sample_weight.sum()
    ece = 0.0
    boundaries = np.linspace(0, 1, bins + 1)
    for index in range(bins):
        selected = ((probabilities >= boundaries[index]) &
                    (probabilities < boundaries[index + 1] if index < bins - 1 else probabilities <= 1))
        if np.any(selected):
            local_weight = sample_weight[selected];mass=local_weight.sum();local_weight=local_weight/mass
            ece += mass * abs(np.sum(probabilities[selected]*local_weight) - np.sum(targets[selected]*local_weight))
    return {
        "n": int(len(targets)), "auroc": float(roc_auc_score(targets, probabilities, sample_weight=sample_weight)),
        "auprc": float(average_precision_score(targets, probabilities, sample_weight=sample_weight)),
        "mcc": float(matthews_corrcoef(targets, predictions, sample_weight=sample_weight)),
        "f1": float(f1_score(targets, predictions, sample_weight=sample_weight)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions, sample_weight=sample_weight)),
        "brier": float(np.sum(sample_weight*(probabilities-targets)**2)), "ece": float(ece),
    }


def regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    difference = predictions - targets
    return {
        "n": int(len(targets)), "pearson": float(pearsonr(targets, predictions).statistic),
        "spearman": float(spearmanr(targets, predictions).statistic),
        "rmse": float(np.sqrt(np.mean(difference ** 2))), "mae": float(np.mean(np.abs(difference))),
    }
