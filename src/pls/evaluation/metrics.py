"""Endpoint-specific solubility metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, roc_auc_score)


def binary_metrics(targets, logits, bins: int = 10):
    targets = np.asarray(targets, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    predictions = probabilities >= .5
    ece = 0.0
    boundaries = np.linspace(0, 1, bins + 1)
    for index in range(bins):
        selected = ((probabilities >= boundaries[index]) &
                    (probabilities < boundaries[index + 1] if index < bins - 1 else probabilities <= 1))
        if np.any(selected):
            ece += selected.mean() * abs(probabilities[selected].mean() - targets[selected].mean())
    return {
        "n": int(len(targets)), "auroc": float(roc_auc_score(targets, probabilities)),
        "auprc": float(average_precision_score(targets, probabilities)),
        "mcc": float(matthews_corrcoef(targets, predictions)),
        "f1": float(f1_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "brier": float(np.mean((probabilities - targets) ** 2)), "ece": float(ece),
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
