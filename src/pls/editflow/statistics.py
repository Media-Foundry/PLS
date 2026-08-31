"""Paired statistics for frozen-anchor EditFlow comparisons."""

from __future__ import annotations

import numpy as np


def exact_sign_flip_pvalue(differences) -> float:
    """Two-sided exact randomization p-value for a paired mean difference."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) < 1:
        raise ValueError("differences must be a nonempty one-dimensional array")
    if len(differences) > 20:
        raise ValueError("exact sign enumeration is limited to 20 pairs")
    if np.any(~np.isfinite(differences)):
        raise ValueError("differences must be finite")
    assignments = np.arange(1 << len(differences), dtype=np.uint64)[:, None]
    bits = (assignments >> np.arange(len(differences), dtype=np.uint64)) & 1
    signs = bits.astype(np.float64) * 2.0 - 1.0
    permuted = (signs * differences).mean(axis=1)
    observed = abs(float(differences.mean()))
    return float(np.mean(np.abs(permuted) >= observed - 1e-15))


def paired_bootstrap_interval(
    differences,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval resampling paired anchor differences."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) < 1:
        raise ValueError("differences must be a nonempty one-dimensional array")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def paired_method_summary(
    path_values,
    uncertainty_values,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    """Summarize path minus uncertainty; negative differences favor path."""
    path = np.asarray(path_values, dtype=np.float64)
    uncertainty = np.asarray(uncertainty_values, dtype=np.float64)
    if path.shape != uncertainty.shape or path.ndim != 1:
        raise ValueError("paired method values must be equal one-dimensional arrays")
    differences = path - uncertainty
    lower, upper = paired_bootstrap_interval(
        differences, samples=bootstrap_samples, seed=bootstrap_seed
    )
    tolerance = 1e-12
    return {
        "pairs": int(len(path)),
        "path_mean": float(path.mean()),
        "uncertainty_mean": float(uncertainty.mean()),
        "path_minus_uncertainty_mean": float(differences.mean()),
        "path_minus_uncertainty_median": float(np.median(differences)),
        "bootstrap_95_ci": [lower, upper],
        "exact_two_sided_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
        "path_wins": int(np.sum(differences < -tolerance)),
        "ties": int(np.sum(np.abs(differences) <= tolerance)),
        "uncertainty_wins": int(np.sum(differences > tolerance)),
        "direction": "negative_favors_path",
    }
