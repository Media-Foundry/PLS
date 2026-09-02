"""Frozen runtime-cost utilities for cached-oracle conformal gating."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_runtime_cost_model(path: str | Path) -> dict:
    """Load and validate a versioned monotone runtime model."""
    model = json.loads(Path(path).read_text())
    if model.get("schema") != "PLS_ESMFold_monotone_runtime_cost_model_v1":
        raise ValueError("unsupported runtime-cost model schema")
    lengths = np.asarray(model["length_knots"], dtype=np.float64)
    seconds = np.asarray(model["seconds_knots"], dtype=np.float64)
    if (
        lengths.ndim != 1
        or seconds.shape != lengths.shape
        or not len(lengths)
        or np.any(~np.isfinite(lengths))
        or np.any(~np.isfinite(seconds))
        or np.any(np.diff(lengths) <= 0)
        or np.any(np.diff(seconds) < -1e-12)
        or np.any(seconds <= 0)
    ):
        raise ValueError("invalid monotone runtime-cost knots")
    if not np.isfinite(float(model["reference_cost_seconds"])) or float(
        model["reference_cost_seconds"]
    ) <= 0:
        raise ValueError("invalid frozen reference cost")
    if model.get("test_evaluated") is not False:
        raise ValueError("runtime-cost model must explicitly remain test-free")
    return model


def predict_runtime_seconds(model: dict, lengths) -> np.ndarray:
    """Predict typical marginal ESMFold inference seconds by sequence length."""
    values = np.asarray(lengths, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("lengths must be finite and positive")
    knots = np.asarray(model["length_knots"], dtype=np.float64)
    seconds = np.asarray(model["seconds_knots"], dtype=np.float64)
    return np.interp(values, knots, seconds, left=seconds[0], right=seconds[-1])


def runtime_cost_scale(
    model: dict,
    lengths,
    *,
    gamma: float,
) -> np.ndarray:
    """Return the frozen cost-aware conformal scale ``(c/c0)^(-gamma)``."""
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    cost = predict_runtime_seconds(model, lengths)
    reference = float(model["reference_cost_seconds"])
    return (cost / reference) ** (-float(gamma))
