"""Decision-focused verification and certified gating for cached-context oracles."""

from __future__ import annotations

import numpy as np


def finite_sample_quantile(values, *, alpha: float) -> float:
    """One-sided split-conformal higher quantile."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    level = min(1.0, np.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def top_m_exact_verification(low, exact, groups, m: int, *, tolerance: float = 1e-12) -> dict:
    """Verify the cheap top-M per decision group and report exact decision regret."""
    low = np.asarray(low, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    groups = np.asarray(groups)
    if low.ndim != 1 or exact.shape != low.shape or groups.shape != low.shape:
        raise ValueError("low, exact, and groups must be equal one-dimensional arrays")
    if m < 1:
        raise ValueError("m must be positive")
    regrets, inclusions, zero_regret, beneficial = [], [], [], []
    queries = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        count = min(m, len(indices))
        chosen = indices[np.argsort(-low[indices], kind="stable")[:count]]
        exact_best = float(np.max(exact[indices]))
        verified_best = float(np.max(exact[chosen]))
        regret = exact_best - verified_best
        exact_optima = set(indices[np.abs(exact[indices] - exact_best) <= tolerance].tolist())
        regrets.append(regret)
        inclusions.append(bool(exact_optima.intersection(chosen.tolist())))
        zero_regret.append(regret <= tolerance)
        beneficial.append(verified_best > 0)
        queries.append(count)
    regret_values = np.asarray(regrets)
    return {
        "groups": int(len(regrets)),
        "m": int(m),
        "mean_exact_queries": float(np.mean(queries)),
        "mean_exact_fraction": float(np.mean([
            min(m, np.sum(groups == group)) / np.sum(groups == group)
            for group in np.unique(groups)
        ])),
        "true_best_inclusion": float(np.mean(inclusions)),
        "zero_regret_fraction": float(np.mean(zero_regret)),
        "beneficial_verified_fraction": float(np.mean(beneficial)),
        "mean_regret": float(np.mean(regret_values)),
        "median_regret": float(np.median(regret_values)),
        "maximum_regret": float(np.max(regret_values)),
        "regret_p90": float(np.quantile(regret_values, 0.9)),
    }


def exact_best_rank(low, exact) -> int:
    """One-based cheap-oracle rank of the deterministic exact maximizer."""
    low = np.asarray(low, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if low.ndim != 1 or exact.shape != low.shape or not len(low):
        raise ValueError("low and exact must be equal nonempty one-dimensional arrays")
    order = np.argsort(-low, kind="stable")
    exact_best = int(np.argmax(exact))
    return int(np.flatnonzero(order == exact_best)[0]) + 1


def certified_best_from_intervals(lower, upper, *, tolerance: float = 0.0) -> int | None:
    """Return the uniquely certified maximizer, if one exists."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.ndim != 1 or upper.shape != lower.shape or not len(lower):
        raise ValueError("interval bounds must be equal nonempty one-dimensional arrays")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError("invalid interval bounds")
    candidate = int(np.argmax(lower))
    competitors = np.delete(upper, candidate)
    if not len(competitors) or lower[candidate] > float(np.max(competitors)) + tolerance:
        return candidate
    return None


def query_until_certified(low, exact, radius: float) -> dict:
    """Refold optimistic ambiguous candidates until one decision is certified.

    Unqueried intervals are ``low +/- radius``. Exact queries collapse one
    interval to a point. The next query is the unqueried candidate with largest
    upper bound, a deterministic optimistic-challenger policy.
    """
    low = np.asarray(low, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if low.ndim != 1 or exact.shape != low.shape or not len(low):
        raise ValueError("low and exact must be equal nonempty one-dimensional arrays")
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and nonnegative")
    lower, upper = low - radius, low + radius
    queried = np.zeros(len(low), dtype=bool)
    certified = certified_best_from_intervals(lower, upper)
    while certified is None:
        available = np.flatnonzero(~queried)
        if not len(available):
            certified = int(np.argmax(exact))
            break
        query = int(available[np.argmax(upper[available])])
        queried[query] = True
        lower[query] = exact[query]
        upper[query] = exact[query]
        certified = certified_best_from_intervals(lower, upper)
    exact_best = float(np.max(exact))
    regret = exact_best - float(exact[certified])
    covered = bool(np.all(np.abs(exact - low) <= radius + 1e-12))
    return {
        "selected_index": int(certified),
        "queries": int(queried.sum()),
        "query_fraction": float(queried.mean()),
        "simultaneous_coverage": covered,
        "correct": bool(regret <= 1e-12),
        "regret": float(regret),
    }


def empirical_bayes_slope_prior(low, exact, groups) -> tuple[float, float]:
    """Estimate a positive global slope and shrinkage precision from other groups."""
    low = np.asarray(low, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    groups = np.asarray(groups)
    if low.ndim != 1 or exact.shape != low.shape or groups.shape != low.shape:
        raise ValueError("invalid slope-prior arrays")
    denominator = float(np.sum(low * low))
    global_slope = max(0.0, float(np.sum(low * exact) / max(denominator, 1e-12)))
    slopes, residuals = [], []
    for group in np.unique(groups):
        selected = groups == group
        group_denominator = float(np.sum(low[selected] ** 2))
        slope = max(0.0, float(np.sum(low[selected] * exact[selected]) / max(group_denominator, 1e-12)))
        slopes.append(slope)
        residuals.extend((exact[selected] - slope * low[selected]).tolist())
    prior_variance = float(np.var(slopes, ddof=1)) if len(slopes) > 1 else 0.0
    noise_variance = float(np.mean(np.square(residuals))) if residuals else 0.0
    shrinkage = noise_variance / max(prior_variance, 1e-8)
    return global_slope, shrinkage


def shrinkage_slope(low_probe, exact_probe, global_slope: float, shrinkage: float) -> float:
    low_probe = np.asarray(low_probe, dtype=np.float64)
    exact_probe = np.asarray(exact_probe, dtype=np.float64)
    if low_probe.shape != exact_probe.shape or low_probe.ndim != 1:
        raise ValueError("probe arrays must have equal one-dimensional shape")
    numerator = float(np.sum(low_probe * exact_probe) + shrinkage * global_slope)
    denominator = float(np.sum(low_probe ** 2) + shrinkage)
    return max(0.0, numerator / max(denominator, 1e-12))
