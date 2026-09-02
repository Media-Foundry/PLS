"""Decision-focused verification and certified gating for cached-context oracles."""

from __future__ import annotations

import numpy as np


def finite_sample_quantile(values, *, alpha: float) -> float:
    """Return the split-conformal finite-sample order statistic.

    For ``n`` calibration scores this is the one-based order statistic
    ``ceil((n + 1) * (1 - alpha))``, clipped to ``n``.  Indexing the sorted
    scores directly avoids the subtly different ``(n - 1) * q`` convention
    used by :func:`numpy.quantile`.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = min(len(values), int(np.ceil((len(values) + 1) * (1.0 - alpha))))
    return float(np.sort(values)[rank - 1])


def legacy_conservative_quantile(values, *, alpha: float) -> float:
    """Reproduce the conservative quantile used by frozen v1 protocols.

    Do not use this helper for new protocols.  The confirmatory v1 margin is
    stored as a literal threshold, so correcting :func:`finite_sample_quantile`
    cannot alter that frozen result.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(values)):
        raise ValueError("values must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    level = min(1.0, np.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def margin_candidate_indices(low, threshold: float, scale=None) -> np.ndarray:
    """Return a fixed or candidate-scaled low-fidelity decision set.

    With no scale this is ``max(low) - low[j] <= threshold``.  Positive
    candidate-specific scales support future cost-aware conformal scores via
    ``(max(low) - low[j]) / scale[j] <= threshold``.
    """
    low = np.asarray(low, dtype=np.float64)
    if low.ndim != 1 or not len(low) or np.any(~np.isfinite(low)):
        raise ValueError("low must be a finite nonempty one-dimensional array")
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be finite and nonnegative")
    gaps = float(np.max(low)) - low
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape != low.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scale must be finite, positive, and match low")
        gaps = gaps / scale
    return np.flatnonzero(gaps <= threshold + 1e-12)


def epsilon_optimal_nonconformity(low, exact, epsilon: float, scale=None) -> float:
    """Score the cheapest low-fidelity inclusion of an exact epsilon-optimum.

    The score is the minimum (possibly scaled) low-fidelity gap among exact
    candidates within ``epsilon`` of the exact optimum.  It reduces to the
    exact-argmax margin score at ``epsilon=0`` when the optimum is unique.
    """
    low = np.asarray(low, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if low.ndim != 1 or exact.shape != low.shape or not len(low):
        raise ValueError("low and exact must be equal nonempty one-dimensional arrays")
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(exact)):
        raise ValueError("low and exact must be finite")
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and nonnegative")
    gaps = float(np.max(low)) - low
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape != low.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scale must be finite, positive, and match low")
        gaps = gaps / scale
    acceptable = exact >= float(np.max(exact)) - epsilon - 1e-12
    return float(np.min(gaps[acceptable]))


def empirical_upper_cvar(values, *, level: float = 0.95) -> float:
    """Average the worst ``ceil((1-level)*n)`` empirical outcomes."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("values must be a finite nonempty one-dimensional array")
    if not 0 <= level < 1:
        raise ValueError("level must lie in [0, 1)")
    count = max(1, int(np.ceil((1.0 - level) * len(values))))
    return float(np.mean(np.sort(values)[-count:]))


def regret_summary(regrets, *, tolerance: float = 1e-12) -> dict:
    """Report decision regret with metrics that remain informative at high coverage."""
    regrets = np.asarray(regrets, dtype=np.float64)
    if regrets.ndim != 1 or not len(regrets) or np.any(~np.isfinite(regrets)):
        raise ValueError("regrets must be a finite nonempty one-dimensional array")
    if np.any(regrets < -tolerance):
        raise ValueError("regrets cannot be negative")
    regrets = np.maximum(regrets, 0.0)
    failures = regrets[regrets > tolerance]
    return {
        "zero_regret_fraction": float(np.mean(regrets <= tolerance)),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "maximum_regret": float(np.max(regrets)),
        "regret_p95": float(np.quantile(regrets, 0.95)),
        "regret_p99": float(np.quantile(regrets, 0.99)),
        "regret_cvar95": empirical_upper_cvar(regrets, level=0.95),
        "failure_count": int(len(failures)),
        "failure_conditional_mean_regret": (
            float(np.mean(failures)) if len(failures) else 0.0
        ),
    }


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
        "regret_p90": float(np.quantile(regret_values, 0.9)),
        **regret_summary(regret_values, tolerance=tolerance),
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
