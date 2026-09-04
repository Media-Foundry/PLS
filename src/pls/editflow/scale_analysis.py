"""Frozen analysis of a complete-neighborhood campaign.

Driven entirely by a protocol frozen before the exact field exists, so the budget
grid, the primary metric, the tail thresholds and the baseline set cannot be
chosen after seeing the outcome. The module refuses to run against a protocol
that is not committed and clean.

Policies are compared at MATCHED total exact budget on the loss distribution
first: mean, CVaR95 and maximum decision regret. Hit rate is secondary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_frozen(path: Path) -> None:
    """A protocol that is untracked or dirty is not a protocol."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        raise SystemExit(f"refusing: {path} is not committed, so it is not frozen")
    dirty = subprocess.run(["git", "status", "--porcelain", str(path)],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"refusing: {path} has uncommitted edits:\n{dirty}")


def score_map(path: Path, key: str = "logits") -> dict[str, float]:
    values = np.load(path)
    if key not in values.files:
        raise SystemExit(f"{path} has no array {key!r}")
    return {str(k): float(v) for k, v in zip(values["sequence_sha256"], values[key])}


def upper_cvar(values: np.ndarray, level: float) -> float:
    """Mean of the worst 1-level tail, always including at least one point."""
    if values.size == 0:
        return 0.0
    ordered = np.sort(values)[::-1]
    keep = max(1, int(np.ceil(values.size * (1.0 - level))))
    return float(ordered[:keep].mean())


def effective_support(delta: np.ndarray, temperature: float) -> float:
    weights = np.exp((delta - delta.max()) / temperature)
    weights = weights / weights.sum()
    return float(1.0 / np.square(weights).sum())


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney form; 0.5 when either class is empty."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([positive, negative]))) + 1.0
    # Average ranks over ties so the statistic stays unbiased.
    order = np.argsort(np.concatenate([positive, negative]))
    values = np.concatenate([positive, negative])[order]
    adjusted = ranks.astype(float)
    start = 0
    sorted_ranks = np.sort(ranks)
    for index in range(1, values.size + 1):
        if index == values.size or values[index] != values[start]:
            mean_rank = sorted_ranks[start:index].mean()
            adjusted[order[start:index]] = mean_rank
            start = index
    return float((adjusted[:positive.size].sum()
                  - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, hits.size + 1)
    return float((precision * hits).sum() / labels.sum())


def parent_features(manifest: dict, artifact_root: Path) -> dict[int, dict]:
    """Label-free parent statistics available before any mutant is folded."""
    compact = artifact_root / "fixed_parent" / "structure_v4_compact"
    patches = artifact_root / "fixed_parent" / "surface_patch_components"
    offsets = np.load(compact / "offsets.npy", mmap_mode="r")
    shape = tuple(json.loads((compact / "metadata.json").read_text())["shape"])
    features = np.memmap(compact / "residue_features.f16", mode="r",
                         dtype=np.float16, shape=shape)
    patch_shape = tuple(json.loads((patches / "metadata.json").read_text())["shape"])
    components = np.memmap(patches / "component_ids.i32", mode="r",
                           dtype=np.int32, shape=patch_shape)
    out = {}
    for node in manifest["nodes"]:
        if node["kind"] != "anchor":
            continue
        index = int(node["node_index"])
        low, high = int(offsets[index]), int(offsets[index + 1])
        plddt = np.asarray(features[low:high, -1], dtype=np.float64)
        ids = np.asarray(components[low:high], dtype=np.int64)
        out[int(node["anchor_rank"])] = {
            "parent_mean_plddt": float(plddt.mean()),
            "parent_min_plddt": float(plddt.min()),
            "surface_patch_count": float(len({
                (column, value)
                for column in range(ids.shape[1])
                for value in np.unique(ids[:, column]) if value >= 0
            })),
        }
    return out


def collect(manifest: dict, cached: Path, exact: Path, gradient: Path,
            artifact_root: Path) -> list[dict]:
    nodes = manifest["nodes"]
    low_map = score_map(cached)
    seq_map = score_map(cached, "sequence_only_logits")
    high_map = score_map(exact)
    field = np.load(gradient)
    alphabet = "".join(str(c) for c in field["alphabet"])
    index_of = {c: i for i, c in enumerate(alphabet)}
    parents = parent_features(manifest, artifact_root)

    grouped: dict[int, dict] = {}
    for edge in manifest["edges"]:
        source = nodes[int(edge["source_node"])]
        target = nodes[int(edge["target_node"])]
        rank = int(edge["anchor_rank"])
        row = grouped.setdefault(rank, {"length": int(source["length"]),
                                        "low": [], "seq": [], "high": [], "grad": []})
        digest = target["sequence_sha256"]
        mutation = target["mutation"]
        parent = source["sequence_sha256"]
        row["low"].append(low_map[digest] - low_map[parent])
        row["seq"].append(seq_map[digest] - seq_map[parent])
        row["high"].append(high_map[digest] - high_map[parent])
        row["grad"].append(float(field[f"field_{rank}"][
            int(mutation["position_zero_based"]), index_of[mutation["target_residue"]]]))

    anchors = []
    for rank in sorted(grouped):
        row = grouped[rank]
        low = np.asarray(row["low"])
        seq = np.asarray(row["seq"])
        high = np.asarray(row["high"])
        grad = np.asarray(row["grad"])
        order_low = np.argsort(-low, kind="stable")
        sorted_low = low[order_low]
        order_seq = np.argsort(-seq, kind="stable")
        scale = max(float(np.median(np.abs(low - np.median(low)))), 1e-9)
        best = int(np.argmax(high))
        signals = {
            "top_margin": float(sorted_low[0] - sorted_low[1]),
            "top8_dispersion": float(sorted_low[0] - sorted_low[min(7, low.size - 1)]),
            "top32_dispersion": float(sorted_low[0] - sorted_low[min(31, low.size - 1)]),
            "median_absolute_deviation": scale,
            "top_margin_over_mad": float((sorted_low[0] - sorted_low[1]) / scale),
            "beneficial_fraction": float(np.mean(low > 0)),
            "effective_support_tau_mad": effective_support(low, scale),
            "struct_seq_rank_disagreement": float(1.0 - spearmanr(low, seq).statistic),
            "struct_seq_top8_disjoint": float(1.0 - len(
                set(order_low[:8].tolist()) & set(order_seq[:8].tolist())) / 8.0),
            "struct_seq_top32_disjoint": float(1.0 - len(
                set(order_low[:32].tolist()) & set(order_seq[:32].tolist())) / 32.0),
            "top_gap_struct_minus_seq": float(low.max() - seq.max()),
            "length": float(row["length"]),
            **parents.get(rank, {}),
        }
        anchors.append({
            "anchor_rank": rank, "length": row["length"], "candidates": int(low.size),
            "low": low, "seq": seq, "high": high, "grad": grad,
            "order_low": order_low, "order_seq": order_seq,
            "order_grad": np.argsort(-grad, kind="stable"),
            "exact_best_index": best,
            "exact_best_gain": float(high.max()),
            "cached_rank_of_exact_best": int(np.where(order_low == best)[0][0]) + 1,
            "signals": signals,
        })
    return anchors


def selections(anchor: dict, policy: str, budget: int, m: int = 0) -> np.ndarray:
    if policy == "cached_parent":
        return anchor["order_low"][:budget]
    if policy == "sequence_only":
        return anchor["order_seq"][:budget]
    if policy == "one_backward_gradient":
        return anchor["order_grad"][:budget]
    if policy == "hybrid_union":
        picked = list(dict.fromkeys(
            anchor["order_low"][:budget - m].tolist()
            + anchor["order_seq"][:m].tolist()))
        return np.asarray(picked[:budget], dtype=int)
    raise ValueError(policy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cached-scores", type=Path, required=True)
    parser.add_argument("--exact-scores", type=Path, required=True)
    parser.add_argument("--gradient-field", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--allow-unfrozen-protocol", action="store_true",
                        help="only for validating this code on an already-unblinded run")
    arguments = parser.parse_args()

    if not arguments.allow_unfrozen_protocol:
        require_frozen(arguments.protocol)
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("test_evaluated") is not False:
        raise SystemExit("protocol is not explicitly test-free")
    manifest = json.loads(arguments.manifest.read_text())
    if manifest.get("test_evaluated") is not False:
        raise SystemExit("manifest is not explicitly test-free")
    if any(node["split"] != "train" for node in manifest["nodes"]):
        raise SystemExit("neighborhood manifest must be train-only")

    budgets = list(protocol["budget_grid"])
    thresholds = list(protocol["primary_metric"]["tail_thresholds"])
    m_values = list(protocol["frozen_baselines"]["hybrid_union"]["m_values"])
    anchors = collect(manifest, arguments.cached_scores, arguments.exact_scores,
                      arguments.gradient_field, arguments.artifact_root)

    generator = np.random.default_rng(0)
    table = []
    for budget in budgets:
        if any(budget > a["candidates"] for a in anchors):
            continue
        policies = [("cached_parent", 0), ("sequence_only", 0),
                    ("one_backward_gradient", 0)]
        policies += [("hybrid_union", m) for m in m_values if m < budget]
        for policy, m in policies:
            regrets, hits = [], []
            for anchor in anchors:
                picked = selections(anchor, policy, budget, m)
                achieved = float(anchor["high"][picked].max())
                regrets.append(anchor["exact_best_gain"] - achieved)
                hits.append(achieved >= anchor["exact_best_gain"] - TOL)
            regrets = np.asarray(regrets)
            row = {
                "budget": budget,
                "policy": policy if policy != "hybrid_union" else f"hybrid_union_m{m}",
                "mean_regret": float(regrets.mean()),
                "median_regret": float(np.median(regrets)),
                "regret_cvar95": upper_cvar(regrets, 0.95),
                "maximum_regret": float(regrets.max()),
                "exact_best_recall": float(np.mean(hits)),
            }
            for epsilon in thresholds:
                row[f"p_regret_le_{epsilon:g}"] = float(np.mean(regrets <= epsilon + TOL))
            table.append(row)
        # Random control, averaged over draws.
        regrets = []
        for anchor in anchors:
            for _ in range(20):
                picked = generator.permutation(anchor["candidates"])[:budget]
                regrets.append(anchor["exact_best_gain"] - float(anchor["high"][picked].max()))
        regrets = np.asarray(regrets)
        table.append({
            "budget": budget, "policy": "random_control",
            "mean_regret": float(regrets.mean()),
            "median_regret": float(np.median(regrets)),
            "regret_cvar95": upper_cvar(regrets, 0.95),
            "maximum_regret": float(regrets.max()),
            "exact_best_recall": float(np.mean(regrets <= TOL)),
            **{f"p_regret_le_{e:g}": float(np.mean(regrets <= e + TOL)) for e in thresholds},
        })

    # Exploratory detector evaluation: predict the catastrophic event, do not
    # correlate against log rank.
    names = sorted(anchors[0]["signals"])
    detectors = []
    for budget in budgets:
        if any(budget > a["candidates"] for a in anchors):
            continue
        rank_event = np.asarray(
            [a["cached_rank_of_exact_best"] > budget for a in anchors], dtype=int)
        events = {"rank_gt_K": rank_event}
        for epsilon in thresholds:
            if epsilon == 0.0:
                continue
            regrets = np.asarray([
                a["exact_best_gain"] - float(a["high"][selections(a, "cached_parent", budget)].max())
                for a in anchors])
            events[f"regret_gt_{epsilon:g}"] = (regrets > epsilon).astype(int)
        for event_name, labels in events.items():
            if labels.sum() in (0, labels.size):
                continue
            for name in names:
                values = np.asarray([a["signals"][name] for a in anchors], dtype=float)
                detectors.append({
                    "budget": budget, "event": event_name, "signal": name,
                    "positives": int(labels.sum()), "anchors": int(labels.size),
                    "roc_auc": roc_auc(values, labels),
                    "average_precision": average_precision(values, labels),
                    "roc_auc_negated": roc_auc(-values, labels),
                })

    output = {
        "schema": "PLS_neighborhood_scale_analysis_v1",
        "protocol_sha256": sha256(arguments.protocol),
        "protocol_status": protocol["status"],
        "anchors": len(anchors),
        "total_candidates": int(sum(a["candidates"] for a in anchors)),
        "policy_table": table,
        "per_anchor": [{
            "anchor_rank": a["anchor_rank"], "length": a["length"],
            "candidates": a["candidates"], "exact_best_gain": a["exact_best_gain"],
            "cached_rank_of_exact_best": a["cached_rank_of_exact_best"],
            "signals": a["signals"],
        } for a in anchors],
        "detector_evaluation": detectors,
        "detector_status": "exploratory; any promising detector requires confirmation on a fresh disjoint component batch",
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Complete-neighborhood campaign under the frozen analysis protocol",
        "",
        f"- anchors: {len(anchors)}",
        f"- candidates: {output['total_candidates']:,}",
        f"- protocol: `{arguments.protocol}` ({protocol['status']})",
        "- test sequences queried: 0",
        "",
        "## Policies at matched exact budget, judged on the loss distribution",
        "",
        "| Budget | Policy | Mean regret | CVaR95 | Max | P(R=0) | Exact-best recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table:
        lines.append(
            f"| {row['budget']} | `{row['policy']}` | {row['mean_regret']:.4f} | "
            f"{row['regret_cvar95']:.4f} | {row['maximum_regret']:.4f} | "
            f"{row['p_regret_le_0']:.3f} | {row['exact_best_recall']:.4f} |")
    lines += [
        "",
        "## Exploratory failure detection",
        "",
        "ROC AUC for predicting the catastrophic event from a label-free signal.",
        "`roc_auc_negated` is the same signal with the sign flipped, which settles",
        "which direction of a signal, if any, carries the information.",
        "",
        "| Budget | Event | Signal | Positives | ROC AUC | Negated | Avg precision |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(detectors, key=lambda r: -max(r["roc_auc"], r["roc_auc_negated"]))[:30]:
        lines.append(
            f"| {row['budget']} | `{row['event']}` | `{row['signal']}` | "
            f"{row['positives']}/{row['anchors']} | {row['roc_auc']:.3f} | "
            f"{row['roc_auc_negated']:.3f} | {row['average_precision']:.3f} |")
    lines += ["", "Detectors here are exploratory and require a fresh confirmatory batch.", ""]
    arguments.md_out.write_text("\n".join(lines))
    print(f"anchors={len(anchors)} rows={len(table)} detector_rows={len(detectors)}")
    for row in table:
        if row["budget"] in (8, 16):
            print(f"  B={row['budget']:>3} {row['policy']:>22}  "
                  f"mean={row['mean_regret']:.4f}  cvar95={row['regret_cvar95']:.4f}  "
                  f"max={row['maximum_regret']:.4f}  recall={row['exact_best_recall']:.4f}")


if __name__ == "__main__":
    main()
