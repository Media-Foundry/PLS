"""Label-free reliability signals and decision-regret curves over complete neighborhoods.

Everything here is computed from fields that cost zero mutant folds: the
cached-parent field, the teacher's sequence-only ablation, and the one-backward
gradient. The exact field is used only to score the outcome, never as an input to
a signal.

Answers three things:
  1. regret@K, not just exact-best recall, because a near-miss may not matter;
  2. whether any label-free statistic separates the reliable anchors from the
     catastrophic one;
  3. whether cheap union budgets rescue an anchor the cached ranking misses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
EPSILONS = (0.0, 0.02, 0.05, 0.10, 0.20)


def score_map(path: Path, key: str = "logits") -> dict[str, float]:
    values = np.load(path)
    return {str(k): float(v) for k, v in zip(values["sequence_sha256"], values[key])}


def effective_support(delta: np.ndarray, temperature: float) -> float:
    weights = np.exp((delta - delta.max()) / temperature)
    weights = weights / weights.sum()
    return float(1.0 / np.square(weights).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cached-scores", type=Path, required=True)
    parser.add_argument("--exact-scores", type=Path, required=True)
    parser.add_argument("--gradient-field", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text())
    if manifest.get("test_evaluated") is not False:
        raise SystemExit("manifest is not explicitly test-free")
    nodes = manifest["nodes"]
    if any(node["split"] != "train" for node in nodes):
        raise SystemExit("neighborhood manifest must be train-only")

    cached = score_map(arguments.cached_scores)
    sequence_only = score_map(arguments.cached_scores, "sequence_only_logits")
    exact = score_map(arguments.exact_scores)
    gradient = np.load(arguments.gradient_field)
    alphabet = "".join(str(c) for c in gradient["alphabet"])
    index_of = {c: i for i, c in enumerate(alphabet)}

    grouped: dict[int, dict] = {}
    for edge in manifest["edges"]:
        source = nodes[int(edge["source_node"])]
        target = nodes[int(edge["target_node"])]
        rank = int(edge["anchor_rank"])
        row = grouped.setdefault(rank, {"length": int(source["length"]),
                                        "low": [], "seq": [], "high": [], "grad": []})
        digest = target["sequence_sha256"]
        mutation = target["mutation"]
        row["low"].append(cached[digest] - cached[source["sequence_sha256"]])
        row["seq"].append(sequence_only[digest] - sequence_only[source["sequence_sha256"]])
        row["high"].append(exact[digest] - exact[source["sequence_sha256"]])
        row["grad"].append(float(gradient[f"field_{rank}"][
            int(mutation["position_zero_based"]), index_of[mutation["target_residue"]]]))

    anchors = []
    for rank in sorted(grouped):
        row = grouped[rank]
        low = np.asarray(row["low"])
        seq = np.asarray(row["seq"])
        high = np.asarray(row["high"])
        grad = np.asarray(row["grad"])
        order_low = np.argsort(-low, kind="stable")
        order_seq = np.argsort(-seq, kind="stable")
        best = int(np.argmax(high))
        sorted_low = low[order_low]

        # Label-free field-shape and disagreement statistics.
        scale = float(np.median(np.abs(low - np.median(low))))
        features = {
            "top_margin": float(sorted_low[0] - sorted_low[1]),
            "top8_dispersion": float(sorted_low[0] - sorted_low[7]),
            "top32_dispersion": float(sorted_low[0] - sorted_low[31]),
            "median_absolute_deviation": scale,
            "top_margin_over_mad": float((sorted_low[0] - sorted_low[1]) / max(scale, 1e-9)),
            "beneficial_fraction": float(np.mean(low > 0)),
            "effective_support_tau_mad": effective_support(low, max(scale, 1e-9)),
            "struct_seq_rank_disagreement": float(1.0 - spearmanr(low, seq).statistic),
            "struct_seq_top8_disjoint": float(1.0 - len(
                set(order_low[:8].tolist()) & set(order_seq[:8].tolist())) / 8.0),
            "struct_seq_top32_disjoint": float(1.0 - len(
                set(order_low[:32].tolist()) & set(order_seq[:32].tolist())) / 32.0),
            "top_gap_struct_minus_seq": float(low.max() - seq.max()),
        }

        regret = {}
        for budget in BUDGETS:
            if budget > low.size:
                continue
            regret[budget] = {
                "cached_parent": float(high.max() - high[order_low[:budget]].max()),
                "sequence_only": float(high.max() - high[order_seq[:budget]].max()),
            }
        # Cheap union budgets: does the sequence-only view add anything?
        unions = {}
        for total, cached_part in ((8, 6), (8, 4), (16, 12), (32, 24)):
            picked = list(dict.fromkeys(
                order_low[:cached_part].tolist() + order_seq[:total - cached_part].tolist()))
            picked = picked[:total] if len(picked) >= total else picked
            unions[f"top{cached_part}_cached_plus_top{total - cached_part}_seq"] = {
                "budget_used": len(picked),
                "regret": float(high.max() - high[picked].max()),
                "hit_exact_best": bool(best in set(picked)),
            }

        anchors.append({
            "anchor_rank": rank,
            "length": row["length"],
            "candidates": int(low.size),
            "exact_best_gain": float(high.max()),
            "cached_rank_of_exact_best": int(np.where(order_low == best)[0][0]) + 1,
            "sequence_only_rank_of_exact_best": int(np.where(order_seq == best)[0][0]) + 1,
            "features": features,
            "regret_by_budget": regret,
            "union_budgets": unions,
        })

    summary = []
    for budget in BUDGETS:
        rows = [a for a in anchors if budget in a["regret_by_budget"]]
        if not rows:
            continue
        values = np.asarray([a["regret_by_budget"][budget]["cached_parent"] for a in rows])
        entry = {
            "budget": budget,
            "anchors": len(rows),
            "mean_regret": float(values.mean()),
            "median_regret": float(np.median(values)),
            "maximum_regret": float(values.max()),
        }
        for epsilon in EPSILONS:
            entry[f"p_regret_le_{epsilon:g}"] = float(np.mean(values <= epsilon + 1e-12))
        summary.append(entry)

    # Which label-free feature tracks the outcome? With eight anchors this is a
    # direction check, not an estimate.
    outcome = np.asarray([np.log10(a["cached_rank_of_exact_best"]) for a in anchors])
    correlations = {}
    for name in anchors[0]["features"]:
        values = np.asarray([a["features"][name] for a in anchors])
        correlations[name] = {
            "spearman_with_log_rank": float(spearmanr(values, outcome).statistic),
            "values": [round(float(v), 5) for v in values],
        }

    output = {
        "schema": "PLS_neighborhood_reliability_v1",
        "anchors": len(anchors),
        "budgets": list(BUDGETS),
        "epsilons": list(EPSILONS),
        "regret_summary": summary,
        "per_anchor": anchors,
        "label_free_feature_correlations": correlations,
        "caveat": "eight anchors; every number here is a direction check, not an estimate",
        "mutant_folds_required_for_signals": 0,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Decision regret and label-free reliability over complete neighborhoods",
        "",
        f"{len(anchors)} pilot anchors. Every signal below costs zero mutant folds; the",
        "exact field scores the outcome and never feeds a signal.",
        "",
        "## Regret, not exact-best identity",
        "",
        "| Budget | Mean regret | Median | Max | P(R<=0) | P(R<=0.02) | P(R<=0.05) | P(R<=0.10) | P(R<=0.20) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['budget']} | {row['mean_regret']:.4f} | {row['median_regret']:.4f} | "
            f"{row['maximum_regret']:.4f} | {row['p_regret_le_0']:.3f} | "
            f"{row['p_regret_le_0.02']:.3f} | {row['p_regret_le_0.05']:.3f} | "
            f"{row['p_regret_le_0.1']:.3f} | {row['p_regret_le_0.2']:.3f} |")
    lines += [
        "",
        "## Per-anchor: does the miss matter?",
        "",
        "| Anchor | L | Cached rank of exact best | Exact best gain | Regret at K=8 | Regret at K=32 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a in anchors:
        lines.append(
            f"| {a['anchor_rank']} | {a['length']} | {a['cached_rank_of_exact_best']} | "
            f"{a['exact_best_gain']:.4f} | {a['regret_by_budget'][8]['cached_parent']:.4f} | "
            f"{a['regret_by_budget'][32]['cached_parent']:.4f} |")
    lines += [
        "",
        "## Label-free signals against the outcome",
        "",
        "Spearman between each zero-fold statistic and log10 of the cached rank of the",
        "exact best. Eight points: read the sign and the magnitude, nothing finer.",
        "",
        "| Signal | Spearman with log rank |",
        "| --- | ---: |",
    ]
    for name, value in sorted(correlations.items(),
                              key=lambda kv: -abs(kv[1]["spearman_with_log_rank"])):
        lines.append(f"| `{name}` | {value['spearman_with_log_rank']:+.4f} |")
    lines += [
        "",
        "## Do cheap union budgets rescue the miss?",
        "",
        "| Union | Anchors hitting exact best | Mean regret |",
        "| --- | ---: | ---: |",
    ]
    for name in anchors[0]["union_budgets"]:
        hits = float(np.mean([a["union_budgets"][name]["hit_exact_best"] for a in anchors]))
        mean_regret = float(np.mean([a["union_budgets"][name]["regret"] for a in anchors]))
        lines.append(f"| `{name}` | {hits:.4f} | {mean_regret:.4f} |")
    lines.append("")
    arguments.md_out.write_text("\n".join(lines))

    print(json.dumps({"regret_summary": summary}, indent=2))
    print()
    for name, value in sorted(correlations.items(),
                              key=lambda kv: -abs(kv[1]["spearman_with_log_rank"])):
        print(f"  {name:>34}  rho_with_log_rank={value['spearman_with_log_rank']:+.4f}")


if __name__ == "__main__":
    main()
