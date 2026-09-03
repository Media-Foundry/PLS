"""Exhaustive single-mutant neighborhood: what actually predicts the exact best?

Three links, measured over complete 19L neighborhoods rather than a sampled
16-candidate slate:

  A  one-backward gradient  ->  exhaustive cached-parent field   (zero folds)
  B  exhaustive cached-parent field  ->  exact field             (the known weak link)
  C  the composition                 ->  exact field

Link B is the load-bearing one. The confirmatory v2 campaign measured it on 16
sampled candidates per anchor; nothing guaranteed it survives a pool of 19L.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, spearmanr

TOL = 1e-12
BUDGETS = (1, 2, 4, 8, 16, 32, 64)


def score_map(path: Path, key: str = "logits") -> dict[str, float]:
    values = np.load(path)
    if key not in values.files:
        raise SystemExit(f"{path} has no array {key!r}")
    return {str(k): float(v) for k, v in zip(values["sequence_sha256"], values[key])}


def recall_and_regret(order: np.ndarray, exact: np.ndarray, budget: int) -> tuple[bool, float]:
    chosen = order[:budget]
    best = float(exact.max())
    achieved = float(exact[chosen].max())
    return bool(achieved >= best - TOL), best - achieved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cached-scores", type=Path, required=True)
    parser.add_argument("--exact-scores", type=Path, required=True)
    parser.add_argument("--gradient-field", type=Path, required=True)
    parser.add_argument("--fold-reports", type=Path, required=True)
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
    exact = score_map(arguments.exact_scores)
    # The teacher's own sequence head with the structure branch bypassed. It
    # isolates what reusing the parent structure actually buys.
    sequence_only = score_map(arguments.cached_scores, "sequence_only_logits")
    gradient = np.load(arguments.gradient_field, allow_pickle=False)
    alphabet = "".join(str(c) for c in gradient["alphabet"])
    index_of = {c: i for i, c in enumerate(alphabet)}

    seconds: dict[str, float] = {}
    for path in sorted(arguments.fold_reports.glob("shard_*_report.json")):
        for row in json.loads(path.read_text())["results"]:
            if row["status"] == "ok":
                seconds.setdefault(row["sequence_sha256"], float(row["seconds"]))

    by_anchor: dict[int, dict] = {}
    for edge in manifest["edges"]:
        source = nodes[int(edge["source_node"])]
        target = nodes[int(edge["target_node"])]
        rank = int(edge["anchor_rank"])
        row = by_anchor.setdefault(rank, {
            "length": int(source["length"]), "parent": source["sequence_sha256"],
            "low": [], "high": [], "grad": [], "seq": [], "seconds": [],
        })
        digest = target["sequence_sha256"]
        mutation = target["mutation"]
        field = gradient[f"field_{rank}"]
        row["low"].append(cached[digest] - cached[source["sequence_sha256"]])
        row["seq"].append(sequence_only[digest] - sequence_only[source["sequence_sha256"]])
        row["high"].append(exact[digest] - exact[source["sequence_sha256"]])
        row["grad"].append(float(field[int(mutation["position_zero_based"]),
                                       index_of[mutation["target_residue"]]]))
        row["seconds"].append(seconds[digest])

    anchors = []
    for rank in sorted(by_anchor):
        row = by_anchor[rank]
        anchors.append({
            "anchor_rank": rank, "length": row["length"],
            "candidates": len(row["low"]),
            "low": np.asarray(row["low"], dtype=np.float64),
            "seq": np.asarray(row["seq"], dtype=np.float64),
            "high": np.asarray(row["high"], dtype=np.float64),
            "grad": np.asarray(row["grad"], dtype=np.float64),
            "seconds": np.asarray(row["seconds"], dtype=np.float64),
        })

    links = {}
    for name, predictor, target in (
        ("A_gradient_vs_cached", "grad", "low"),
        ("B_cached_vs_exact", "low", "high"),
        ("C_gradient_vs_exact", "grad", "high"),
        ("D_sequence_only_vs_exact", "seq", "high"),
        ("E_cached_vs_sequence_only", "low", "seq"),
    ):
        per_anchor = []
        for row in anchors:
            x, y = row[predictor], row[target]
            rho = float(spearmanr(x, y).statistic)
            pearson = float(np.corrcoef(x, y)[0, 1])
            sign = float(np.mean(np.sign(x) == np.sign(y)))
            per_anchor.append({
                "anchor_rank": row["anchor_rank"], "length": row["length"],
                "candidates": row["candidates"], "spearman": rho,
                "pearson": pearson, "sign_agreement": sign,
            })
        links[name] = {
            "predictor": predictor, "target": target,
            "per_anchor": per_anchor,
            "mean_spearman": float(np.mean([r["spearman"] for r in per_anchor])),
            "median_spearman": float(np.median([r["spearman"] for r in per_anchor])),
            "minimum_spearman": float(np.min([r["spearman"] for r in per_anchor])),
            "mean_sign_agreement": float(np.mean([r["sign_agreement"] for r in per_anchor])),
        }

    rankings = {}
    for name, key in (("cached_parent", "low"), ("sequence_only", "seq"),
                      ("one_backward_gradient", "grad")):
        rows = []
        for budget in BUDGETS:
            hits, regrets, costs = [], [], []
            for row in anchors:
                order = np.argsort(-row[key], kind="stable")
                hit, regret = recall_and_regret(order, row["high"], budget)
                hits.append(hit)
                regrets.append(regret)
                costs.append(float(row["seconds"][order[:budget]].sum()))
            exhaustive = float(sum(r["seconds"].sum() for r in anchors))
            rows.append({
                "budget": budget,
                "exact_best_recall": float(np.mean(hits)),
                "mean_regret": float(np.mean(regrets)),
                "maximum_regret": float(np.max(regrets)),
                "measured_gpu_seconds": float(sum(costs)),
                "measured_cost_fraction": float(sum(costs) / exhaustive),
            })
        rankings[name] = rows

    # A random-order control makes the recall numbers legible.
    generator = np.random.default_rng(0)
    random_rows = []
    for budget in BUDGETS:
        hits = []
        for row in anchors:
            for _ in range(20):
                order = generator.permutation(row["candidates"])
                hits.append(recall_and_regret(order, row["high"], budget)[0])
        random_rows.append({"budget": budget, "exact_best_recall": float(np.mean(hits))})
    rankings["random_control"] = random_rows

    # Where the exact best actually sits in each cheap ranking. With eight
    # anchors the recall numbers are wide, so report the ranks themselves.
    placement = []
    for row in anchors:
        target = int(np.argmax(row["high"]))
        entry = {"anchor_rank": row["anchor_rank"], "length": row["length"],
                 "candidates": row["candidates"],
                 "exact_best_gain": float(row["high"][target])}
        for name, key in (("cached_parent", "low"), ("sequence_only", "seq"),
                          ("one_backward_gradient", "grad")):
            order = np.argsort(-row[key], kind="stable")
            entry[f"{name}_rank"] = int(np.where(order == target)[0][0]) + 1
        placement.append(entry)
    cached_ranks = np.asarray([p["cached_parent_rank"] for p in placement])
    for name in ("cached_parent", "sequence_only", "one_backward_gradient"):
        ranks = np.asarray([p[f"{name}_rank"] for p in placement])
        for row in rankings[name]:
            successes = int((ranks <= row["budget"]).sum())
            total = len(ranks)
            row["exact_best_recall_clopper_pearson_95"] = [
                0.0 if successes == 0 else float(beta.ppf(0.025, successes, total - successes + 1)),
                1.0 if successes == total else float(beta.ppf(0.975, successes + 1, total - successes)),
            ]

    exhaustive_seconds = float(sum(r["seconds"].sum() for r in anchors))
    output = {
        "schema": "PLS_neighborhood_pilot_analysis_v1",
        "anchors": len(anchors),
        "total_candidates": int(sum(r["candidates"] for r in anchors)),
        "measured_exhaustive_gpu_seconds": exhaustive_seconds,
        "links": links,
        "rankings": rankings,
        "exact_best_placement": placement,
        "cached_parent_rank_of_exact_best": sorted(cached_ranks.tolist()),
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Exhaustive single-mutant neighborhood pilot",
        "",
        f"- anchors: {len(anchors)} fresh SI30 components, lengths "
        f"{min(r['length'] for r in anchors)}-{max(r['length'] for r in anchors)}",
        f"- candidates: {output['total_candidates']} complete single-mutant neighborhoods",
        f"- measured exhaustive ESMFold: {exhaustive_seconds:.1f} GPU-seconds",
        "- test sequences queried: 0",
        "",
        "## Three links",
        "",
        "| Link | Mean Spearman | Median | Min | Sign agreement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "A_gradient_vs_cached": "A. one-backward gradient to cached-parent field",
        "B_cached_vs_exact": "B. cached-parent field to exact field",
        "C_gradient_vs_exact": "C. gradient to exact field",
        "D_sequence_only_vs_exact": "D. sequence-only ablation to exact field",
        "E_cached_vs_sequence_only": "E. cached-parent to sequence-only",
    }
    for key, label in labels.items():
        v = links[key]
        lines.append(f"| {label} | {v['mean_spearman']:.4f} | {v['median_spearman']:.4f} | "
                     f"{v['minimum_spearman']:.4f} | {v['mean_sign_agreement']:.4f} |")
    lines += [
        "",
        "## Exact-best recall at a fixed fold budget",
        "",
        "Every row folds `budget` candidates out of the complete neighborhood and asks",
        "whether the exact best single mutant is among them.",
        "",
        "| Budget | Cached-parent | 95% CI | Sequence-only | Gradient | Random | "
        "Cached mean regret | Cost fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, budget in enumerate(BUDGETS):
        c = rankings["cached_parent"][i]
        q = rankings["sequence_only"][i]
        g = rankings["one_backward_gradient"][i]
        r = rankings["random_control"][i]
        ci = c["exact_best_recall_clopper_pearson_95"]
        lines.append(
            f"| {budget} | {c['exact_best_recall']:.4f} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{q['exact_best_recall']:.4f} | "
            f"{g['exact_best_recall']:.4f} | {r['exact_best_recall']:.4f} | "
            f"{c['mean_regret']:.4f} | {c['measured_cost_fraction']:.4f} |")
    lines += [
        "",
        "## Per-anchor link B",
        "",
        "| Anchor | L | Candidates | Spearman | Sign agreement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in links["B_cached_vs_exact"]["per_anchor"]:
        lines.append(f"| {row['anchor_rank']} | {row['length']} | {row['candidates']} | "
                     f"{row['spearman']:.4f} | {row['sign_agreement']:.4f} |")
    lines += [
        "",
        "## Where the exact best mutation sits in each cheap ranking",
        "",
        "| Anchor | L | Candidates | Cached-parent rank | Sequence-only rank | Gradient rank | Exact gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in placement:
        lines.append(
            f"| {row['anchor_rank']} | {row['length']} | {row['candidates']} | "
            f"{row['cached_parent_rank']} | {row['sequence_only_rank']} | "
            f"{row['one_backward_gradient_rank']} | {row['exact_best_gain']:.4f} |")
    lines += [
        "",
        "The cached-parent ranking is bimodal: it puts the exact best mutation first",
        "or near-first on most anchors and fails almost completely on one. Read the",
        "recall column with its interval; eight anchors is a pilot, not an estimate.",
        "",
    ]
    arguments.md_out.write_text("\n".join(lines))

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_anchor"}
                      for k, v in links.items()}, indent=2))
    print()
    for i, budget in enumerate(BUDGETS):
        c = rankings["cached_parent"][i]
        q = rankings["sequence_only"][i]
        g = rankings["one_backward_gradient"][i]
        r = rankings["random_control"][i]
        print(f"  budget={budget:>3}  cached={c['exact_best_recall']:.4f}  "
              f"seq_only={q['exact_best_recall']:.4f}  grad={g['exact_best_recall']:.4f}  "
              f"random={r['exact_best_recall']:.4f}  "
              f"cached_regret={c['mean_regret']:.4f}  cost={c['measured_cost_fraction']:.4f}")


if __name__ == "__main__":
    main()
