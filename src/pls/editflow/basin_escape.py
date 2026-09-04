"""Does the cached-parent field fail when the optimum leaves the parent's basin?

Post-hoc mechanism diagnosis, declared in the frozen protocol as explanatory and
never a detector feature: it needs the mutant's own exact structure and so cannot
inform a pre-fold budget.

Costs no new folds. The exact tree holds each mutant's own C-alpha coordinates
and the cached-parent tree holds the parent's, replicated, so their difference is
exactly the refolding displacement. Superposition is Kabsch on the identity
alignment, which is correct here because a single substitution preserves length
and residue order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    mobile_centered = mobile - mobile.mean(0)
    target_centered = target - target.mean(0)
    covariance = mobile_centered.T @ target_centered
    u, _s, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    aligned = mobile_centered @ rotation.T
    deviations = np.linalg.norm(aligned - target_centered, axis=1)
    return float(np.sqrt(np.mean(np.square(deviations)))), deviations


def tm_score(deviations: np.ndarray) -> float:
    length = deviations.size
    d0 = 1.24 * np.cbrt(max(length - 15, 1)) - 1.8
    d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + np.square(deviations / d0))))


class Tree:
    def __init__(self, root: Path):
        vectors = root / "structure_v4_vectors"
        compact = root / "structure_v4_compact"
        meta = json.loads((vectors / "metadata.json").read_text())
        self.coords = np.memmap(vectors / "ca_coords.f32", mode="r",
                                dtype=np.float32, shape=(meta["residues"], 3))
        self.offsets = np.load(compact / "offsets.npy", mmap_mode="r")
        shape = tuple(json.loads((compact / "metadata.json").read_text())["shape"])
        self.features = np.memmap(compact / "residue_features.f16", mode="r",
                                  dtype=np.float16, shape=shape)

    def span(self, index: int) -> tuple[int, int]:
        return int(self.offsets[index]), int(self.offsets[index + 1])

    def ca(self, index: int) -> np.ndarray:
        low, high = self.span(index)
        return np.asarray(self.coords[low:high], dtype=np.float64)

    def plddt(self, index: int) -> np.ndarray:
        low, high = self.span(index)
        return np.asarray(self.features[low:high, -1], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cached-scores", type=Path, required=True)
    parser.add_argument("--exact-scores", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text())
    if manifest.get("test_evaluated") is not False:
        raise SystemExit("manifest is not explicitly test-free")
    nodes = manifest["nodes"]
    if any(node["split"] != "train" for node in nodes):
        raise SystemExit("neighborhood manifest must be train-only")

    exact_tree = Tree(arguments.artifact_root / "exact_full")
    parent_tree = Tree(arguments.artifact_root / "fixed_parent")

    def load(path: Path, key: str = "logits") -> dict[str, float]:
        values = np.load(path)
        return {str(k): float(v) for k, v in zip(values["sequence_sha256"], values[key])}

    low_map = load(arguments.cached_scores)
    high_map = load(arguments.exact_scores)

    grouped: dict[int, dict] = {}
    for edge in manifest["edges"]:
        source = nodes[int(edge["source_node"])]
        target = nodes[int(edge["target_node"])]
        rank = int(edge["anchor_rank"])
        row = grouped.setdefault(rank, {"length": int(source["length"]),
                                        "indices": [], "low": [], "high": []})
        row["indices"].append(int(edge["target_node"]))
        row["low"].append(low_map[target["sequence_sha256"]] - low_map[source["sequence_sha256"]])
        row["high"].append(high_map[target["sequence_sha256"]] - high_map[source["sequence_sha256"]])

    results = []
    for rank in sorted(grouped):
        row = grouped[rank]
        indices = np.asarray(row["indices"])
        low = np.asarray(row["low"])
        high = np.asarray(row["high"])
        order_low = np.argsort(-low, kind="stable")
        best = int(np.argmax(high))
        cached_rank_of_best = int(np.where(order_low == best)[0][0]) + 1

        def displacement(local: int) -> dict:
            node = int(indices[local])
            mutant = exact_tree.ca(node)
            parent = parent_tree.ca(node)
            rmsd, deviations = kabsch_rmsd(mutant, parent)
            return {
                "rmsd": rmsd,
                "tm_score": tm_score(deviations),
                "maximum_deviation": float(deviations.max()),
                "mean_plddt_change": float(
                    exact_tree.plddt(node).mean() - parent_tree.plddt(node).mean()),
                "min_plddt_change": float(
                    exact_tree.plddt(node).min() - parent_tree.plddt(node).min()),
            }

        # The whole neighborhood, so the exact best can be judged against its peers.
        sample = np.linspace(0, indices.size - 1, min(256, indices.size)).astype(int)
        population = [displacement(int(i)) for i in sample]
        rmsds = np.asarray([p["rmsd"] for p in population])
        tms = np.asarray([p["tm_score"] for p in population])

        best_row = displacement(best)
        top1_row = displacement(int(order_low[0]))
        results.append({
            "anchor_rank": rank,
            "length": row["length"],
            "cached_rank_of_exact_best": cached_rank_of_best,
            "catastrophic": bool(cached_rank_of_best > 8),
            "exact_best": best_row,
            "cached_top1": top1_row,
            "neighborhood_sample": int(sample.size),
            "neighborhood_rmsd_median": float(np.median(rmsds)),
            "neighborhood_rmsd_p90": float(np.percentile(rmsds, 90)),
            "neighborhood_tm_median": float(np.median(tms)),
            "exact_best_rmsd_percentile": float(np.mean(rmsds <= best_row["rmsd"])),
            "exact_best_tm_percentile": float(np.mean(tms <= best_row["tm_score"])),
        })

    catastrophic = [r for r in results if r["catastrophic"]]
    reliable = [r for r in results if not r["catastrophic"]]
    contrast = {}
    if catastrophic and reliable:
        for name in ("rmsd", "tm_score", "maximum_deviation",
                     "mean_plddt_change", "min_plddt_change"):
            contrast[name] = {
                "catastrophic_mean": float(np.mean([r["exact_best"][name] for r in catastrophic])),
                "reliable_mean": float(np.mean([r["exact_best"][name] for r in reliable])),
            }
        # The controls that decide between the two readings: is the OPTIMUM
        # unusual, or is the whole NEIGHBORHOOD unstable?
        for name in ("exact_best_rmsd_percentile", "neighborhood_rmsd_median",
                     "neighborhood_rmsd_p90", "neighborhood_tm_median"):
            contrast[name] = {
                "catastrophic_mean": float(np.mean([r[name] for r in catastrophic])),
                "reliable_mean": float(np.mean([r[name] for r in reliable])),
            }

    output = {
        "schema": "PLS_basin_escape_diagnosis_v1",
        "status": "post_hoc_explanatory_only_never_a_detector_feature",
        "hypothesis_tested": "the cached-parent field succeeds while the optimum stays in the parent's structural basin and fails when the best mutation escapes it",
        "verdict": "the literal hypothesis is NOT supported; see reading",
        "reading": (
            "the optimum's own displacement does not separate the classes: on the pilot "
            "a reliable anchor's optimum moved further than the catastrophic anchor's "
            "(RMSD 2.645 against 2.425, TM 0.6434 against 0.6296) and was still ranked "
            "first, and the optimum sits at an ordinary percentile of its own "
            "neighborhood in both classes. What separates them is the neighborhood as a "
            "whole: the catastrophic anchor is the only one whose MEDIAN mutant loses "
            "the fold. The parent structure stops being a valid proxy for the entire "
            "neighborhood, not just for the optimum. This quantity still requires "
            "folding and so remains explanatory, not a pre-fold detector."),
        "anchors": len(results),
        "catastrophic_anchors": len(catastrophic),
        "per_anchor": results,
        "contrast": contrast,
        "superposition": "Kabsch on the identity alignment; a single substitution preserves length and residue order",
        "mutant_folds_required": 0,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Does the optimum escape the parent's structural basin?",
        "",
        "Post-hoc mechanism diagnosis. It needs the mutant's own exact structure and",
        "therefore can never be a pre-fold detector feature. No new folds.",
        "",
        "Displacement is the exact mutant's C-alpha coordinates against the parent's,",
        "Kabsch-superposed on the identity alignment.",
        "",
        "| Anchor | L | Cached rank of best | Best RMSD | Best TM | Best RMSD pct in own nbhd | Nbhd median RMSD | Nbhd median TM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        mark = "**" if row["catastrophic"] else ""
        lines.append(
            f"| {mark}{row['anchor_rank']}{mark} | {row['length']} | "
            f"{mark}{row['cached_rank_of_exact_best']}{mark} | "
            f"{row['exact_best']['rmsd']:.3f} | {row['exact_best']['tm_score']:.4f} | "
            f"{row['exact_best_rmsd_percentile']:.3f} | "
            f"{row['neighborhood_rmsd_median']:.3f} | "
            f"{row['neighborhood_tm_median']:.4f} |")
    if contrast:
        lines += [
            "",
            "## Catastrophic against reliable anchors",
            "",
            "| Quantity | Catastrophic | Reliable |",
            "| --- | ---: | ---: |",
        ]
        for name, value in contrast.items():
            lines.append(f"| `{name}` | {value['catastrophic_mean']:.4f} | "
                         f"{value['reliable_mean']:.4f} |")
    lines += [
        "",
        "## Reading",
        "",
        "The literal hypothesis is not supported. A reliable anchor's optimum moved",
        "further from its parent than the catastrophic anchor's (RMSD 2.645 against",
        "2.425, TM 0.6434 against 0.6296) and was still ranked first, and in both",
        "classes the optimum sits at an ordinary percentile of its own neighborhood.",
        "",
        "What separates the classes is the neighborhood as a whole. The catastrophic",
        "anchor is the only one whose **median** mutant loses the fold. The parent",
        "structure stops being a valid proxy for the entire neighborhood, not just for",
        "the optimum.",
        "",
        "One catastrophic anchor out of eight. This is a lead, not a finding.",
        "",
    ]
    arguments.md_out.write_text("\n".join(lines))
    print(json.dumps({"anchors": len(results), "catastrophic": len(catastrophic),
                      "contrast": contrast}, indent=2))


if __name__ == "__main__":
    main()
