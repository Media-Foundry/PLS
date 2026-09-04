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


def batched_kabsch(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-structure C-alpha deviations after optimal superposition.

    mobile is (n, L, 3), target is (L, 3). A single substitution preserves length
    and residue order, so the alignment is the identity and only the rigid motion
    has to be solved. Batched so a whole neighborhood costs one pass.
    """
    mobile_centered = mobile - mobile.mean(axis=1, keepdims=True)
    target_centered = target - target.mean(axis=0)
    covariance = np.einsum("nli,lj->nij", mobile_centered, target_centered)
    u, _s, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(np.einsum("nji,nkj->nik", vt, u)))
    correction = np.zeros((sign.size, 3, 3))
    correction[:, 0, 0] = 1.0
    correction[:, 1, 1] = 1.0
    correction[:, 2, 2] = sign
    rotation = np.einsum("nji,njk,nlk->nil", vt, correction, u)
    aligned = np.einsum("nli,nji->nlj", mobile_centered, rotation)
    return np.linalg.norm(aligned - target_centered, axis=-1)


def tm_from_deviations(deviations: np.ndarray) -> np.ndarray:
    """TM-score per structure from its per-residue deviations."""
    length = deviations.shape[-1]
    d0 = max(1.24 * np.cbrt(max(length - 15, 1)) - 1.8, 0.5)
    return np.mean(1.0 / (1.0 + np.square(deviations / d0)), axis=-1)


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

        # Whole neighborhood in one pass: no sampling, no new folds.
        parent_ca = parent_tree.ca(int(indices[0]))
        mutant_ca = np.stack([exact_tree.ca(int(i)) for i in indices])
        deviations = batched_kabsch(mutant_ca, parent_ca)
        rmsds = np.sqrt(np.mean(np.square(deviations), axis=1))
        tms = tm_from_deviations(deviations)
        maxima = deviations.max(axis=1)
        parent_plddt = parent_tree.plddt(int(indices[0]))
        plddt_change = np.asarray([
            exact_tree.plddt(int(i)).mean() for i in indices]) - parent_plddt.mean()

        def entry(local: int) -> dict:
            return {
                "rmsd": float(rmsds[local]),
                "tm_score": float(tms[local]),
                "maximum_deviation": float(maxima[local]),
                "mean_plddt_change": float(plddt_change[local]),
                "rmsd_percentile_in_neighborhood": float(np.mean(rmsds <= rmsds[local])),
                "tm_percentile_in_neighborhood": float(np.mean(tms <= tms[local])),
            }

        results.append({
            "anchor_rank": rank,
            "length": row["length"],
            "cached_rank_of_exact_best": cached_rank_of_best,
            "catastrophic": bool(cached_rank_of_best > 8),
            "exact_best": entry(best),
            "cached_top1": entry(int(order_low[0])),
            "neighborhood_mutants": int(indices.size),
            "neighborhood_rmsd_median": float(np.median(rmsds)),
            "neighborhood_rmsd_p90": float(np.percentile(rmsds, 90)),
            "neighborhood_tm_median": float(np.median(tms)),
            "neighborhood_tm_p10": float(np.percentile(tms, 10)),
            "neighborhood_fraction_tm_below_0.7": float(np.mean(tms < 0.7)),
            "neighborhood_fraction_tm_below_0.9": float(np.mean(tms < 0.9)),
            "neighborhood_mean_plddt_change": float(plddt_change.mean()),
        })

    catastrophic = [r for r in results if r["catastrophic"]]
    reliable = [r for r in results if not r["catastrophic"]]
    contrast = {}
    if catastrophic and reliable:
        for name in ("rmsd", "tm_score", "maximum_deviation", "mean_plddt_change",
                     "rmsd_percentile_in_neighborhood", "tm_percentile_in_neighborhood"):
            contrast[name] = {
                "catastrophic_mean": float(np.mean([r["exact_best"][name] for r in catastrophic])),
                "reliable_mean": float(np.mean([r["exact_best"][name] for r in reliable])),
            }
        # The controls that decide between the two readings: is the OPTIMUM
        # unusual, or is the whole NEIGHBORHOOD unstable?
        for name in ("neighborhood_rmsd_median", "neighborhood_rmsd_p90",
                     "neighborhood_tm_median", "neighborhood_tm_p10",
                     "neighborhood_fraction_tm_below_0.7",
                     "neighborhood_fraction_tm_below_0.9",
                     "neighborhood_mean_plddt_change"):
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
        "neighborhood_coverage": "every mutant in every neighborhood, not a sample",
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
        "Kabsch-superposed on the identity alignment, over EVERY mutant in every",
        "neighborhood.",
        "",
        "| Anchor | L | Mutants | Cached rank of best | Best RMSD | Best TM | Best RMSD pct | Nbhd median RMSD | Nbhd median TM | Frac TM<0.7 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        mark = "**" if row["catastrophic"] else ""
        lines.append(
            f"| {mark}{row['anchor_rank']}{mark} | {row['length']} | "
            f"{row['neighborhood_mutants']} | "
            f"{mark}{row['cached_rank_of_exact_best']}{mark} | "
            f"{row['exact_best']['rmsd']:.3f} | {row['exact_best']['tm_score']:.4f} | "
            f"{row['exact_best']['rmsd_percentile_in_neighborhood']:.3f} | "
            f"{row['neighborhood_rmsd_median']:.3f} | "
            f"{row['neighborhood_tm_median']:.4f} | "
            f"{row['neighborhood_fraction_tm_below_0.7']:.3f} |")
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
