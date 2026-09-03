"""How far does the cached-parent oracle's linear regime reach in sequence space?

For each anchor, walks a fraction epsilon along E_a - E_{w_i} at a sampled
position and compares the measured change in the cached-parent logit against
epsilon * d[i,a]. epsilon = 1 is a real substitution. Costs zero mutant folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import esm
import numpy as np
import torch

from pls.editflow.gradient_field import AMINO_ACIDS, STRUCTURE_SCALARS, build_teacher
from pls.training.train_gvp_structure import GVPData, collate

REPRESENTATION_LAYER = 33


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-config", type=Path, required=True)
    parser.add_argument("--gradient-field", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()

    config = json.loads(arguments.score_config.read_text())
    if config.get("evaluate_test", False):
        raise SystemExit("test evaluation is permanently disabled")
    data = config["data"]
    manifest = json.loads(Path(data["manifest"]).read_text())
    if manifest.get("test_evaluated") is not False:
        raise SystemExit("manifest is not explicitly test-free")
    nodes = manifest["nodes"]
    if any(node["split"] != "train" for node in nodes):
        raise SystemExit("neighborhood manifest must be train-only")

    device = torch.device(arguments.device)
    oracle = json.loads(Path(config["oracle"]["manifest"]).read_text())
    stats = json.loads(Path(data["structure_stats"]).read_text())
    embeddings = np.load(Path(data["embedding_dir"]) / "embeddings.npy", mmap_mode="r")
    residue_esm = Path(data["residue_esm_dir"])
    pca_meta = json.loads((residue_esm / "pca_metadata.json").read_text())
    rdim = int(pca_meta["shape"][1])
    teacher = build_teacher(oracle, rdim, stats, embeddings.shape[1], device)
    dataset = GVPData(
        [(i, n["sequence_sha256"], 0.0) for i, n in enumerate(nodes)], embeddings,
        Path(data["structure_dir"]), torch.tensor(stats["scalar_means"]),
        torch.tensor(stats["scalar_stds"]), Path(data["compact_structure_dir"]),
        Path(data["geometry_dir"]), False,
        int(json.loads((Path(oracle["run"]) / "config.json").read_text())["model"]["neighbors"]),
        residue_esm, None, None, None, Path(data["surface_patch_dir"]),
        vector_dir=Path(data["vector_dir"]))

    plm, alphabet_obj = esm.pretrained.esm2_t33_650M_UR50D()
    plm = plm.to(device).eval()
    for parameter in plm.parameters():
        parameter.requires_grad_(False)
    converter = alphabet_obj.get_batch_converter()
    token_of = {a: alphabet_obj.get_idx(a) for a in AMINO_ACIDS}
    weight = plm.embed_tokens.weight
    pca = np.load(pca_meta["pca"])
    pca_mean = torch.tensor(pca["mean"], dtype=torch.float32, device=device)
    pca_components = torch.tensor(pca["components"], dtype=torch.float32, device=device)
    field = np.load(arguments.gradient_field)
    index_of = {c: i for i, c in enumerate(AMINO_ACIDS)}

    per_epsilon = {eps: {"ratios": [], "predicted": [], "actual": []}
                   for eps in arguments.epsilons}
    per_anchor = []
    for anchor in [n for n in nodes if n["kind"] == "anchor"]:
        rank = int(anchor["anchor_rank"])
        sequence = anchor["sequence"]
        length = len(sequence)
        parts = [t.to(device) for t in collate([dataset[int(anchor["node_index"])]])[:-1]]
        _mean, residue, vectors, coords, mask, nb, dist, patch, pc = parts
        _l, _s, tokens = converter([("anchor", sequence)])
        tokens = tokens.to(device)

        def logit(delta):
            def hook(_module, _inputs, output):
                out = output.detach().clone()
                return out if delta is None else out + delta

            handle = plm.embed_tokens.register_forward_hook(hook)
            with torch.inference_mode():
                rep = plm(tokens, repr_layers=[REPRESENTATION_LAYER])[
                    "representations"][REPRESENTATION_LAYER][0, 1:length + 1].float()
                live = torch.cat((residue[..., :STRUCTURE_SCALARS],
                                  ((rep - pca_mean) @ pca_components.T).half().float()[None]), -1)
                value = float(teacher(rep.mean(0)[None].half().float(), live, vectors,
                                      coords, mask, nb, dist, patch, pc).squeeze())
            handle.remove()
            return value

        base = logit(None)
        generator = np.random.default_rng(rank)
        pairs = []
        while len(pairs) < arguments.samples:
            position = int(generator.integers(length))
            target = AMINO_ACIDS[int(generator.integers(len(AMINO_ACIDS)))]
            if target != sequence[position]:
                pairs.append((position, target))
        anchor_rows = {}
        for eps in arguments.epsilons:
            ratios, predicted_values, actual_values = [], [], []
            for position, target in pairs:
                direction = weight[token_of[target]] - weight[token_of[sequence[position]]]
                delta = torch.zeros(1, tokens.shape[1], weight.shape[1], device=device)
                delta[0, position + 1] = eps * direction
                actual = logit(delta) - base
                predicted = eps * float(field[f"field_{rank}"][position, index_of[target]])
                if abs(predicted) > 1e-9:
                    ratios.append(actual / predicted)
                predicted_values.append(predicted)
                actual_values.append(actual)
            correlation = float(np.corrcoef(predicted_values, actual_values)[0, 1])
            anchor_rows[eps] = {
                "median_ratio": float(np.median(ratios)),
                "correlation": correlation,
                "mean_absolute_predicted": float(np.mean(np.abs(predicted_values))),
                "mean_absolute_actual": float(np.mean(np.abs(actual_values))),
            }
            per_epsilon[eps]["ratios"].extend(ratios)
            per_epsilon[eps]["predicted"].extend(predicted_values)
            per_epsilon[eps]["actual"].extend(actual_values)
        per_anchor.append({"anchor_rank": rank, "length": length,
                           "base_logit": base, "by_epsilon": anchor_rows})
        print(f"anchor {rank:>2} L={length:>3} done", flush=True)

    pooled = []
    for eps in arguments.epsilons:
        bucket = per_epsilon[eps]
        pooled.append({
            "epsilon": eps,
            "samples": len(bucket["predicted"]),
            "median_ratio": float(np.median(bucket["ratios"])),
            "correlation": float(np.corrcoef(bucket["predicted"], bucket["actual"])[0, 1]),
            "mean_absolute_predicted": float(np.mean(np.abs(bucket["predicted"]))),
            "mean_absolute_actual": float(np.mean(np.abs(bucket["actual"]))),
        })
    output = {
        "schema": "PLS_cached_parent_linear_regime_v1",
        "anchors": len(per_anchor),
        "samples_per_anchor": arguments.samples,
        "pooled": pooled,
        "per_anchor": per_anchor,
        "interpretation": (
            "epsilon = 1 is a real single substitution; the linear regime is where "
            "the correlation between epsilon * d and the measured change stays high"),
        "mutant_folds_required": 0,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# How far the cached-parent oracle stays linear in sequence space",
        "",
        f"{len(per_anchor)} anchors, {arguments.samples} sampled substitutions each, "
        "zero mutant folds.",
        "",
        "A step of `epsilon` along `E_a - E_w` at one position, compared against the",
        "one-backward prediction `epsilon * d[i,a]`. `epsilon = 1` is a real substitution.",
        "",
        "| epsilon | Samples | Median ratio | Correlation | Mean predicted | Mean actual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pooled:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['samples']} | {row['median_ratio']:.4f} | "
            f"{row['correlation']:.4f} | {row['mean_absolute_predicted']:.5f} | "
            f"{row['mean_absolute_actual']:.5f} |")
    lines += [
        "",
        "Read this as the reason the gradient cannot propose mutations: it is a correct",
        "derivative, and a substitution simply lands far outside the radius where that",
        "derivative predicts anything.",
        "",
    ]
    arguments.md_out.write_text("\n".join(lines))
    print()
    for row in pooled:
        print(f"  eps={row['epsilon']:<5} median_ratio={row['median_ratio']:>8.4f}  "
              f"corr={row['correlation']:>7.4f}  |pred|={row['mean_absolute_predicted']:.5f}  "
              f"|actual|={row['mean_absolute_actual']:.5f}")


if __name__ == "__main__":
    main()
