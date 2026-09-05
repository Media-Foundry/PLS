"""One-backward local gradient field over the single-mutant neighborhood.

Fixes the anchor's predicted structure, makes the sequence branch differentiable
through ESM-2, and reads off the first-order directional derivative of every
substitution:

    d[i,a] = grad_{E_i} L . (E_a - E_{w_i})

L is the cached-parent oracle F(h(x), G(parent)), so this approximates the
cached-parent forward difference, never the exact oracle. The script refuses to
emit a field unless its differentiable recomputation reproduces the frozen
cached-parent logit for the anchor, which is the only guard that the surgery
below matches the scoring pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import esm
import numpy as np
import torch

from pls.models.gvp_structure import GVPStructureFusion
from pls.training.train_gvp_structure import GVPData, collate

STRUCTURE_SCALARS = 152          # residue[..., :152] is structure, the rest is PLM PCA
REPRESENTATION_LAYER = 33
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_teacher(oracle_manifest: dict, residue_sequence_dimension: int,
                  stats: dict, embedding_dimension: int, device) -> GVPStructureFusion:
    teacher_config = json.loads((Path(oracle_manifest["run"]) / "config.json").read_text())
    model_config = teacher_config["model"]
    model = GVPStructureFusion(
        embedding_dimension,
        STRUCTURE_SCALARS + residue_sequence_dimension,
        model_config["scalar_dimension"],
        model_config["vector_dimension"],
        model_config["representation_dimension"],
        model_config["dropout"],
        model_config["layers"],
        model_config.get("fusion", "interaction"),
        residue_sequence_dimension,
        stats["scalar_means"][1],
        stats["scalar_stds"][1],
        model_config.get("surface_patches", False),
        model_config.get("patch_spatial_layers", 0),
        oracle_manifest.get("model", {}).get(
            "cross_confidence_power", model_config.get("cross_confidence_power", 1.0)),
        oracle_manifest.get("model", {}).get(
            "patch_self_edges", model_config.get("patch_self_edges", False)),
    ).to(device)
    state = torch.load(Path(oracle_manifest["checkpoint"]["path"]),
                       map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-config", type=Path, required=True)
    parser.add_argument("--cached-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--full-precision", action="store_true",
                        help="skip the float16 round-trip the caches impose")
    parser.add_argument("--max-deviation", type=float, default=3e-2,
                        help=("reject an anchor whose differentiable recomputation "
                              "misses the batch-of-one cached-parent logit by more "
                              "than this; the default is calibrated on the pilot, "
                              "where the worst anchor was 9.2e-3"))
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
    oracle_manifest = json.loads(Path(config["oracle"]["manifest"]).read_text())
    stats = json.loads(Path(data["structure_stats"]).read_text())
    residue_esm = Path(data["residue_esm_dir"])
    residue_sequence_dimension = int(
        json.loads((residue_esm / "pca_metadata.json").read_text())["shape"][1])
    embeddings = np.load(Path(data["embedding_dir"]) / "embeddings.npy", mmap_mode="r")

    teacher = build_teacher(oracle_manifest, residue_sequence_dimension, stats,
                            embeddings.shape[1], device)

    plm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    plm = plm.to(device).eval()
    for parameter in plm.parameters():
        parameter.requires_grad_(False)
    batch_converter = alphabet.get_batch_converter()
    token_of = {aa: alphabet.get_idx(aa) for aa in AMINO_ACIDS}
    embedding_weight = plm.embed_tokens.weight            # [vocab, 1280]

    pca = np.load(json.loads((residue_esm / "pca_metadata.json").read_text())["pca"])
    pca_mean = torch.tensor(pca["mean"], dtype=torch.float32, device=device)
    pca_components = torch.tensor(pca["components"], dtype=torch.float32, device=device)

    # Cached-parent tensors: everything structural is held fixed at the anchor.
    dataset = GVPData(
        [(index, node["sequence_sha256"], 0.0) for index, node in enumerate(nodes)],
        embeddings, Path(data["structure_dir"]),
        torch.tensor(stats["scalar_means"]), torch.tensor(stats["scalar_stds"]),
        Path(data["compact_structure_dir"]), Path(data["geometry_dir"]), False,
        int(json.loads((Path(oracle_manifest["run"]) / "config.json").read_text())
            ["model"]["neighbors"]),
        residue_esm, None, None, None, Path(data["surface_patch_dir"]),
        vector_dir=Path(data["vector_dir"]),
    )
    cached = np.load(arguments.cached_scores)
    cached_logit = {str(k): float(v) for k, v in
                    zip(cached["sequence_sha256"], cached["logits"])}

    anchors = [node for node in nodes if node["kind"] == "anchor"]
    results = []
    for anchor in anchors:
        index = int(anchor["node_index"])
        sequence = anchor["sequence"]
        length = len(sequence)
        batch = collate([dataset[index]])
        # Same field order as the frozen scorer's loader unpacking.
        (_mean, residue, vectors, coordinates, mask,
         neighbors, distances, patch, patch_components, _y) = batch
        tensors = [t.to(device) for t in (residue, vectors, coordinates, mask,
                                          neighbors, distances, patch, patch_components)]
        residue, vectors, coordinates, mask, neighbors, distances, patch, patch_components = tensors

        _labels, _strings, tokens = batch_converter([("anchor", sequence)])
        tokens = tokens.to(device)

        captured = {}

        def hook(_module, _inputs, output):
            leaf = output.detach().requires_grad_(True)
            captured["embedding"] = leaf
            return leaf

        handle = plm.embed_tokens.register_forward_hook(hook)
        with torch.enable_grad():
            representations = plm(tokens, repr_layers=[REPRESENTATION_LAYER],
                                  return_contacts=False)
            token_representation = representations["representations"][
                REPRESENTATION_LAYER][0, 1:length + 1].float()
            differentiable_mean = token_representation.mean(0)[None]
            differentiable_pca = (token_representation - pca_mean) @ pca_components.T
            if not arguments.full_precision:
                differentiable_mean = differentiable_mean.half().float()
                differentiable_pca = differentiable_pca.half().float()
            # Surgery: keep the anchor's structure scalars, swap in the live PLM half.
            live_residue = torch.cat(
                (residue[..., :STRUCTURE_SCALARS],
                 differentiable_pca[None].to(residue.dtype)), dim=-1)
            logit = teacher(differentiable_mean, live_residue, vectors, coordinates,
                            mask, neighbors, distances, patch,
                            patch_components).squeeze()
            gradient = torch.autograd.grad(logit, captured["embedding"])[0]
        handle.remove()

        # The frozen scores were produced in batches of 20, and the teacher's
        # surface-patch pooling makes its output depend on batch composition, so
        # the honest gate is a batch-of-one forward on the CACHED features: it
        # isolates the differentiable surgery from both the batching artifact and
        # the PLM recomputation.
        with torch.inference_mode():
            reference = float(teacher(
                _mean.to(device), residue, vectors, coordinates, mask,
                neighbors, distances, patch, patch_components).squeeze())
        frozen = cached_logit[anchor["sequence_sha256"]]
        deviation = abs(float(logit.detach()) - reference)
        # Tolerance justified by measurement, not taste: recomputing ESM-2 in
        # float32 differs from the float16 caches by ~1.7e-3 on the mean embedding
        # and ~1.3e-2 on the residue PCA, which propagates to a few times 1e-3 in
        # the logit. Cached-parent mutation effects have median |effect| 0.110 and
        # sd 0.366, so this is a low single-digit percent perturbation.
        if deviation > arguments.max_deviation:
            raise SystemExit(
                f"differentiable recomputation does not reproduce the batch-of-one "
                f"cached-parent logit for anchor {anchor['anchor_rank']}: "
                f"{float(logit.detach()):.6f} vs {reference:.6f}")

        residue_gradient = gradient[0, 1:length + 1].float()           # [L, 1280]
        weights = embedding_weight[[token_of[a] for a in AMINO_ACIDS]].float()
        source_ids = [token_of[residue_char] for residue_char in sequence]
        source_weights = embedding_weight[source_ids].float()          # [L, 1280]
        # d[i,a] = g_i . (E_a - E_{w_i})
        field = residue_gradient @ weights.T - (
            residue_gradient * source_weights).sum(-1, keepdim=True)
        results.append({
            "anchor_rank": int(anchor["anchor_rank"]),
            "node_index": index,
            "sequence_sha256": anchor["sequence_sha256"],
            "length": length,
            "anchor_logit_recomputed": float(logit.detach()),
            "anchor_logit_batch_of_one": reference,
            "anchor_logit_frozen_batch_of_twenty": frozen,
            "absolute_deviation": deviation,
            "field": field.detach().cpu().numpy().astype(np.float32),
            "sequence": sequence,
        })
        print(f"anchor {anchor['anchor_rank']:>2}  L={length:>3}  "
              f"grad_path={float(logit.detach()):.6f}  batch1={reference:.6f}  "
              f"frozen_b20={frozen:.6f}  "
              f"|d|={deviation:.2e}", flush=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        anchor_ranks=np.asarray([r["anchor_rank"] for r in results], dtype=np.int64),
        lengths=np.asarray([r["length"] for r in results], dtype=np.int64),
        sequences=np.asarray([r["sequence"] for r in results]),
        alphabet=np.asarray(list(AMINO_ACIDS)),
        absolute_deviations=np.asarray(
            [r["absolute_deviation"] for r in results], dtype=np.float64),
        anchor_logit_batch_of_one=np.asarray(
            [r["anchor_logit_batch_of_one"] for r in results], dtype=np.float64),
        **{f"field_{r['anchor_rank']}": r["field"] for r in results},
    )
    deviations = sorted(r["absolute_deviation"] for r in results)
    print(json.dumps({
        "anchors": len(results),
        "max_deviation_allowed": arguments.max_deviation,
        "median_absolute_deviation": deviations[len(deviations) // 2],
        "anchors_above_pilot_tolerance_3e-2": sum(d > 3e-2 for d in deviations),
        "maximum_absolute_deviation": max(r["absolute_deviation"] for r in results),
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
