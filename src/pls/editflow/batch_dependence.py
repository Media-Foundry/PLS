"""The frozen teacher's logit depends on the batch it was scored in.

`pool_surface_patches` derives `label_stride` from the maximum surface-patch
component id across the whole batch, so the group encoding, the `torch.unique`
ordering, the top-64 patch selection and the `index_add_` accumulation order all
shift with batch composition. The scores are reproducible when the batching is
reproduced, and only then. This quantifies the size of the effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pls.editflow.gradient_field import build_teacher
from pls.training.train_gvp_structure import GVPData, collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-config", type=Path, required=True)
    parser.add_argument("--cached-scores", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()

    config = json.loads(arguments.score_config.read_text())
    data = config["data"]
    manifest = json.loads(Path(data["manifest"]).read_text())
    if manifest.get("test_evaluated") is not False:
        raise SystemExit("manifest is not explicitly test-free")
    nodes = manifest["nodes"]
    device = torch.device(arguments.device)
    oracle = json.loads(Path(config["oracle"]["manifest"]).read_text())
    stats = json.loads(Path(data["structure_stats"]).read_text())
    embeddings = np.load(Path(data["embedding_dir"]) / "embeddings.npy", mmap_mode="r")
    residue_esm = Path(data["residue_esm_dir"])
    rdim = int(json.loads((residue_esm / "pca_metadata.json").read_text())["shape"][1])
    teacher = build_teacher(oracle, rdim, stats, embeddings.shape[1], device)
    dataset = GVPData(
        [(i, n["sequence_sha256"], 0.0) for i, n in enumerate(nodes)], embeddings,
        Path(data["structure_dir"]), torch.tensor(stats["scalar_means"]),
        torch.tensor(stats["scalar_stds"]), Path(data["compact_structure_dir"]),
        Path(data["geometry_dir"]), False,
        int(json.loads((Path(oracle["run"]) / "config.json").read_text())["model"]["neighbors"]),
        residue_esm, None, None, None, Path(data["surface_patch_dir"]),
        vector_dir=Path(data["vector_dir"]))
    cached = np.load(arguments.cached_scores)
    frozen = {str(k): float(v) for k, v in zip(cached["sequence_sha256"], cached["logits"])}
    batch_size = int(config["inference"]["batch_size"])

    generator = np.random.default_rng(0)
    sample = sorted(generator.choice(len(nodes), size=min(arguments.nodes, len(nodes)),
                                     replace=False).tolist())
    alone_gap, batch_gap = [], []
    for index in sample:
        start = (index // batch_size) * batch_size
        members = list(range(start, min(start + batch_size, len(nodes))))
        with torch.inference_mode():
            alone = float(teacher(*[t.to(device) for t in collate([dataset[index]])[:-1]]).squeeze())
            grouped = teacher(*[t.to(device) for t in
                                collate([dataset[j] for j in members])[:-1]]).squeeze()
            in_batch = float(grouped[members.index(index)])
        reference = frozen[nodes[index]["sequence_sha256"]]
        alone_gap.append(abs(alone - reference))
        batch_gap.append(abs(in_batch - reference))

    alone_gap = np.asarray(alone_gap)
    batch_gap = np.asarray(batch_gap)
    output = {
        "schema": "PLS_teacher_batch_dependence_v1",
        "nodes_sampled": len(sample),
        "scorer_batch_size": batch_size,
        "reproducing_the_scorer_batching": {
            "max_absolute_gap": float(batch_gap.max()),
            "median_absolute_gap": float(np.median(batch_gap)),
            "p99_absolute_gap": float(np.percentile(batch_gap, 99)),
        },
        "scoring_one_sequence_at_a_time": {
            "max_absolute_gap": float(alone_gap.max()),
            "median_absolute_gap": float(np.median(alone_gap)),
            "p99_absolute_gap": float(np.percentile(alone_gap, 99)),
        },
        "cause": (
            "pool_surface_patches derives label_stride from the batch-wide maximum "
            "component id, so group ordering, top-64 patch selection and index_add_ "
            "accumulation order depend on batch composition"),
        "consequence": (
            "frozen scores are reproducible only when the batching is reproduced; "
            "they are not a pure function of a single sequence"),
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
