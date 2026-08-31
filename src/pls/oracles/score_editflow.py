"""Score safe PLS EditFlow sequences with one frozen coherent GVP teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pls.models.gvp_structure import GVPStructureFusion
from pls.training.train_gvp_structure import GVPData, collate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safe_nodes(manifest_path: Path, entities_path: Path) -> tuple[list[dict], list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("oracle scoring refuses manifests without a test-free assertion")
    nodes = manifest["nodes"]
    if any(node["split"] not in {"train", "validation"} for node in nodes):
        raise ValueError("oracle scoring manifest contains a forbidden split")
    with entities_path.open(newline="", encoding="utf-8") as handle:
        entities = list(csv.DictReader(handle))
    if len(entities) != len(nodes):
        raise ValueError("oracle entities and node manifest differ in length")
    for index, (node, entity) in enumerate(zip(nodes, entities)):
        if int(node["node_index"]) != index:
            raise ValueError("oracle node order is not canonical")
        if node["sequence_sha256"] != entity["sequence_sha256"]:
            raise ValueError("oracle node and entity identity mismatch")
        if node["sequence"] != entity["sequence"]:
            raise ValueError("oracle node and entity sequence mismatch")
    return nodes, entities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    if config.get("evaluate_test", False):
        parser.error("test evaluation is permanently disabled")
    inference = config["inference"]
    data = config["data"]
    nodes, entities = load_safe_nodes(Path(data["manifest"]), Path(data["entities"]))
    oracle_manifest_path = Path(config["oracle"]["manifest"])
    oracle_manifest = json.loads(oracle_manifest_path.read_text())
    if oracle_manifest.get("test_evaluated") is not False:
        raise ValueError("frozen oracle manifest violates the permanent test freeze")
    if oracle_manifest.get("output") != "raw_logit":
        raise ValueError("EditFlow requires raw teacher logits")
    checkpoint = Path(oracle_manifest["checkpoint"]["path"])
    checkpoint_sha = file_sha256(checkpoint)
    if checkpoint_sha != oracle_manifest["checkpoint"]["sha256"]:
        raise ValueError("frozen teacher checkpoint SHA-256 mismatch")
    required_paths = {
        "mean_embeddings": Path(data["embedding_dir"]) / "embeddings.npy",
        "structure_status": Path(data["structure_status"]),
        "compact_metadata": Path(data["compact_structure_dir"]) / "metadata.json",
        "geometry_metadata": Path(data["geometry_dir"]) / "metadata.json",
        "residue_esm_metadata": Path(data["residue_esm_dir"]) / "pca_metadata.json",
        "vector_metadata": Path(data["vector_dir"]) / "metadata.json",
        "surface_patch_metadata": Path(data["surface_patch_dir"]) / "metadata.json",
    }
    if arguments.dry_run:
        readiness = {
            name: path.is_file() for name, path in required_paths.items()
        }
        print(json.dumps({
            "mode": "dry_run",
            "safe_queries": len(nodes),
            "checkpoint_sha256_verified": True,
            "features": readiness,
            "ready": all(readiness.values()),
            "test_evaluated": False,
        }, indent=2, sort_keys=True))
        return
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(inference["hip_device"]):
        parser.error("HIP device mismatch")

    status = np.load(data["structure_status"], mmap_mode="r")
    if status.shape != (len(nodes),) or np.any(status != 1):
        raise ValueError("exact-sequence structure features are incomplete")
    embeddings = np.load(Path(data["embedding_dir"]) / "embeddings.npy", mmap_mode="r")
    if embeddings.shape[0] != len(nodes) or not np.isfinite(embeddings).all():
        raise ValueError("mean ESM-2 embeddings are incomplete")
    stats = json.loads(Path(data["structure_stats"]).read_text())
    mean = torch.tensor(stats["scalar_means"])
    standard_deviation = torch.tensor(stats["scalar_stds"])
    compact = Path(data["compact_structure_dir"])
    geometry = Path(data["geometry_dir"])
    residue_esm = Path(data["residue_esm_dir"])
    vector_dir = Path(data["vector_dir"])
    surface_patches = Path(data["surface_patch_dir"])
    rows = [
        (index, node["sequence_sha256"], 0.0)
        for index, node in enumerate(nodes)
    ]
    teacher_config = json.loads(
        (Path(oracle_manifest["run"]) / "config.json").read_text()
    )
    model_config = teacher_config["model"]
    dataset = GVPData(
        rows,
        embeddings,
        Path(data["structure_dir"]),
        mean,
        standard_deviation,
        compact,
        geometry,
        False,
        int(model_config["neighbors"]),
        residue_esm,
        None,
        None,
        None,
        surface_patches,
        vector_dir=vector_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(inference["batch_size"]),
        shuffle=False,
        num_workers=int(inference["workers"]),
        collate_fn=collate,
        persistent_workers=int(inference["workers"]) > 0,
        pin_memory=True,
    )
    residue_sequence_dimension = int(
        json.loads((residue_esm / "pca_metadata.json").read_text())["shape"][1]
    )
    residue_dimension = 152 + residue_sequence_dimension
    device = torch.device("cuda:0")
    model = GVPStructureFusion(
        embeddings.shape[1],
        residue_dimension,
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
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    logits = []
    with torch.inference_mode():
        for sequence, residue, vectors, coordinates, mask, neighbors, distances, patch, components, _ in loader:
            values = [
                value.to(device, non_blocking=True)
                for value in (
                    sequence, residue, vectors, coordinates, mask,
                    neighbors, distances, patch, components,
                )
            ]
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=bool(inference.get("amp_bfloat16", True)),
            ):
                prediction = model(*values)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("frozen oracle produced a non-finite logit")
            logits.extend(prediction.float().cpu().tolist())
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.shape != (len(nodes),):
        raise RuntimeError("frozen oracle output count mismatch")
    arguments.output_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        arguments.output_root / "oracle_logits.npz",
        node_indices=np.arange(len(nodes), dtype=np.int64),
        sequence_sha256=np.asarray([node["sequence_sha256"] for node in nodes]),
        logits=logits_array,
    )
    report = {
        "schema": "PLS_EditFlow_frozen_GVP_oracle_scores_v1",
        "output": "raw_logit",
        "queries": len(nodes),
        "queries_by_split": {
            split: sum(node["split"] == split for node in nodes)
            for split in ("train", "validation")
        },
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(state["epoch"]),
        "oracle_manifest_sha256": file_sha256(oracle_manifest_path),
        "logit_minimum": float(logits_array.min()),
        "logit_maximum": float(logits_array.max()),
        "logit_mean": float(logits_array.mean()),
        "test_evaluated": False,
    }
    (arguments.output_root / "oracle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
