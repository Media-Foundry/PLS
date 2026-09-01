"""Score safe PLS EditFlow sequences with one frozen coherent GVP teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
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


def matched_sequence_only_logits(model: GVPStructureFusion, sequence: torch.Tensor) -> torch.Tensor:
    """Use the frozen checkpoint's own sequence branch with structure bypassed."""
    if not model.aligned_fusion or not hasattr(model, "sequence_head"):
        raise ValueError("matched sequence ablation requires an aligned MoE teacher")
    return model.sequence_head(model.sequence(sequence)).squeeze(-1)


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
    accelerator_backend = str(inference.get("accelerator_backend", "rocm"))
    if accelerator_backend == "rocm":
        if os.environ.get("HIP_VISIBLE_DEVICES") != str(inference["hip_device"]):
            parser.error("HIP device mismatch")
    elif accelerator_backend == "cuda_slurm":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not os.environ.get("SLURM_JOB_ID") or not visible or "," in visible:
            parser.error("cuda_slurm requires one Slurm-assigned visible GPU")
    else:
        parser.error("accelerator_backend must be rocm or cuda_slurm")

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
        oracle_manifest.get("model", {}).get(
            "cross_confidence_power",
            model_config.get("cross_confidence_power", 1.0),
        ),
        oracle_manifest.get("model", {}).get(
            "patch_self_edges", model_config.get("patch_self_edges", False)
        ),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    include_sequence_ablation = bool(
        config["oracle"].get("include_matched_sequence_ablation", False)
    )
    logits = []
    sequence_only_logits = []
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
                if include_sequence_ablation:
                    sequence_prediction = matched_sequence_only_logits(model, values[0])
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("frozen oracle produced a non-finite logit")
            logits.extend(prediction.float().cpu().tolist())
            if include_sequence_ablation:
                if not torch.isfinite(sequence_prediction).all():
                    raise FloatingPointError("matched sequence ablation produced a non-finite logit")
                sequence_only_logits.extend(sequence_prediction.float().cpu().tolist())
    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.shape != (len(nodes),):
        raise RuntimeError("frozen oracle output count mismatch")
    arguments.output_root.mkdir(parents=True, exist_ok=False)
    arrays = {
        "node_indices": np.arange(len(nodes), dtype=np.int64),
        "sequence_sha256": np.asarray([node["sequence_sha256"] for node in nodes]),
        "logits": logits_array,
    }
    sequence_array = None
    if include_sequence_ablation:
        sequence_array = np.asarray(sequence_only_logits, dtype=np.float32)
        if sequence_array.shape != logits_array.shape:
            raise RuntimeError("matched sequence ablation output count mismatch")
        arrays["sequence_only_logits"] = sequence_array
    np.savez_compressed(
        arguments.output_root / "oracle_logits.npz",
        **arrays,
    )
    report = {
        "schema": (
            "PLS_EditFlow_frozen_GVP_oracle_scores_with_matched_ablation_v2"
            if include_sequence_ablation else "PLS_EditFlow_frozen_GVP_oracle_scores_v1"
        ),
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
    if sequence_array is not None:
        residual = logits_array - sequence_array
        report["matched_sequence_ablation"] = {
            "definition": "same_checkpoint_sequence_projection_and_sequence_head",
            "logit_minimum": float(sequence_array.min()),
            "logit_maximum": float(sequence_array.max()),
            "logit_mean": float(sequence_array.mean()),
            "full_sequence_pearson": float(np.corrcoef(logits_array, sequence_array)[0, 1]),
            "residual_mean": float(residual.mean()),
            "residual_standard_deviation": float(residual.std()),
        }
    environment = {
        "python": sys.version,
        "accelerator_backend": accelerator_backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("biopython", "fair-esm", "numpy", "torch")
        },
        "test_evaluated": False,
    }
    (arguments.output_root / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )
    (arguments.output_root / "oracle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
