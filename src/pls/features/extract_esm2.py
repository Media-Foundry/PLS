"""Resumable frozen ESM-2 mean-embedding extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import esm
import numpy as np
import torch


MODEL_NAME = "esm2_t33_650M_UR50D"
REPRESENTATION_LAYER = 33
EMBEDDING_DIMENSION = 1280


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunks(sequence: str, maximum_residues: int):
    for start in range(0, len(sequence), maximum_residues):
        yield sequence[start:start + maximum_residues]


def load_entities(path: Path, limit: int | None):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--maximum-residues", type=int, default=1022)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.token_budget < args.maximum_residues + 2:
        parser.error("token budget must hold at least one maximum-length chunk")
    if args.device.startswith("cuda") and os.environ.get("HIP_VISIBLE_DEVICES") != "7":
        parser.error("GPU extraction is pinned to physical device 7; set HIP_VISIBLE_DEVICES=7")

    rows = load_entities(args.entities, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1, "model": MODEL_NAME, "representation_layer": REPRESENTATION_LAYER,
        "embedding_dimension": EMBEDDING_DIMENSION, "entity_manifest_sha256": sha256(args.entities),
        "entity_count": len(rows), "maximum_residues": args.maximum_residues,
        "pooling": "exact_residue_mean_over_nonoverlapping_chunks", "precision": args.precision,
        "torch_version": torch.__version__, "torch_hip_version": torch.version.hip,
        "esm_module": str(Path(esm.__file__).resolve()),
    }
    config_path = args.output_dir / "config.json"
    existing_config = json.loads(config_path.read_text()) if config_path.exists() else None
    if existing_config and any(existing_config.get(key) != value for key, value in config.items()):
        raise ValueError("existing feature config differs; use a new output directory")

    embedding_path = args.output_dir / "embeddings.npy"
    status_path = args.output_dir / "status.npy"
    if embedding_path.exists():
        embeddings = np.load(embedding_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        if embeddings.shape != (len(rows), EMBEDDING_DIMENSION) or status.shape != (len(rows),):
            raise ValueError("existing feature array shape differs")
    else:
        embeddings = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32,
                                               shape=(len(rows), EMBEDDING_DIMENSION))
        status = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.uint8, shape=(len(rows),))
        status[:] = 0; status.flush()

    print(f"loading {MODEL_NAME}", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    config["model_checkpoint_sha256"] = sha256(checkpoint_dir / f"{MODEL_NAME}.pt")
    regression_path = checkpoint_dir / f"{MODEL_NAME}-contact-regression.pt"
    config["contact_regression_sha256"] = sha256(regression_path) if regression_path.exists() else None
    if existing_config:
        for key in ("model_checkpoint_sha256", "contact_regression_sha256"):
            if key in existing_config and existing_config[key] != config[key]:
                raise ValueError(f"cached checkpoint differs from existing feature config: {key}")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    model.eval().requires_grad_(False).to(args.device)
    if args.precision == "float16":
        model.half()
    batch_converter = alphabet.get_batch_converter()

    pending = [index for index in range(len(rows)) if status[index] != 1]
    pending.sort(key=lambda index: (len(rows[index]["sequence"]), rows[index]["sequence_sha256"]))
    completed = int(np.sum(status == 1))
    cursor = 0
    while cursor < len(pending):
        batch_indices, batch_chunks, padded_tokens = [], [], 0
        while cursor < len(pending):
            index = pending[cursor]
            protein_chunks = list(chunks(rows[index]["sequence"], args.maximum_residues))
            candidate_max = max([len(item) for _, item in batch_chunks] + [len(item) for item in protein_chunks]) + 2
            candidate_count = len(batch_chunks) + len(protein_chunks)
            if batch_indices and candidate_max * candidate_count > args.token_budget:
                break
            batch_indices.append(index)
            batch_chunks.extend((index, item) for item in protein_chunks)
            padded_tokens = candidate_max * candidate_count
            cursor += 1
        labels = [(f"{index}:{part}", sequence) for part, (index, sequence) in enumerate(batch_chunks)]
        _, _, tokens = batch_converter(labels)
        tokens = tokens.to(args.device, non_blocking=True)
        with torch.inference_mode():
            result = model(tokens, repr_layers=[REPRESENTATION_LAYER], return_contacts=False)
        representations = result["representations"][REPRESENTATION_LAYER]
        sums = {index: torch.zeros(EMBEDDING_DIMENSION, dtype=torch.float32) for index in batch_indices}
        counts = {index: 0 for index in batch_indices}
        for position, (index, sequence) in enumerate(batch_chunks):
            sums[index] += representations[position, 1:len(sequence) + 1].float().sum(dim=0).cpu()
            counts[index] += len(sequence)
        for index in batch_indices:
            embeddings[index] = (sums[index] / counts[index]).numpy()
            status[index] = 1
        embeddings.flush(); status.flush()
        completed += len(batch_indices)
        print(f"completed={completed:,}/{len(rows):,} proteins padded_tokens={padded_tokens:,}", flush=True)

    print("feature extraction complete", flush=True)


if __name__ == "__main__":
    main()
