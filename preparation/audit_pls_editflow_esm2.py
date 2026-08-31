"""Audit a completed safe-sequence ESM-2 mean embedding cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(entities: Path, feature_dir: Path) -> dict:
    with entities.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    config = json.loads((feature_dir / "config.json").read_text())
    embeddings = np.load(feature_dir / "embeddings.npy", mmap_mode="r")
    status = np.load(feature_dir / "status.npy", mmap_mode="r")
    expected_shape = (len(rows), int(config["embedding_dimension"]))
    if embeddings.shape != expected_shape or status.shape != (len(rows),):
        raise ValueError("ESM-2 cache shape mismatch")
    if np.any(status != 1):
        raise ValueError("ESM-2 cache is incomplete")
    norms = []
    for start in range(0, len(rows), 128):
        batch = np.asarray(embeddings[start:start + 128], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise ValueError("ESM-2 cache contains a non-finite value")
        norms.extend(np.linalg.vector_norm(batch, axis=1).tolist())
    return {
        "schema": "PLS_EditFlow_ESM2_mean_audit_v1",
        "entities": len(rows),
        "shape": list(embeddings.shape),
        "complete": int(np.sum(status == 1)),
        "embedding_norm_mean": float(np.mean(norms)),
        "embedding_norm_minimum": float(np.min(norms)),
        "embedding_norm_maximum": float(np.max(norms)),
        "embeddings_sha256": file_sha256(feature_dir / "embeddings.npy"),
        "status_sha256": file_sha256(feature_dir / "status.npy"),
        "config_sha256": file_sha256(feature_dir / "config.json"),
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = audit(arguments.entities, arguments.feature_dir)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
