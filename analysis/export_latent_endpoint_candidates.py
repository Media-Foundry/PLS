"""Export task-specific validation candidates from one latent-endpoint run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = (("pdbsol", 0, "logits"), ("uesolds", 1, "logits"), ("esol", 2, "predictions"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pdb-entity-reference", type=Path)
    args = parser.parse_args()
    values = np.load(args.run / "validation_predictions.npz")
    report = {"selection_data": "strict-validation only", "test_evaluated": False, "run": str(args.run), "tasks": {}}
    for task, endpoint, prediction_key in TASKS:
        selected = values["endpoints"] == endpoint
        selected_indices = np.flatnonzero(selected)
        if task == "pdbsol" and args.pdb_entity_reference:
            reference = np.load(args.pdb_entity_reference / "validation_predictions.npz")["entity_indices"]
            available = {int(entity): int(index) for index, entity in zip(selected_indices, values["entity_indices"][selected])}
            if any(int(entity) not in available for entity in reference): raise ValueError("latent PDB candidate does not cover the frozen entity reference")
            selected_indices = np.asarray([available[int(entity)] for entity in reference], dtype=np.int64)
        destination = args.run / "candidates" / task
        destination.mkdir(parents=True, exist_ok=True)
        payload = {"targets": values["targets"][selected_indices], prediction_key: values["predictions"][selected_indices], "entity_indices": values["entity_indices"][selected_indices]}
        np.savez_compressed(destination / "validation_predictions.npz", **payload)
        report["tasks"][task] = {"endpoint": endpoint, "entities": int(len(selected_indices)), "path": str(destination)}
    (args.run / "candidate_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
