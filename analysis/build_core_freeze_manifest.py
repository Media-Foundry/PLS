"""Build an immutable, test-free manifest for the frozen core PLS recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from strings(nested)


def selected_artifacts(selection: dict):
    reports: set[Path] = set()
    runs: set[Path] = set()
    pending = []
    for value in strings(selection):
        path = Path(value)
        if path.is_dir() and str(path).startswith("outputs/"):
            runs.add(path)
        elif path.is_file() and path.suffix == ".json" and str(path).startswith("outputs/"):
            pending.append(path)
    while pending:
        report = pending.pop()
        if report in reports:
            continue
        reports.add(report)
        payload = json.loads(report.read_text())
        if payload.get("test_evaluated") is True:
            raise ValueError(f"report records forbidden test evaluation: {report}")
        for value in strings(payload):
            path = Path(value)
            if path.is_dir() and str(path).startswith("outputs/"):
                runs.add(path)
            elif (
                path.is_file()
                and path.suffix == ".json"
                and str(path).startswith("outputs/validation_selection/")
            ):
                pending.append(path)
    return sorted(reports), sorted(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", type=Path, default=Path("configs/validation_selection_v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if selection.get("test_evaluated") is not False:
        raise ValueError("selection config lacks the permanent test freeze")
    reports, runs = selected_artifacts(selection)

    report_records = []
    for path in reports:
        payload = json.loads(path.read_text())
        report_records.append({
            "path": str(path),
            "sha256": sha256(path),
            "test_freeze_assertion": (
                "explicit_false"
                if payload.get("test_evaluated") is False
                else "inherited_from_frozen_selection"
            ),
        })
    run_records = []
    for run in runs:
        config_path = run / "config.json"
        checkpoint_candidates = (
            run / "checkpoints" / "best.pt",
            run / "checkpoints" / "best.pkl",
        )
        checkpoint = next((path for path in checkpoint_candidates if path.is_file()), None)
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        if config.get("evaluate_test", False):
            raise ValueError(f"selected run permits test evaluation: {run}")
        run_records.append({
            "path": str(run),
            "config_sha256": sha256(config_path) if config_path.is_file() else None,
            "seed": config.get("training", {}).get("seed"),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        })

    feature_files = [
        Path("artifacts/plm/esm2_t33_650M_UR50D_mean/config.json"),
        Path("artifacts/plm/esm2_t33_650M_UR50D_mean/embeddings.npy"),
        Path("artifacts/features/pdbsol_structure_v4_train_stats.json"),
        Path("artifacts/features/pdbsol_structure_v4_compact/metadata.json"),
        Path("artifacts/features/pdbsol_structure_v4_geometry/metadata.json"),
        Path("artifacts/plm/esm2_t33_650M_UR50D_residue_pdbsol/pca_metadata.json"),
        Path("artifacts/features/pdbsol_structure_v4_vectors/metadata.json"),
        Path("artifacts/features/pdbsol_surface_patch_components_v1/metadata.json"),
        Path("artifacts/features/sequence_descriptors_v1/metadata.json"),
    ]
    if not all(path.is_file() for path in feature_files):
        missing = [str(path) for path in feature_files if not path.is_file()]
        raise FileNotFoundError(f"missing frozen feature artifacts: {missing}")
    split_report = json.loads(Path("benchmark/strict_si30_split_report.json").read_text())
    fold_report = json.loads(
        Path("benchmark/pls_editflow_esmfold_v1_checkpoint_report.json").read_text()
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": "PLS_core_frozen_pretest_state_v1",
        "date": "2026-09-01",
        "frozen_core_code_revision": revision,
        "selection_config": {
            "path": str(args.selection),
            "sha256": sha256(args.selection),
            "recipes": selection["tasks"],
        },
        "selected_reports": report_records,
        "selected_runs": run_records,
        "feature_artifacts": [
            {"path": str(path), "sha256": sha256(path)} for path in feature_files
        ],
        "split_seal": {
            "entity_split_sha256": split_report["entity_split_sha256"],
            "observation_split_sha256": split_report["observation_split_sha256"],
            "note": "Full immutable split hashes transitively seal test membership; no test-only rows were read to build this manifest."
        },
        "esmfold": {
            "checkpoint": fold_report["checkpoint"],
            "sha256": fold_report["sha256"],
            "fair_esm_version": fold_report["fair_esm_version"],
        },
        "policy": {
            "core_predictor_development": "frozen",
            "new_ideas_destination": "PLS-v2 or methods branches only",
            "test_evaluation": "permanently prohibited",
            "test_evaluated": False,
        },
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "reports": len(report_records),
        "runs": len(run_records),
        "checkpoints_hashed": sum(row["checkpoint"] is not None for row in run_records),
        "test_evaluated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
