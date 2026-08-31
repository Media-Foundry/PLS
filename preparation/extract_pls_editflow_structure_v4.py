"""Materialize exact PLS V4 features for safe EditFlow anchors and mutants."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from preparation.extract_pdbsol_structure_v4 import (extract_one, init_worker)
from pls.features.external_v4_loader import load_external_v4


def load_safe_records(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("test_evaluated") is not False:
        raise ValueError("feature extraction refuses manifests without a test-free assertion")
    records = []
    for node in manifest["nodes"]:
        if node["split"] not in {"train", "validation"}:
            raise ValueError("feature manifest contains a forbidden split")
        records.append({
            "node_index": int(node["node_index"]),
            "kind": node["kind"],
            "name": node["sequence_sha256"],
            "sequence": node["sequence"],
            "sequence_sha256": node["sequence_sha256"],
            "upstream_split": node["split"],
        })
    if [row["node_index"] for row in records] != list(range(len(records))):
        raise ValueError("feature manifest node order is not canonical")
    return records


def raw_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / f"{digest}.pt"


def extraction_status(
    records: list[dict],
    pdb_root: Path,
    parent_raw_root: Path,
    output_root: Path,
) -> dict:
    anchors = [row for row in records if row["kind"] == "anchor"]
    mutants = [row for row in records if row["kind"] == "single_mutant"]
    return {
        "records": len(records),
        "anchors": len(anchors),
        "mutants": len(mutants),
        "parent_raw_available": sum(
            raw_path(parent_raw_root, row["sequence_sha256"]).is_file()
            for row in anchors
        ),
        "mutant_pdb_available": sum(
            (pdb_root / f"{row['sequence_sha256']}.ef.pdb").is_file()
            for row in mutants
        ),
        "output_raw_available": sum(
            raw_path(output_root, row["sequence_sha256"]).is_file()
            for row in records
        ),
        "test_evaluated": False,
    }


def materialize_parent(source: Path, destination: Path) -> str:
    if destination.is_file():
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        temporary = destination.with_name(f".{destination.name}.copying")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return "copied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pdb-root", type=Path, required=True)
    parser.add_argument("--parent-raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/home/pc/Code/BIO/protein"))
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    records = load_safe_records(arguments.manifest)
    status_before = extraction_status(
        records,
        arguments.pdb_root,
        arguments.parent_raw_root,
        arguments.output_root,
    )
    if arguments.dry_run:
        print(json.dumps({"mode": "dry_run", **status_before}, indent=2, sort_keys=True))
        return
    if arguments.workers < 1:
        parser.error("workers must be positive")
    anchors = [row for row in records if row["kind"] == "anchor"]
    mutants = [row for row in records if row["kind"] == "single_mutant"]
    missing_parents = [
        row["sequence_sha256"]
        for row in anchors
        if not raw_path(arguments.parent_raw_root, row["sequence_sha256"]).is_file()
    ]
    missing_pdbs = [
        row["sequence_sha256"]
        for row in mutants
        if not (arguments.pdb_root / f"{row['sequence_sha256']}.ef.pdb").is_file()
    ]
    if missing_parents or missing_pdbs:
        raise FileNotFoundError(
            f"oracle features are incomplete: missing_parents={len(missing_parents)}, "
            f"missing_mutant_pdbs={len(missing_pdbs)}"
        )
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    _, source_hashes = load_external_v4(arguments.source_root)
    materialized = {"hardlinked": 0, "copied": 0, "skipped": 0}
    for row in anchors:
        outcome = materialize_parent(
            raw_path(arguments.parent_raw_root, row["sequence_sha256"]),
            raw_path(arguments.output_root, row["sequence_sha256"]),
        )
        materialized[outcome] += 1

    tasks = [
        (row, str(arguments.pdb_root), str(arguments.output_root), False)
        for row in mutants
    ]
    results = []
    started = time.monotonic()
    with mp.get_context("spawn").Pool(
        arguments.workers,
        initializer=init_worker,
        initargs=(str(arguments.source_root),),
        maxtasksperchild=100,
    ) as pool:
        for index, result in enumerate(
            pool.imap_unordered(extract_one, tasks, chunksize=1), 1
        ):
            results.append(result)
            if index == 1 or index % 50 == 0 or index == len(tasks):
                print(json.dumps({
                    "completed_mutants": index,
                    "mutants": len(tasks),
                    "ok": sum(row["status"] == "ok" for row in results),
                    "skipped": sum(row["status"] == "skipped" for row in results),
                    "failed": sum(row["status"] == "failed" for row in results),
                }), flush=True)
    status = np.zeros(len(records), dtype=np.uint8)
    for row in records:
        status[row["node_index"]] = int(
            raw_path(arguments.output_root, row["sequence_sha256"]).is_file()
        )
    np.save(arguments.output_root / "status.npy", status)
    report = {
        "schema": "PLS_EditFlow_structure_v4_extraction_v1",
        "records": len(records),
        "anchors": len(anchors),
        "mutants": len(mutants),
        "parent_materialization": materialized,
        "mutant_ok": sum(row["status"] == "ok" for row in results),
        "mutant_skipped": sum(row["status"] == "skipped" for row in results),
        "mutant_failed": sum(row["status"] == "failed" for row in results),
        "complete_records": int(status.sum()),
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": source_hashes,
        "test_evaluated": False,
    }
    (arguments.output_root / "extraction_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (arguments.output_root / "extraction_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda item: item["sequence_sha256"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["mutant_failed"] or report["complete_records"] != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
