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


def shard_mutant_indices(plan_path: Path, shard_index: int) -> set[int]:
    plan = json.loads(plan_path.read_text())
    if plan.get("test_evaluated") is not False:
        raise ValueError("feature sharding refuses a plan without a test-free assertion")
    if not 0 <= shard_index < int(plan["shard_count"]):
        raise ValueError("shard index is outside the query plan")
    return {
        int(row["node_index"])
        for row in plan["assignments"]
        if int(row["shard"]) == shard_index
    }


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
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if (arguments.plan is None) != (arguments.shard_index is None):
        parser.error("--plan and --shard-index must be provided together")
    records = load_safe_records(arguments.manifest)
    selected_indices = (
        shard_mutant_indices(arguments.plan, arguments.shard_index)
        if arguments.plan is not None
        else None
    )
    if selected_indices is not None:
        selected_records = [row for row in records if row["node_index"] in selected_indices]
        if len(selected_records) != len(selected_indices) or any(
            row["kind"] != "single_mutant" for row in selected_records
        ):
            raise ValueError("feature shard does not map exactly to safe mutant records")
    else:
        selected_records = records
    status_before = extraction_status(
        records,
        arguments.pdb_root,
        arguments.parent_raw_root,
        arguments.output_root,
    )
    if arguments.dry_run:
        print(json.dumps({
            "mode": "dry_run",
            "shard_index": arguments.shard_index,
            "selected_records": len(selected_records),
            **status_before,
        }, indent=2, sort_keys=True))
        return
    if arguments.workers < 1:
        parser.error("workers must be positive")
    anchors = (
        [row for row in records if row["kind"] == "anchor"]
        if selected_indices is None
        else []
    )
    mutants = [row for row in selected_records if row["kind"] == "single_mutant"]
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
    status = np.fromiter(
        (
            raw_path(arguments.output_root, row["sequence_sha256"]).is_file()
            for row in records
        ),
        dtype=np.uint8,
        count=len(records),
    )
    if selected_indices is None:
        np.save(arguments.output_root / "status.npy", status)
    report = {
        "schema": "PLS_EditFlow_structure_v4_extraction_v1",
        "shard_index": arguments.shard_index,
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
    summary_name = (
        "extraction_summary.json"
        if arguments.shard_index is None
        else f"shard_{arguments.shard_index:03d}_extraction_summary.json"
    )
    manifest_name = (
        "extraction_manifest.jsonl"
        if arguments.shard_index is None
        else f"shard_{arguments.shard_index:03d}_extraction_manifest.jsonl"
    )
    (arguments.output_root / summary_name).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (arguments.output_root / manifest_name).open("w", encoding="utf-8") as handle:
        for row in sorted(results, key=lambda item: item["sequence_sha256"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    expected_complete = len(records) if selected_indices is None else None
    if report["mutant_failed"] or (
        expected_complete is not None and report["complete_records"] != expected_complete
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
