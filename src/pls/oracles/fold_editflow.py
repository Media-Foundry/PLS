"""Resumable ESMFold execution for strict PLS EditFlow mutant shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shard(manifest_path: Path, plan_path: Path, shard_index: int) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    if manifest.get("test_evaluated") is not False or plan.get("test_evaluated") is not False:
        raise ValueError("folding refuses inputs without a test-free assertion")
    if not 0 <= shard_index < int(plan["shard_count"]):
        raise ValueError("shard index is outside the query plan")
    nodes = {int(row["node_index"]): row for row in manifest["nodes"]}
    records = []
    for assignment in plan["assignments"]:
        if int(assignment["shard"]) != shard_index:
            continue
        node = nodes[int(assignment["node_index"])]
        if node["kind"] != "single_mutant":
            raise ValueError("ESMFold plan may contain only new mutant queries")
        if node["split"] not in {"train", "validation"}:
            raise ValueError("ESMFold plan contains a forbidden split")
        if node["sequence_sha256"] != assignment["sequence_sha256"]:
            raise ValueError("ESMFold plan sequence identity mismatch")
        if len(node["sequence"]) != int(assignment["length"]):
            raise ValueError("ESMFold plan sequence length mismatch")
        records.append(node)
    records.sort(key=lambda row: (int(row["length"]), row["sequence_sha256"]))
    return records, plan


def output_path(root: Path, digest: str) -> Path:
    return root / f"{digest}.ef.pdb"


def shard_status(records: list[dict], output_root: Path) -> dict:
    existing = sum(output_path(output_root, row["sequence_sha256"]).is_file() for row in records)
    return {
        "assigned": len(records),
        "existing": existing,
        "pending": len(records) - existing,
        "assigned_residues": sum(int(row["length"]) for row in records),
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--hip-device", type=int, choices=(6, 7), required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    records, plan = load_shard(arguments.manifest, arguments.plan, arguments.shard_index)
    status = shard_status(records, arguments.output_root)
    if arguments.dry_run:
        print(json.dumps({"mode": "dry_run", "shard": arguments.shard_index, **status}, sort_keys=True))
        return
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(arguments.hip_device):
        parser.error("HIP device mismatch")
    if arguments.chunk_size < 1 or arguments.num_recycles < 0:
        parser.error("invalid ESMFold inference settings")

    import esm
    import torch

    arguments.output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(json.dumps({"event": "loading_esmfold_v1", "shard": arguments.shard_index}), flush=True)
    model = esm.pretrained.esmfold_v1()
    model.eval().requires_grad_(False).to("cuda:0")
    model.set_chunk_size(arguments.chunk_size)
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_records = [
        {"name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(checkpoint_dir.glob("*esmfold*"))
        if path.is_file()
    ]
    results = []
    for index, record in enumerate(records, 1):
        digest = record["sequence_sha256"]
        destination = output_path(arguments.output_root, digest)
        if destination.is_file():
            results.append({"sequence_sha256": digest, "status": "skipped", "seconds": 0.0})
            continue
        query_started = time.monotonic()
        try:
            with torch.inference_mode():
                pdb = model.infer_pdb(
                    record["sequence"], num_recycles=arguments.num_recycles
                )
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=arguments.output_root,
                prefix=f".{digest}.",
                delete=False,
            ) as handle:
                handle.write(pdb)
                temporary = Path(handle.name)
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            result = {
                "sequence_sha256": digest,
                "status": "ok",
                "length": int(record["length"]),
                "seconds": time.monotonic() - query_started,
                "pdb_bytes": destination.stat().st_size,
            }
        except Exception as error:
            result = {
                "sequence_sha256": digest,
                "status": "failed",
                "length": int(record["length"]),
                "seconds": time.monotonic() - query_started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        results.append(result)
        print(
            json.dumps({
                "shard": arguments.shard_index,
                "completed": index,
                "assigned": len(records),
                "status": result["status"],
                "length": int(record["length"]),
            }),
            flush=True,
        )
    report = {
        "schema": "PLS_EditFlow_ESMFold_shard_report_v1",
        "shard_index": arguments.shard_index,
        "shard_count": int(plan["shard_count"]),
        "hip_device": arguments.hip_device,
        "chunk_size": arguments.chunk_size,
        "num_recycles": arguments.num_recycles,
        "assigned": len(records),
        "ok": sum(row["status"] == "ok" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_files": checkpoint_records,
        "results": results,
        "test_evaluated": False,
    }
    report_path = arguments.output_root / f"shard_{arguments.shard_index:03d}_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
