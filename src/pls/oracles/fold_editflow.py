"""Resumable ESMFold execution for strict PLS EditFlow mutant shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


ESMFOLD_POINT_PROJECTION_REMAP = {
    "trunk.structure_module.ipa.linear_kv_points.weight":
        "trunk.structure_module.ipa.linear_kv_points.linear.weight",
    "trunk.structure_module.ipa.linear_kv_points.bias":
        "trunk.structure_module.ipa.linear_kv_points.linear.bias",
    "trunk.structure_module.ipa.linear_q_points.weight":
        "trunk.structure_module.ipa.linear_q_points.linear.weight",
    "trunk.structure_module.ipa.linear_q_points.bias":
        "trunk.structure_module.ipa.linear_q_points.linear.bias",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remap_esmfold_point_projection_keys(
    model_state: dict,
    expected_keys: set[str],
) -> tuple[dict, dict[str, str]]:
    """Adapt the four IPA point projections used by wrapped OpenFold Linear.

    Some OpenFold wheels wrap these projections and expose ``.linear.weight``
    while the official fair-esm v1 checkpoint stores ``.weight``.  Only exact
    source/target pairs are remapped; collisions and partial pairs hard-fail.
    """
    state = dict(model_state)
    applied: dict[str, str] = {}
    for source, target in ESMFOLD_POINT_PROJECTION_REMAP.items():
        source_present = source in state
        target_expected = target in expected_keys
        if source_present and target_expected:
            if target in state:
                raise ValueError(f"ESMFold remap target already exists: {target}")
            state[target] = state.pop(source)
            applied[source] = target
        elif source_present != target_expected:
            raise ValueError(
                f"incomplete ESMFold/OpenFold compatibility pair: {source} -> {target}"
            )
    return state, applied


def load_esmfold_v1_compatible(torch_module):
    """Load the official v1 checkpoint with an audited four-key compatibility map."""
    from esm.esmfold.v1.esmfold import ESMFold

    checkpoint = Path(torch_module.hub.get_dir()) / "checkpoints" / "esmfold_3B_v1.pt"
    model_data = torch_module.load(checkpoint, map_location="cpu", weights_only=False)
    model = ESMFold(esmfold_config=model_data["cfg"]["model"])
    expected = set(model.state_dict())
    state, remapped = remap_esmfold_point_projection_keys(model_data["model"], expected)
    missing = sorted(
        key for key in expected - set(state) if not key.startswith("esm.")
    )
    if missing:
        raise RuntimeError(f"essential ESMFold keys are missing after compatibility remap: {missing}")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"unexpected ESMFold checkpoint keys after remap: {unexpected}")
    return model, remapped


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


def validate_visible_device(
    hip_device: int | None,
    cuda_device: int | None,
    environment: dict[str, str],
) -> tuple[str, int]:
    """Validate one explicitly masked physical accelerator.

    Local ROCm execution is restricted to the explicitly authorized physical
    devices 0--3 or 6/7.  The
    separate CUDA path exists for the two-device star host and accepts only
    physical devices 0/1.  Both expose the selected device as logical cuda:0
    to ESMFold.
    """
    if (hip_device is None) == (cuda_device is None):
        raise ValueError("select exactly one of hip_device or cuda_device")
    if hip_device is not None:
        if hip_device not in {0, 1, 2, 3, 6, 7}:
            raise ValueError("ROCm execution is restricted to authorized physical devices")
        if environment.get("HIP_VISIBLE_DEVICES") != str(hip_device):
            raise ValueError("HIP device mismatch")
        return "rocm", hip_device
    assert cuda_device is not None
    if cuda_device not in {0, 1}:
        raise ValueError("star CUDA execution is restricted to physical devices 0/1")
    if environment.get("CUDA_VISIBLE_DEVICES") != str(cuda_device):
        raise ValueError("CUDA device mismatch")
    return "cuda", cuda_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--hip-device", type=int, choices=(0, 1, 2, 3, 6, 7))
    device.add_argument("--cuda-device", type=int, choices=(0, 1))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    records, plan = load_shard(arguments.manifest, arguments.plan, arguments.shard_index)
    status = shard_status(records, arguments.output_root)
    if arguments.dry_run:
        print(json.dumps({"mode": "dry_run", "shard": arguments.shard_index, **status}, sort_keys=True))
        return
    try:
        accelerator_backend, physical_device = validate_visible_device(
            arguments.hip_device, arguments.cuda_device, dict(os.environ)
        )
    except ValueError as error:
        parser.error(str(error))
    if arguments.chunk_size < 1 or arguments.num_recycles < 0:
        parser.error("invalid ESMFold inference settings")

    import esm
    import torch

    arguments.output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(json.dumps({"event": "loading_esmfold_v1", "shard": arguments.shard_index}), flush=True)
    model, checkpoint_key_remaps = load_esmfold_v1_compatible(torch)
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
        "accelerator_backend": accelerator_backend,
        "physical_device": physical_device,
        "chunk_size": arguments.chunk_size,
        "num_recycles": arguments.num_recycles,
        "assigned": len(records),
        "ok": sum(row["status"] == "ok" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_files": checkpoint_records,
        "checkpoint_key_remaps": checkpoint_key_remaps,
        "results": results,
        "test_evaluated": False,
    }
    report_path = arguments.output_root / f"shard_{arguments.shard_index:03d}_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
