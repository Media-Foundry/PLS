"""Verify a fold campaign that ran across two machines.

The standard postfold check reads one plan and expects shard_000..N-1. This
campaign has two plans and two report families, so completeness is checked
against the manifest itself and against both machines' reports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("artifacts/oracles/pls_editflow_neighborhood_scale_v1")
ESMFOLD = ROOT / "exact_full" / "esmfold"
MANIFEST = Path("benchmark/generated/pls_editflow_neighborhood_scale_v1.json")
SPLIT = Path("benchmark/generated/pls_editflow_neighborhood_scale_split_report_v1.json")


def load(prefix: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(ESMFOLD.glob(f"{prefix}_*_report.json"))]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("test_evaluated") is not False:
        sys.exit("manifest is not explicitly test-free")
    mutants = [n for n in manifest["nodes"] if n["kind"] == "single_mutant"]
    anchors = [n for n in manifest["nodes"] if n["kind"] == "anchor"]

    families = {"diamondhill": load("shard"), "dd": load("dd_shard")}
    summary = {}
    for name, reports in families.items():
        if not reports:
            sys.exit(f"no shard reports for {name}")
        seconds = sum(r["seconds"] for rep in reports for r in rep["results"] if r["status"] == "ok")
        summary[name] = {
            "shards": len(reports),
            "assigned": sum(r["assigned"] for r in reports),
            "ok": sum(r["ok"] for r in reports),
            "skipped": sum(r["skipped"] for r in reports),
            "failed": sum(r["failed"] for r in reports),
            "measured_gpu_seconds": round(seconds, 1),
            "backends": sorted({r["accelerator_backend"] for r in reports}),
            "devices": sorted({r["physical_device"] for r in reports}),
            "num_recycles": sorted({r["num_recycles"] for r in reports}),
            "chunk_size": sorted({r["chunk_size"] for r in reports}),
        }
        if summary[name]["failed"]:
            sys.exit(f"{name} reported {summary[name]['failed']} failed folds")

    total_assigned = sum(s["assigned"] for s in summary.values())
    if total_assigned != len(mutants):
        sys.exit(f"assigned {total_assigned:,} != {len(mutants):,} mutants in the manifest")

    # Every mutant must have a structure on disk; anchors reuse the cached parent.
    missing = [n["sequence_sha256"] for n in mutants
               if not (ESMFOLD / f"{n['sequence_sha256']}.ef.pdb").is_file()]
    if missing:
        sys.exit(f"{len(missing):,} mutant structures are missing, e.g. {missing[:3]}")

    recycles = {v for s in summary.values() for v in s["num_recycles"]}
    chunks = {v for s in summary.values() for v in s["chunk_size"]}
    if recycles != {3} or chunks != {64}:
        sys.exit(f"inconsistent oracle settings: recycles={recycles} chunk={chunks}")

    split = json.loads(SPLIT.read_text())
    provenance = {m["name"]: m["anchor_ranks"] for m in split["machines"]}
    covered = sorted(r for ranks in provenance.values() for r in ranks)
    if covered != sorted(int(a["anchor_rank"]) for a in anchors):
        sys.exit("the split report does not cover exactly the manifest's anchors")

    out = {
        "schema": "PLS_neighborhood_scale_fold_verification_v1",
        "mutants": len(mutants),
        "anchors": len(anchors),
        "structures_on_disk": len(mutants) - len(missing),
        "by_machine": summary,
        "total_measured_gpu_seconds": round(
            sum(s["measured_gpu_seconds"] for s in summary.values()), 1),
        "anchor_provenance": provenance,
        "oracle_settings": {"num_recycles": 3, "chunk_size": 64},
        "cross_hardware_validation": split["cross_hardware_validation"],
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    Path("analysis/neighborhood_scale_fold_verification_v1.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "anchor_provenance"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
