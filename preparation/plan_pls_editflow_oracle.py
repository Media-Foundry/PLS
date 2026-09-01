"""Create deterministic cost-balanced ESMFold shards for PLS mutants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def lpt_shards(nodes: list[dict], shard_count: int) -> list[dict]:
    """Longest-processing-time assignment using squared sequence length cost."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    loads = [0] * shard_count
    assignments = []
    ordered = sorted(
        nodes,
        key=lambda row: (-int(row["length"]) ** 2, row["sequence_sha256"]),
    )
    for node in ordered:
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        cost = int(node["length"]) ** 2
        assignments.append({
            "node_index": int(node["node_index"]),
            "sequence_sha256": node["sequence_sha256"],
            "length": int(node["length"]),
            "estimated_cost_l2": cost,
            "shard": shard,
        })
        loads[shard] += cost
    assignments.sort(key=lambda row: (row["shard"], row["node_index"]))
    return assignments


def build_plan(manifest: dict, shard_count: int) -> tuple[dict, dict]:
    if manifest.get("test_evaluated") is not False:
        raise ValueError("oracle plan refuses manifests without a test-free assertion")
    nodes = manifest["nodes"]
    if any(node["split"] not in {"train", "validation"} for node in nodes):
        raise ValueError("oracle plan contains a forbidden split")
    mutants = [node for node in nodes if node["kind"] == "single_mutant"]
    assignments = lpt_shards(mutants, shard_count)
    loads = [
        sum(row["estimated_cost_l2"] for row in assignments if row["shard"] == shard)
        for shard in range(shard_count)
    ]
    counts = [
        sum(row["shard"] == shard for row in assignments)
        for shard in range(shard_count)
    ]
    residues = [
        sum(row["length"] for row in assignments if row["shard"] == shard)
        for shard in range(shard_count)
    ]
    identity = "\n".join(
        f"{row['node_index']}:{row['sequence_sha256']}:{row['shard']}"
        for row in assignments
    )
    plan = {
        "schema": "PLS_EditFlow_ESMFold_query_plan_v1",
        "shard_count": shard_count,
        "cost_proxy": "sequence_length_squared",
        "assignments": assignments,
        "assignments_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "test_evaluated": False,
    }
    report = {
        "schema": "PLS_EditFlow_ESMFold_query_plan_report_v2",
        # A query plan assigns every mutant that requires an exact-structure
        # lookup.  Whether that lookup is already cached is only known when
        # the plan is executed, so do not mislabel this total as new work.
        "exact_fold_queries_total": len(assignments),
        "exact_fold_cached": None,
        "exact_fold_new_required": None,
        "cache_accounting_status": "not_observed_at_planning",
        "parent_queries_reusing_exact_sequence_cache": len(nodes) - len(assignments),
        "shard_count": shard_count,
        "queries_by_shard": counts,
        "residues_by_shard": residues,
        "estimated_l2_cost_by_shard": loads,
        "maximum_to_minimum_cost_ratio": max(loads) / min(loads),
        "assignments_sha256": plan["assignments_sha256"],
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    return plan, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text())
    plan, report = build_plan(manifest, arguments.shards)
    arguments.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
