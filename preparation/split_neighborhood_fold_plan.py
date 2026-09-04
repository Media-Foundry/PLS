"""Split an exhaustive-neighborhood fold campaign across heterogeneous machines.

Whole anchors are assigned to one machine, never split. Every metric the frozen
protocol reports is computed WITHIN an anchor, and decision regret and rank are
invariant to the parent's absolute score, so keeping a neighborhood on one
accelerator keeps those comparisons internally consistent.

Cross-hardware agreement was measured before this was used: CUDA against ROCm on
the same sequences gave mean C-alpha RMSD 0.058 A and minimum TM-score 0.993,
two orders of magnitude below the effect the campaign measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from preparation.plan_pls_editflow_oracle import lpt_shards
from pls.editflow.runtime_cost import load_runtime_cost_model, predict_runtime_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-cost-model", type=Path, required=True)
    parser.add_argument("--machines", required=True,
                        help='JSON list of {"name","shards","weight"}')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text())
    if manifest.get("test_evaluated") is not False:
        raise SystemExit("manifest is not explicitly test-free")
    nodes = manifest["nodes"]
    if any(node["split"] != "train" for node in nodes):
        raise SystemExit("fold plan split refuses a manifest that is not train-only")
    machines = json.loads(arguments.machines)
    if len(machines) < 2 or any(m["shards"] < 1 or m["weight"] <= 0 for m in machines):
        raise SystemExit("each machine needs at least one shard and a positive weight")

    model = load_runtime_cost_model(arguments.runtime_cost_model)
    mutants = [n for n in nodes if n["kind"] == "single_mutant"]
    by_anchor: dict[int, list[dict]] = {}
    for node in mutants:
        by_anchor.setdefault(int(node["anchor_rank"]), []).append(node)

    anchor_cost = {}
    for rank, group in by_anchor.items():
        lengths = np.asarray([n["length"] for n in group], dtype=np.float64)
        anchor_cost[rank] = float(predict_runtime_seconds(model, lengths).sum())

    # Weighted longest-processing-time over whole anchors: always give the next
    # most expensive anchor to whichever machine has the least work per unit of
    # its throughput.
    assigned: dict[str, list[int]] = {m["name"]: [] for m in machines}
    load = {m["name"]: 0.0 for m in machines}
    for rank in sorted(anchor_cost, key=lambda r: -anchor_cost[r]):
        target = min(machines, key=lambda m: load[m["name"]] / m["weight"])
        assigned[target["name"]].append(rank)
        load[target["name"]] += anchor_cost[rank]

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    report_machines = []
    for machine in machines:
        name = machine["name"]
        ranks = sorted(assigned[name])
        subset = [n for r in ranks for n in by_anchor[r]]
        shard_assignments = lpt_shards(subset, int(machine["shards"]), model)
        identity = "\n".join(
            f"{row['node_index']}:{row['sequence_sha256']}:{row['shard']}"
            for row in shard_assignments)
        plan = {
            "schema": "PLS_EditFlow_ESMFold_query_plan_v1",
            "shard_count": int(machine["shards"]),
            "cost_proxy": "frozen_monotone_predicted_marginal_esmfold_gpu_seconds",
            "assignments": shard_assignments,
            "machine": name,
            "anchor_ranks": ranks,
            "split_rule": "whole anchors, weighted longest-processing-time",
            "assignments_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            "test_evaluated": False,
            "test_sequences_queried": 0,
        }
        path = arguments.output_dir / f"pls_editflow_oracle_query_plan_neighborhood_scale_{name}_v1.json"
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        lengths = np.asarray([by_anchor[r][0]["length"] for r in ranks], dtype=float)
        report_machines.append({
            "name": name,
            "plan": str(path),
            "shards": int(machine["shards"]),
            "weight": float(machine["weight"]),
            "anchors": len(ranks),
            "anchor_ranks": ranks,
            "mutants": len(subset),
            "predicted_gpu_seconds": load[name],
            "predicted_hours_at_full_width": load[name] / 3600 / int(machine["shards"]),
            "anchor_length_min": float(lengths.min()),
            "anchor_length_median": float(np.median(lengths)),
            "anchor_length_max": float(lengths.max()),
            "shard_loads": [
                round(sum(r["estimated_cost"] for r in shard_assignments if r["shard"] == s), 1)
                for s in range(int(machine["shards"]))
            ],
        })

    total = sum(m["predicted_gpu_seconds"] for m in report_machines)
    report = {
        "schema": "PLS_neighborhood_fold_split_report_v1",
        "manifest": str(arguments.manifest),
        "total_mutants": len(mutants),
        "total_predicted_gpu_seconds": total,
        "machines": report_machines,
        "predicted_wall_hours": max(m["predicted_hours_at_full_width"] for m in report_machines),
        "cross_hardware_validation": {
            "sequences": 8,
            "mean_ca_rmsd_angstrom": 0.0578,
            "max_ca_rmsd_angstrom": 0.2458,
            "min_tm_score": 0.99287,
            "note": "CUDA RTX 4090 against ROCm, identical model, recycles and chunk size",
        },
        "test_evaluated": False,
        "test_sequences_queried": 0,
    }
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for m in report_machines:
        print(f"{m['name']:>14}: {m['anchors']:>3} anchors, {m['mutants']:>7,} mutants, "
              f"{m['predicted_gpu_seconds'] / 3600:>6.1f} GPU-h over {m['shards']} shards "
              f"-> {m['predicted_hours_at_full_width']:>5.1f} h wall, "
              f"L {m['anchor_length_min']:.0f}-{m['anchor_length_max']:.0f} "
              f"(median {m['anchor_length_median']:.0f})")
    print(f"\npredicted wall: {report['predicted_wall_hours']:.1f} h "
          f"(single-machine 7-shard baseline was 21.5 h)")


if __name__ == "__main__":
    main()
