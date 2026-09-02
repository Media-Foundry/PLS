"""Select the deployment exact-fold set before any confirmatory exact query."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from pls.editflow.cost_aware_gating import load_landscapes, select_candidates
from pls.editflow.runtime_cost import load_runtime_cost_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_manifest(source: dict, selections: list[dict]) -> tuple[dict, dict]:
    selected_targets = {
        int(index)
        for row in selections
        for index in row["selected_target_node_indices"]
    }
    selected_edges = [
        edge for edge in source["edges"] if int(edge["target_node"]) in selected_targets
    ]
    selected_anchors = {int(edge["source_node"]) for edge in selected_edges}
    old_indices = sorted(selected_anchors | selected_targets)
    old_to_new = {old: new for new, old in enumerate(old_indices)}
    nodes = []
    for old in old_indices:
        row = copy.deepcopy(source["nodes"][old])
        row["node_index"] = old_to_new[old]
        nodes.append(row)
    edges = []
    for edge in selected_edges:
        edges.append({
            **copy.deepcopy(edge),
            "edge_index": len(edges),
            "source_node": old_to_new[int(edge["source_node"])],
            "target_node": old_to_new[int(edge["target_node"])],
        })
    manifest = {
        "schema": "PLS_cost_aware_selected_exact_manifest_v2",
        "selection": {
            **copy.deepcopy(source["selection"]),
            "label_blind_confirmatory_selection": True,
            "variable_mutations_per_anchor": True,
        },
        "nodes": nodes,
        "edges": edges,
        "test_evaluated": False,
        "forbidden_sequences_loaded": False,
    }
    identity = "\n".join(row["sequence_sha256"] for row in nodes)
    report = {
        "schema": "PLS_cost_aware_selected_exact_manifest_report_v2",
        "anchors": len(selected_anchors),
        "single_mutation_edges": len(edges),
        "unique_sequence_queries_including_cached_anchors": len(nodes),
        "nodes_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    return manifest, report


def _write_entities(manifest: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["entity_id", "sequence_sha256", "sequence", "length"],
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(manifest["nodes"]):
            writer.writerow({
                "entity_id": f"cost_gate_{index:06d}",
                "sequence_sha256": row["sequence_sha256"],
                "sequence": row["sequence"],
                "length": row["length"],
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-entities", type=Path, required=True)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    calibration = json.loads(arguments.calibration.read_text())
    if protocol.get("test_evaluated") is not False or calibration.get("test_evaluated") is not False:
        raise ValueError("test evaluation is permanently disabled")
    if calibration.get("protocol_sha256") != _sha256(arguments.protocol):
        raise ValueError("calibration does not match the frozen protocol")
    primary_id = protocol["primary_policy_id"]
    calibrated = next(row for row in calibration["policies"] if row["policy_id"] == primary_id)
    section = protocol["fresh_confirmatory"]
    landscapes = load_landscapes(section["manifest"], section["fixed_scores"])
    runtime_model = load_runtime_cost_model(protocol["runtime_cost_model"])
    selections = select_candidates(landscapes, calibrated, runtime_model)
    source = json.loads(Path(section["manifest"]).read_text())
    selected_manifest, report = _selected_manifest(source, selections)
    result = {
        "schema": "PLS_cost_aware_confirmatory_selection_v2",
        "status": "frozen_before_any_confirmatory_exact_mutant_query",
        "protocol_sha256": _sha256(arguments.protocol),
        "calibration_sha256": _sha256(arguments.calibration),
        "confirmatory_manifest_sha256": _sha256(Path(section["manifest"])),
        "primary_policy": calibrated,
        "anchors": len(selections),
        "selected_exact_queries": sum(row["selected_queries"] for row in selections),
        "predicted_gpu_seconds": sum(row["predicted_marginal_gpu_seconds"] for row in selections),
        "predicted_exhaustive_gpu_seconds": sum(row["predicted_exhaustive_gpu_seconds"] for row in selections),
        "selections": selections,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.output_selection.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    arguments.output_manifest.write_text(json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n")
    arguments.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_entities(selected_manifest, arguments.output_entities)
    print(json.dumps({key: value for key, value in result.items() if key != "selections"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
