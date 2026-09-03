"""Build the exact-fold manifest restricted to a persisted gated selection.

The subset holds every anchor plus only the mutants a frozen policy selected, so
exact feature extraction and scoring cover exactly the candidates that were
actually bought. It reads the selection artifact, never any exact score.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: dict, selected: set[int]) -> tuple[dict, dict]:
    if source.get("test_evaluated") is not False:
        raise ValueError("subset refuses a manifest without a test-free assertion")
    nodes = source["nodes"]
    if any(node["split"] not in {"train", "validation"} for node in nodes):
        raise ValueError("subset manifest contains a forbidden split")

    anchors = {
        int(node["anchor_rank"]): node for node in nodes if node["kind"] == "anchor"
    }
    edges_by_target = {int(edge["target_node"]): edge for edge in source["edges"]}
    missing = selected - set(edges_by_target)
    if missing:
        raise ValueError(f"{len(missing)} selected nodes are not mutant edge targets")

    ordered: list[int] = []
    kept_edges: list[dict] = []
    for rank in sorted(anchors):
        anchor = anchors[rank]
        ordered.append(int(anchor["node_index"]))
        chosen = sorted(
            (edges_by_target[i] for i in selected
             if int(edges_by_target[i]["anchor_rank"]) == rank),
            key=lambda edge: int(edge["edge_index"]),
        )
        for edge in chosen:
            ordered.append(int(edge["target_node"]))
            kept_edges.append(edge)

    old_to_new: dict[int, int] = {}
    subset_nodes: list[dict] = []
    for old_index in ordered:
        if old_index in old_to_new:
            raise ValueError("source manifest contains a repeated selected node")
        node = copy.deepcopy(nodes[old_index])
        node["node_index"] = len(subset_nodes)
        old_to_new[old_index] = node["node_index"]
        subset_nodes.append(node)

    subset_edges = []
    for edge in kept_edges:
        subset_edges.append({
            **copy.deepcopy(edge),
            "edge_index": len(subset_edges),
            "source_node": old_to_new[int(edge["source_node"])],
            "target_node": old_to_new[int(edge["target_node"])],
        })

    selection_block = copy.deepcopy(source.get("selection", {}))
    selection_block["subset_scope"] = "gated_selected_candidates_plus_all_anchors"
    selection_block["subset_label_blind"] = True
    selection_block["subset_used_exact_scores"] = False

    manifest = {
        **{
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key not in {"nodes", "edges", "selection"}
        },
        "schema": "PLS_EditFlow_sequence_oracle_subset_manifest_v1",
        "selection": selection_block,
        "nodes": subset_nodes,
        "edges": subset_edges,
        "test_evaluated": False,
        "forbidden_sequences_loaded": False,
    }
    canonical_nodes = "\n".join(row["sequence_sha256"] for row in subset_nodes)
    canonical_edges = "\n".join(
        f"{row['source_node']}->{row['target_node']}" for row in subset_edges
    )
    report = {
        "schema": "PLS_EditFlow_selected_subset_report_v1",
        "anchors": len(anchors),
        "selected_mutants": len(subset_edges),
        "unique_sequence_queries": len(subset_nodes),
        "nodes_sha256": hashlib.sha256(canonical_nodes.encode()).hexdigest(),
        "edges_sha256": hashlib.sha256(canonical_edges.encode()).hexdigest(),
        "selection_used_target_values": False,
        "selection_used_exact_scores": False,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entities-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    selection = json.loads(arguments.selection.read_text())
    if selection.get("test_evaluated") is not False:
        raise ValueError("selection artifact is not test-free")
    if selection.get("exact_scores_read") is not False:
        raise ValueError("selection artifact claims exact scores were read")
    if selection.get("confirmatory_manifest_sha256") != _sha256(arguments.manifest):
        raise ValueError("selection was produced against a different manifest revision")

    policy = next(
        (p for p in selection["policies"] if p["policy_id"] == arguments.policy_id),
        None,
    )
    if policy is None:
        raise ValueError(f"selection artifact has no policy {arguments.policy_id!r}")

    selected: set[int] = set()
    for row in policy["selections"]:
        selected.update(int(index) for index in row["selected_target_node_indices"])
    if len(selected) != int(policy["selected_queries_total"]):
        raise ValueError("selected node indices are not unique across anchors")

    source = json.loads(arguments.manifest.read_text())
    manifest, report = build(source, selected)
    report["policy_id"] = arguments.policy_id
    report["selection_artifact_sha256"] = _sha256(arguments.selection)

    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with arguments.entities_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entity_id", "sequence_sha256", "sequence", "length"])
        for node in manifest["nodes"]:
            writer.writerow([
                f"editflow_{node['node_index']:06d}",
                node["sequence_sha256"],
                node["sequence"],
                int(node["length"]),
            ])
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
