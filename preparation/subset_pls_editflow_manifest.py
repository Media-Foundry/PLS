"""Create a deterministic split-specific edge subset of a safe PLS manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from preparation.build_pls_editflow_manifest import validate_manifest, write_entities


def build_subset(
    source: dict,
    mutations_per_anchor_by_split: dict[str, int],
) -> tuple[dict, dict]:
    if source.get("test_evaluated") is not False:
        raise ValueError("subset construction requires a test-free source manifest")
    nodes = source["nodes"]
    edges_by_anchor: dict[int, list[dict]] = defaultdict(list)
    for edge in source["edges"]:
        edges_by_anchor[int(edge["anchor_rank"])].append(edge)
    for edges in edges_by_anchor.values():
        edges.sort(key=lambda row: int(row["edge_index"]))

    selected_old_indices = []
    selected_edges = []
    anchors = [node for node in nodes if node["kind"] == "anchor"]
    for anchor in anchors:
        split = anchor["split"]
        if split not in {"train", "validation"}:
            raise ValueError("subset construction encountered a forbidden split")
        requested = int(mutations_per_anchor_by_split[split])
        candidates = edges_by_anchor[int(anchor["anchor_rank"])]
        if requested < 1 or len(candidates) < requested:
            raise ValueError(f"invalid mutation subset for {split} anchor")
        chosen = candidates[:requested]
        selected_old_indices.append(int(anchor["node_index"]))
        selected_old_indices.extend(int(edge["target_node"]) for edge in chosen)
        selected_edges.extend(chosen)

    old_to_new = {}
    subset_nodes = []
    for old_index in selected_old_indices:
        if old_index in old_to_new:
            raise ValueError("source manifest contains a repeated selected node")
        node = copy.deepcopy(nodes[old_index])
        node["node_index"] = len(subset_nodes)
        old_to_new[old_index] = node["node_index"]
        subset_nodes.append(node)
    subset_edges = []
    for edge in selected_edges:
        subset_edges.append({
            **copy.deepcopy(edge),
            "edge_index": len(subset_edges),
            "source_node": old_to_new[int(edge["source_node"])],
            "target_node": old_to_new[int(edge["target_node"])],
        })

    selection = copy.deepcopy(source.get("selection", {}))
    selection["subset_mutations_per_anchor_by_split"] = {
        split: int(count) for split, count in mutations_per_anchor_by_split.items()
    }
    selection["subset_label_blind"] = True
    manifest = {
        **{key: copy.deepcopy(value) for key, value in source.items() if key not in {"nodes", "edges", "selection"}},
        "schema": "PLS_EditFlow_sequence_oracle_subset_manifest_v1",
        "selection": selection,
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
        "schema": "PLS_EditFlow_sequence_oracle_subset_report_v1",
        "anchors": len(anchors),
        "anchors_by_split": {
            split: sum(anchor["split"] == split for anchor in anchors)
            for split in ("train", "validation")
        },
        "mutations_per_anchor_by_split": selection["subset_mutations_per_anchor_by_split"],
        "single_mutation_edges": len(subset_edges),
        "unique_sequence_queries": len(subset_nodes),
        "nodes_sha256": hashlib.sha256(canonical_nodes.encode()).hexdigest(),
        "edges_sha256": hashlib.sha256(canonical_edges.encode()).hexdigest(),
        "selection_used_target_values": False,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    validate_manifest(manifest, report)
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-mutations", type=int, required=True)
    parser.add_argument("--validation-mutations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--entities-output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest, report = build_subset(
        json.loads(arguments.manifest.read_text()),
        {"train": arguments.train_mutations, "validation": arguments.validation_mutations},
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_entities(manifest, arguments.entities_output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
