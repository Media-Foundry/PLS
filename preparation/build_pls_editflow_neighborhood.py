"""Build a strict-train exhaustive single-mutant neighborhood manifest.

Emits every valid single substitution of each anchor, not a sampled budget. The
anchor selection, split guards, collision rules and validation are imported from
the audited sampled builder so the two agree by construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from preparation.build_pls_editflow_manifest import (
    load_excluded_anchor_identities,
    mutation_candidates,
    select_anchors,
    sequence_sha256,
    validate_manifest,
    write_entities,
)


def build(config: dict) -> tuple[dict, dict]:
    if config.get("evaluate_test", False):
        raise ValueError("test evaluation is permanently disabled")
    allowed_splits = tuple(config["allowed_splits"])
    if allowed_splits != ("train", "validation"):
        raise ValueError("neighborhood manifests are restricted to train and validation")
    alphabet = str(config["alphabet"])
    if len(alphabet) != 20 or len(set(alphabet)) != 20:
        raise ValueError("alphabet must contain 20 unique residues")

    allowed_by_hash: dict[str, tuple[str, str]] = {}
    forbidden_hashes: set[str] = set()
    with Path(config["entity_split"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            digest = row["sequence_sha256"]
            if row["split"] in allowed_splits:
                allowed_by_hash[digest] = (row["split"], row["component_root_sha256"])
            else:
                forbidden_hashes.add(digest)

    source_membership: dict[str, set[str]] = defaultdict(set)
    with Path(config["observation_split"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] in allowed_splits and row["source_dataset"] == config["source_dataset"]:
                source_membership[row["split"]].add(row["sequence_sha256"])

    rows = []
    all_entity_hashes = set(allowed_by_hash) | forbidden_hashes
    with Path(config["entities"]).open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            digest = row["sequence_sha256"]
            if digest not in allowed_by_hash:
                continue
            split, component = allowed_by_hash[digest]
            length = int(row["length"])
            if digest not in source_membership[split]:
                continue
            if not int(config["minimum_length"]) <= length <= int(config["maximum_length"]):
                continue
            if any(residue not in alphabet for residue in row["sequence"]):
                continue
            rows.append({
                "entity_index": index,
                "sequence_sha256": digest,
                "component_root_sha256": component,
                "split": split,
                "sequence": row["sequence"],
                "length": length,
            })

    exclusion_manifests = [Path(v) for v in config.get("exclude_anchor_manifests", [])]
    excluded_components, excluded_anchor_hashes = load_excluded_anchor_identities(
        exclusion_manifests, allowed_splits
    )
    if excluded_components:
        rows = [r for r in rows if r["component_root_sha256"] not in excluded_components]

    status = np.load(config["structure_status"], mmap_mode="r")
    candidates_by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["entity_index"] < len(status) and int(status[row["entity_index"]]) == 1:
            candidates_by_split[row["split"]].append(row)

    anchors = select_anchors(
        candidates_by_split,
        config["anchor_counts"],
        salt=config["anchor_salt"],
        unique_components=bool(config.get("component_unique_anchors", False)),
        priority_hashes_by_split=defaultdict(list),
    )

    nodes: list[dict] = []
    edges: list[dict] = []
    generated_hashes: set[str] = set()
    collision_rejections = 0
    neighborhood_sizes = []
    for anchor_rank, anchor in enumerate(anchors):
        parent_node = len(nodes)
        nodes.append({
            "node_index": parent_node,
            "anchor_rank": anchor_rank,
            "kind": "anchor",
            "split": anchor["split"],
            "component_root_sha256": anchor["component_root_sha256"],
            "sequence_sha256": anchor["sequence_sha256"],
            "sequence": anchor["sequence"],
            "length": anchor["length"],
        })
        generated_hashes.add(anchor["sequence_sha256"])
        # Enumerate in a stable, position-major order: this is the whole
        # neighborhood, so the hash priority only fixes tie-breaking.
        candidates = sorted(
            mutation_candidates(
                anchor["sequence"], alphabet,
                salt=f"{config['mutation_salt']}:{anchor['sequence_sha256']}",
            ),
            key=lambda row: (row["position_zero_based"], row["target_residue"]),
        )
        accepted = 0
        for mutation in candidates:
            position = mutation["position_zero_based"]
            sequence = (
                anchor["sequence"][:position]
                + mutation["target_residue"]
                + anchor["sequence"][position + 1:]
            )
            digest = sequence_sha256(sequence)
            if digest in all_entity_hashes or digest in generated_hashes:
                collision_rejections += 1
                continue
            target_node = len(nodes)
            nodes.append({
                "node_index": target_node,
                "anchor_rank": anchor_rank,
                "kind": "single_mutant",
                "split": anchor["split"],
                "component_root_sha256": anchor["component_root_sha256"],
                "sequence_sha256": digest,
                "sequence": sequence,
                "length": anchor["length"],
                "mutation": {k: v for k, v in mutation.items() if k != "priority"},
            })
            edges.append({
                "edge_index": len(edges),
                "anchor_rank": anchor_rank,
                "source_node": parent_node,
                "target_node": target_node,
            })
            generated_hashes.add(digest)
            accepted += 1
        expected = 19 * int(anchor["length"])
        if accepted + collision_rejections < expected - 5000:
            raise RuntimeError("neighborhood is implausibly incomplete")
        neighborhood_sizes.append({
            "anchor_rank": anchor_rank,
            "length": int(anchor["length"]),
            "theoretical": expected,
            "materialized": accepted,
        })

    manifest = {
        "schema": "PLS_EditFlow_sequence_oracle_manifest_v1",
        "oracle_manifest": config["oracle_manifest"],
        "selection": {
            "allowed_splits": list(allowed_splits),
            "source_dataset": config["source_dataset"],
            "anchor_counts": config["anchor_counts"],
            "anchor_salt": config["anchor_salt"],
            "mutation_salt": config["mutation_salt"],
            "mutations_per_anchor": None,
            "exhaustive_neighborhood": True,
            "minimum_length": int(config["minimum_length"]),
            "maximum_length": int(config["maximum_length"]),
            "component_unique_anchors": bool(config.get("component_unique_anchors", False)),
            "priority_anchor_manifest": None,
            "exclude_anchor_manifests": [str(p) for p in exclusion_manifests],
            "excluded_components": len(excluded_components),
            "label_blind": True,
        },
        "nodes": nodes,
        "edges": edges,
        "test_evaluated": False,
        "forbidden_sequences_loaded": False,
    }
    canonical_nodes = "\n".join(r["sequence_sha256"] for r in nodes)
    canonical_edges = "\n".join(f"{r['source_node']}->{r['target_node']}" for r in edges)
    report = {
        "schema": "PLS_EditFlow_neighborhood_manifest_report_v1",
        "anchors": len(anchors),
        "exhaustive_neighborhood": True,
        "anchors_by_split": {
            s: sum(a["split"] == s for a in anchors) for s in allowed_splits
        },
        "unique_components_by_split": {
            s: len({a["component_root_sha256"] for a in anchors if a["split"] == s})
            for s in allowed_splits
        },
        "excluded_anchor_manifests": [str(p) for p in exclusion_manifests],
        "excluded_components": len(excluded_components),
        "excluded_anchor_hashes": len(excluded_anchor_hashes),
        "unique_sequence_queries": len(nodes),
        "single_mutation_edges": len(edges),
        "mutations_per_anchor": None,
        "neighborhood_sizes": neighborhood_sizes,
        "collision_rejections": collision_rejections,
        "nodes_sha256": hashlib.sha256(canonical_nodes.encode()).hexdigest(),
        "edges_sha256": hashlib.sha256(canonical_edges.encode()).hexdigest(),
        "selection_used_target_values": False,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--entities-output", type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    manifest, report = build(config)
    validate_manifest(manifest, report)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.entities_output:
        write_entities(manifest, arguments.entities_output)
    summary = {k: report[k] for k in (
        "anchors", "unique_sequence_queries", "single_mutation_edges",
        "collision_rejections", "test_sequences_queried", "test_evaluated",
    )}
    print(json.dumps(summary, indent=2))
    for row in report["neighborhood_sizes"]:
        print(f"  anchor {row['anchor_rank']:>2}  L={row['length']:>4}  "
              f"materialized={row['materialized']:>5}/{row['theoretical']}")


if __name__ == "__main__":
    main()
