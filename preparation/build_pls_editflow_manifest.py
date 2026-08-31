"""Build a strict-train/validation PLS single-mutation oracle manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def ranked_items(items, *, salt: str, identity) -> list:
    return sorted(
        items,
        key=lambda item: (hashlib.sha256(f"{salt}:{identity(item)}".encode()).digest(), identity(item)),
    )


def select_anchors(
    candidates_by_split: dict[str, list[dict]],
    counts: dict[str, int],
    *,
    salt: str,
) -> list[dict]:
    selected = []
    for split in ("train", "validation"):
        requested = int(counts.get(split, 0))
        ranked = ranked_items(
            candidates_by_split.get(split, []),
            salt=f"{salt}:{split}",
            identity=lambda row: row["sequence_sha256"],
        )
        if len(ranked) < requested:
            raise ValueError(f"not enough eligible {split} anchors")
        selected.extend(ranked[:requested])
    return selected


def mutation_candidates(sequence: str, alphabet: str, *, salt: str) -> list[dict]:
    candidates = []
    for position, source in enumerate(sequence):
        for target in alphabet:
            if target == source:
                continue
            identity = f"{position}:{source}>{target}"
            priority = hashlib.sha256(f"{salt}:{identity}".encode()).digest()
            candidates.append({
                "position_zero_based": position,
                "position_one_based": position + 1,
                "source_residue": source,
                "target_residue": target,
                "priority": priority,
            })
    candidates.sort(key=lambda row: (row["priority"], row["position_zero_based"], row["target_residue"]))
    return candidates


def build_manifest(config: dict) -> tuple[dict, dict]:
    if config.get("evaluate_test", False):
        raise ValueError("test evaluation is permanently disabled")
    allowed_splits = tuple(config["allowed_splits"])
    if allowed_splits != ("train", "validation"):
        raise ValueError("PLS EditFlow manifests are restricted to train and validation")
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
                # Keep only the digest for collision prevention. Never retain a
                # forbidden entity's sequence, observation, or target value.
                forbidden_hashes.add(digest)

    source_membership: dict[str, set[str]] = defaultdict(set)
    with Path(config["observation_split"]).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in allowed_splits:
                continue
            if row["source_dataset"] == config["source_dataset"]:
                source_membership[row["split"]].add(row["sequence_sha256"])

    rows = []
    all_entity_hashes = set(allowed_by_hash) | forbidden_hashes
    with Path(config["entities"]).open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            digest = row["sequence_sha256"]
            if digest not in allowed_by_hash:
                continue
            split, component = allowed_by_hash[digest]
            sequence = row["sequence"]
            length = int(row["length"])
            if digest not in source_membership[split]:
                continue
            if not int(config["minimum_length"]) <= length <= int(config["maximum_length"]):
                continue
            if any(residue not in alphabet for residue in sequence):
                continue
            rows.append({
                "entity_index": index,
                "sequence_sha256": digest,
                "component_root_sha256": component,
                "split": split,
                "sequence": sequence,
                "length": length,
            })

    status = np.load(config["structure_status"], mmap_mode="r")
    candidates_by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["entity_index"] < len(status) and int(status[row["entity_index"]]) == 1:
            candidates_by_split[row["split"]].append(row)
    anchors = select_anchors(
        candidates_by_split,
        config["anchor_counts"],
        salt=config["anchor_salt"],
    )

    nodes = []
    edges = []
    generated_hashes = set()
    collision_rejections = 0
    mutations_per_anchor = int(config["mutations_per_anchor"])
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
        selected_mutations = 0
        candidates = mutation_candidates(
            anchor["sequence"],
            alphabet,
            salt=f"{config['mutation_salt']}:{anchor['sequence_sha256']}",
        )
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
                "mutation": {
                    key: value
                    for key, value in mutation.items()
                    if key != "priority"
                },
            })
            edges.append({
                "edge_index": len(edges),
                "anchor_rank": anchor_rank,
                "source_node": parent_node,
                "target_node": target_node,
            })
            generated_hashes.add(digest)
            selected_mutations += 1
            if selected_mutations == mutations_per_anchor:
                break
        if selected_mutations != mutations_per_anchor:
            raise RuntimeError("could not construct the exact per-anchor mutation budget")

    manifest = {
        "schema": "PLS_EditFlow_sequence_oracle_manifest_v1",
        "oracle_manifest": config["oracle_manifest"],
        "selection": {
            "allowed_splits": list(allowed_splits),
            "source_dataset": config["source_dataset"],
            "anchor_counts": config["anchor_counts"],
            "anchor_salt": config["anchor_salt"],
            "mutation_salt": config["mutation_salt"],
            "mutations_per_anchor": mutations_per_anchor,
            "minimum_length": int(config["minimum_length"]),
            "maximum_length": int(config["maximum_length"]),
            "label_blind": True,
        },
        "nodes": nodes,
        "edges": edges,
        "test_evaluated": False,
        "forbidden_sequences_loaded": False,
    }
    canonical_nodes = "\n".join(row["sequence_sha256"] for row in nodes)
    canonical_edges = "\n".join(
        f"{row['source_node']}->{row['target_node']}" for row in edges
    )
    report = {
        "schema": "PLS_EditFlow_sequence_oracle_manifest_report_v1",
        "anchors": len(anchors),
        "anchors_by_split": {
            split: sum(anchor["split"] == split for anchor in anchors)
            for split in allowed_splits
        },
        "unique_sequence_queries": len(nodes),
        "single_mutation_edges": len(edges),
        "mutations_per_anchor": mutations_per_anchor,
        "collision_rejections": collision_rejections,
        "nodes_sha256": hashlib.sha256(canonical_nodes.encode()).hexdigest(),
        "edges_sha256": hashlib.sha256(canonical_edges.encode()).hexdigest(),
        "selection_used_target_values": False,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    return manifest, report


def validate_manifest(manifest: dict, report: dict) -> None:
    """Hard-fail on split, identity, mutation, or query-budget inconsistencies."""
    if manifest.get("test_evaluated") is not False:
        raise ValueError("manifest violates the permanent test freeze")
    if manifest.get("forbidden_sequences_loaded") is not False:
        raise ValueError("manifest retained a forbidden sequence")
    nodes = manifest["nodes"]
    edges = manifest["edges"]
    if [row["node_index"] for row in nodes] != list(range(len(nodes))):
        raise ValueError("node indices must be consecutive")
    hashes = [row["sequence_sha256"] for row in nodes]
    if len(set(hashes)) != len(hashes):
        raise ValueError("oracle query nodes must be exact-sequence unique")
    for row in nodes:
        if row["split"] not in {"train", "validation"}:
            raise ValueError("oracle manifest contains a forbidden split")
        if sequence_sha256(row["sequence"]) != row["sequence_sha256"]:
            raise ValueError("sequence SHA-256 mismatch")
        if len(row["sequence"]) != int(row["length"]):
            raise ValueError("sequence length mismatch")
    for expected_edge, edge in enumerate(edges):
        if int(edge["edge_index"]) != expected_edge:
            raise ValueError("edge indices must be consecutive")
        source = nodes[int(edge["source_node"])]
        target = nodes[int(edge["target_node"])]
        if source["kind"] != "anchor" or target["kind"] != "single_mutant":
            raise ValueError("every edge must connect an anchor to a single mutant")
        if source["anchor_rank"] != target["anchor_rank"]:
            raise ValueError("edge crosses anchor landscapes")
        if source["split"] != target["split"]:
            raise ValueError("mutant did not inherit its anchor split")
        if source["component_root_sha256"] != target["component_root_sha256"]:
            raise ValueError("mutant did not inherit its anchor component")
        differences = [
            position
            for position, (left, right) in enumerate(zip(source["sequence"], target["sequence"]))
            if left != right
        ]
        if differences != [int(target["mutation"]["position_zero_based"])]:
            raise ValueError("edge is not the declared single substitution")
        position = differences[0]
        if source["sequence"][position] != target["mutation"]["source_residue"]:
            raise ValueError("mutation source residue mismatch")
        if target["sequence"][position] != target["mutation"]["target_residue"]:
            raise ValueError("mutation target residue mismatch")
    canonical_nodes = "\n".join(hashes)
    canonical_edges = "\n".join(
        f"{row['source_node']}->{row['target_node']}" for row in edges
    )
    if hashlib.sha256(canonical_nodes.encode()).hexdigest() != report["nodes_sha256"]:
        raise ValueError("node manifest identity mismatch")
    if hashlib.sha256(canonical_edges.encode()).hexdigest() != report["edges_sha256"]:
        raise ValueError("edge manifest identity mismatch")
    if report.get("test_sequences_queried") != 0 or report.get("test_evaluated") is not False:
        raise ValueError("report violates the permanent test freeze")


def write_entities(manifest: dict, path: Path) -> None:
    """Materialize the safe manifest nodes for existing feature extractors."""
    fields = ["entity_id", "sequence_sha256", "sequence", "length"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in manifest["nodes"]:
            writer.writerow({
                "entity_id": f"editflow_{int(node['node_index']):06d}",
                "sequence_sha256": node["sequence_sha256"],
                "sequence": node["sequence"],
                "length": int(node["length"]),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--entities-output", type=Path)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    manifest, report = build_manifest(config)
    validate_manifest(manifest, report)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if arguments.entities_output is not None:
        write_entities(manifest, arguments.entities_output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
