"""Materialize a test-free fixed-parent structural cache for mutation ablations.

Every mutant receives its own exact-sequence cache key, but the underlying V4
structure tensor is a hardlink (or copy fallback) of its anchor.  Exact mutant
PLM features remain separate.  This defines an explicit fixed-backbone oracle;
it must never be described as an exact ESMFold query.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def raw_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / f"{digest}.pt"


def materialize(source: Path, destination: Path) -> str:
    if destination.is_file():
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        temporary = destination.with_name(f".{destination.name}.copying")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        return "copied"


def build_mapping(manifest: dict) -> list[tuple[dict, dict]]:
    if manifest.get("test_evaluated") is not False:
        raise ValueError("fixed-parent materialization requires a test-free manifest")
    nodes = manifest["nodes"]
    anchors = {
        int(node["anchor_rank"]): node
        for node in nodes
        if node["kind"] == "anchor"
    }
    mapping = []
    for node in nodes:
        if node["split"] not in {"train", "validation"}:
            raise ValueError("fixed-parent materialization encountered a forbidden split")
        anchor = anchors[int(node["anchor_rank"])]
        if anchor["split"] != node["split"]:
            raise ValueError("mutant and fixed parent do not share a split")
        if anchor["component_root_sha256"] != node["component_root_sha256"]:
            raise ValueError("mutant and fixed parent do not share an SI component")
        if int(anchor["length"]) != int(node["length"]):
            raise ValueError("fixed-parent reuse requires equal sequence length")
        mapping.append((node, anchor))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exact-raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text())
    mapping = build_mapping(manifest)
    outcomes = {"hardlinked": 0, "copied": 0, "skipped": 0}
    mutant_nodes = 0
    for node, anchor in mapping:
        source = raw_path(arguments.exact_raw_root, anchor["sequence_sha256"])
        if not source.is_file():
            raise FileNotFoundError(f"missing exact anchor V4 tensor: {source}")
        destination = raw_path(arguments.output_root, node["sequence_sha256"])
        outcomes[materialize(source, destination)] += 1
        mutant_nodes += node["kind"] == "single_mutant"
    status = np.ones(len(mapping), dtype=np.uint8)
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    np.save(arguments.output_root / "status.npy", status)
    report = {
        "schema": "PLS_EditFlow_fixed_parent_structure_cache_v1",
        "definition": "anchor_exact_v4_tensor_reused_under_exact_mutant_sequence_key",
        "nodes": len(mapping),
        "mutant_nodes": mutant_nodes,
        "outcomes": outcomes,
        "exact_mutant_folds_required": 0,
        "mutation_aware_structure_channels": False,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    (arguments.output_root / "materialization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
