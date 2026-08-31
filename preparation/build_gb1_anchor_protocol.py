"""Build a fitness-blind GB1 confirmatory-anchor protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pls.editflow.hamming import (queried_nodes_sha256,
                                  select_hashed_anchors,
                                  variants_from_tokens)


def build_protocol(
    landscape_path: Path,
    count: int,
    salt: str,
    excluded: tuple[str, ...],
) -> dict:
    landscape = np.load(landscape_path)
    # Intentionally access only sequence identity and label availability.  The
    # fitness array is never loaded during anchor selection.
    tokens = landscape["tokens"].astype(np.int64)
    measured = landscape["is_measured"].astype(bool)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    variants = variants_from_tokens(tokens, alphabet)
    anchors = select_hashed_anchors(
        variants,
        measured,
        count,
        salt=salt,
        excluded=excluded,
    )
    return {
        "schema": "PLS_EditFlow_GB1_anchor_protocol_v1",
        "landscape": str(landscape_path),
        "selection": {
            "algorithm": "ascending sha256(salt:variant), node index tie-break",
            "salt": salt,
            "count": count,
            "measured_only": True,
            "fitness_accessed": False,
            "excluded_variants": list(excluded),
        },
        "anchors_sha256": queried_nodes_sha256(anchors),
        "anchors": [
            {"rank": rank, "node_index": int(node), "variant": variants[int(node)]}
            for rank, node in enumerate(anchors)
        ],
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscape", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    protocol = build_protocol(
        arguments.landscape,
        arguments.count,
        arguments.salt,
        tuple(arguments.exclude),
    )
    text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.write_text(text)


if __name__ == "__main__":
    main()
