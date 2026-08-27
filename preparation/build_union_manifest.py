"""Build provenance-preserving observation and canonical sequence manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


OBSERVATION_FIELDS = [
    "observation_id", "entity_id", "sequence_sha256", "source_dataset",
    "source_record_id", "source_detail", "endpoint", "target_value",
    "target_kind", "upstream_split", "sequence",
]


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uesolds", type=Path, required=True)
    parser.add_argument("--pdbsol-dir", type=Path, required=True)
    parser.add_argument("--esol", type=Path, required=True)
    parser.add_argument("--observations-output", type=Path, required=True)
    parser.add_argument("--entities-output", type=Path, required=True)
    args = parser.parse_args()

    observations: list[dict[str, str]] = []
    for row in read_rows(args.uesolds):
        observations.append({
            "observation_id": row["observation_id"],
            "sequence_sha256": row["sequence_sha256"],
            "source_dataset": row["source_dataset"],
            "source_record_id": row["source_record_id"],
            "source_detail": row["source_header"],
            "endpoint": row["endpoint"],
            "target_value": row["label"],
            "target_kind": "binary",
            "upstream_split": row["upstream_split"],
            "sequence": row["sequence"],
        })

    for split_name, filename in (("train", "train.csv"), ("validation", "valid.csv"), ("test", "test.csv")):
        for index, row in enumerate(read_rows(args.pdbsol_dir / filename), 1):
            sequence = row["aa_seq"].strip().upper()
            observations.append({
                "observation_id": f"pdbsol:{split_name}:{index}",
                "sequence_sha256": sequence_hash(sequence),
                "source_dataset": "PDBSol_ProtSolM",
                "source_record_id": row["name"],
                "source_detail": row["detail"],
                "endpoint": "weak_composite_binary_solubility",
                "target_value": row["label"],
                "target_kind": "binary",
                "upstream_split": split_name,
                "sequence": sequence,
            })

    for row in read_rows(args.esol):
        observations.append({
            "observation_id": f"esol:fgnnsol:{row['split']}:{row['protein_id']}",
            "sequence_sha256": row["sequence_sha256"],
            "source_dataset": "eSOL_FGNNSol",
            "source_record_id": row["protein_id"],
            "source_detail": "",
            "endpoint": "PURE_continuous_solubility",
            "target_value": row["solubility"],
            "target_kind": "continuous",
            "upstream_split": row["split"],
            "sequence": row["sequence"],
        })

    entities: dict[str, str] = {}
    for row in observations:
        digest = sequence_hash(row["sequence"])
        if digest != row["sequence_sha256"]:
            raise ValueError(f"sequence hash mismatch in {row['observation_id']}")
        previous = entities.setdefault(digest, row["sequence"])
        if previous != row["sequence"]:
            raise ValueError(f"SHA-256 collision or inconsistent sequence: {digest}")

    ordered_hashes = sorted(entities)
    entity_ids = {digest: f"entity:{index:06d}" for index, digest in enumerate(ordered_hashes)}
    for row in observations:
        row["entity_id"] = entity_ids[row["sequence_sha256"]]

    args.observations_output.parent.mkdir(parents=True, exist_ok=True)
    with args.observations_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS)
        writer.writeheader()
        writer.writerows(observations)
    with args.entities_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["entity_id", "sequence_sha256", "sequence", "length"])
        writer.writeheader()
        for digest in ordered_hashes:
            sequence = entities[digest]
            writer.writerow({"entity_id": entity_ids[digest], "sequence_sha256": digest, "sequence": sequence, "length": len(sequence)})

    print(f"Wrote {len(observations):,} observations and {len(entities):,} entities")


if __name__ == "__main__":
    main()
