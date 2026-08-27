"""Normalize PLM_Sol/UESolDS FASTA files without dropping observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


FILES = {
    "Train_dataset.fasta": "train",
    "validation_dataset.fasta": "validation",
    "test_dataset.fasta": "test",
}


def fasta_records(path: Path):
    header = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header, chunks = line[1:], []
            else:
                if header is None:
                    raise ValueError(f"sequence before header in {path}")
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks).upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observations = []
    for filename, upstream_split in FILES.items():
        for index, (header, sequence) in enumerate(
            fasta_records(args.input_dir / filename), start=1
        ):
            tokens = header.rsplit(maxsplit=1)
            if len(tokens) != 2 or tokens[1] not in {"A-0", "A-1"}:
                raise ValueError(f"unrecognized UESolDS header: {header!r}")
            upstream_id, encoded_label = tokens
            observations.append(
                {
                    "observation_id": f"uesolds:{upstream_split}:{index}",
                    "source_dataset": "UESolDS_PLM_Sol_1.1",
                    "source_record_id": upstream_id,
                    "source_header": header,
                    # The released FASTA does not expose its four-way subsource.
                    "source_subcollection": "not_provided_in_release",
                    "endpoint": "weak_binary_ecoli_expression_solubility",
                    "label": encoded_label.removeprefix("A-"),
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "upstream_split": upstream_split,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    print(f"Wrote {len(observations)} UESolDS observations to {args.output}")


if __name__ == "__main__":
    main()

