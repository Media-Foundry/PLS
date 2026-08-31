"""Build an auditable GB1 edit landscape from the official eLife supplements.

The parser uses only the Python standard library for XLSX input. It deliberately
keeps measured and publication-imputed values separate. Generated node arrays are
ignored by Git; the aggregate integrity report is versioned.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
XML_NAMESPACE = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXPECTED_MEASURED_SHA256 = "ac9a16295cd24110ad3ef4bcc57741f023ff9f3f5c3d9f5fdc0a94b996dca9b7"
EXPECTED_IMPUTED_SHA256 = "01db8ef421236dc35ca0481e8c9afe9d1673cb3ac4da1e692a3486ad6a4b807f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_npz(path: Path, **arrays) -> None:
    """Write NPZ content with fixed ZIP metadata so its SHA-256 is reproducible."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            archive.writestr(member, buffer.getvalue())


def column_index(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference)
    if not match:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def xlsx_rows(path: Path):
    """Yield values from the first worksheet without an Excel dependency."""
    with zipfile.ZipFile(path) as archive:
        shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.findall(".//x:t", XML_NAMESPACE))
            for item in shared_root.findall("x:si", XML_NAMESPACE)
        ]
        with archive.open("xl/worksheets/sheet1.xml") as worksheet:
            for _, element in ElementTree.iterparse(worksheet, events=("end",)):
                if element.tag != f"{{{XML_NAMESPACE['x']}}}row":
                    continue
                cells: dict[int, str] = {}
                for cell in element.findall("x:c", XML_NAMESPACE):
                    value = cell.find("x:v", XML_NAMESPACE)
                    text = "" if value is None or value.text is None else value.text
                    if cell.get("t") == "s" and text:
                        text = shared[int(text)]
                    cells[column_index(cell.get("r", ""))] = text
                width = max(cells, default=-1) + 1
                yield [cells.get(index, "") for index in range(width)]
                element.clear()


def load_measured(path: Path) -> dict[str, tuple[float, int, int, int]]:
    rows = xlsx_rows(path)
    header = next(rows)
    if header[:5] != ["Variants", "HD", "Count input", "Count selected", "Fitness"]:
        raise ValueError(f"unexpected measured header: {header}")
    result = {}
    for row in rows:
        if not row or not row[0]:
            continue
        variant = row[0]
        if variant in result:
            raise ValueError(f"duplicate measured variant: {variant}")
        result[variant] = (float(row[4]), int(row[1]), int(row[2]), int(row[3]))
    return result


def load_imputed(path: Path) -> dict[str, float]:
    rows = xlsx_rows(path)
    header = next(rows)
    if header[:2] != ["Variants", "Imputed fitness"]:
        raise ValueError(f"unexpected imputed header: {header}")
    result = {}
    for row in rows:
        if not row or not row[0]:
            continue
        if row[0] in result:
            raise ValueError(f"duplicate imputed variant: {row[0]}")
        result[row[0]] = float(row[1])
    return result


def build(measured_path: Path, imputed_path: Path, output: Path, report_path: Path) -> dict:
    observed_hashes = {"measured": sha256(measured_path), "imputed": sha256(imputed_path)}
    expected_hashes = {"measured": EXPECTED_MEASURED_SHA256, "imputed": EXPECTED_IMPUTED_SHA256}
    if observed_hashes != expected_hashes:
        raise ValueError(f"GB1 source checksum mismatch: {observed_hashes}")
    measured, imputed = load_measured(measured_path), load_imputed(imputed_path)
    if set(measured) & set(imputed):
        raise ValueError("measured and imputed GB1 variants overlap")
    variants = ["".join(values) for values in itertools.product(AMINO_ACIDS, repeat=4)]
    expected = set(variants)
    if set(measured) | set(imputed) != expected:
        missing = len(expected - set(measured) - set(imputed))
        extra = len((set(measured) | set(imputed)) - expected)
        raise ValueError(f"GB1 landscape is not complete: missing={missing}, extra={extra}")
    token_by_amino_acid = {value: index for index, value in enumerate(AMINO_ACIDS)}
    tokens = np.asarray([[token_by_amino_acid[value] for value in variant] for variant in variants], dtype=np.uint8)
    fitness = np.empty(len(variants), dtype=np.float32)
    is_measured = np.zeros(len(variants), dtype=np.bool_)
    hamming_from_wild_type = np.empty(len(variants), dtype=np.uint8)
    input_count = np.full(len(variants), -1, dtype=np.int64)
    selected_count = np.full(len(variants), -1, dtype=np.int64)
    wild_type = "VDGV"
    for index, variant in enumerate(variants):
        hamming_from_wild_type[index] = sum(a != b for a, b in zip(variant, wild_type))
        if variant in measured:
            value, reported_hamming, input_reads, selected_reads = measured[variant]
            if reported_hamming != hamming_from_wild_type[index]:
                raise ValueError(f"reported Hamming distance mismatch: {variant}")
            fitness[index] = value;is_measured[index] = True
            input_count[index] = input_reads;selected_count[index] = selected_reads
        else:
            fitness[index] = imputed[variant]
    output.parent.mkdir(parents=True, exist_ok=True)
    deterministic_npz(
        output, tokens=tokens, fitness=fitness, is_measured=is_measured,
        hamming_from_wild_type=hamming_from_wild_type,
        input_count=input_count, selected_count=selected_count,
        amino_acids=np.asarray(list(AMINO_ACIDS)), wild_type=np.asarray(wild_type),
    )
    report = {
        "schema": "PLS_EditFlow_GB1_Wu2016_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_doi": "10.7554/eLife.16965",
        "source_sha256": observed_hashes,
        "output": str(output), "output_sha256": sha256(output),
        "variants": len(variants), "measured_variants": int(is_measured.sum()),
        "imputed_variants": int((~is_measured).sum()), "wild_type": wild_type,
        "wild_type_fitness": float(fitness[variants.index(wild_type)]),
        "primary_evaluation": "measured variants only",
        "test_evaluated": False,
        "note": "This audit does not inspect or select PLS test entities. Imputed GB1 values are excluded from primary experimental-regret claims."
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", type=Path, default=Path("preparation/external/gb1_wu2016/elife-16965-supp1-v4.xlsx"))
    parser.add_argument("--imputed", type=Path, default=Path("preparation/external/gb1_wu2016/elife-16965-supp2-v4.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/generated/gb1_wu2016_editflow_v1.npz"))
    parser.add_argument("--report", type=Path, default=Path("benchmark/gb1_wu2016_report.json"))
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.measured, arguments.imputed, arguments.output, arguments.report), indent=2))


if __name__ == "__main__":
    main()
