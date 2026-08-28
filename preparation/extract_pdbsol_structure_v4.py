"""Resumable extraction of raw PLS V4 residue-level features for PDBSol."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import torch
from Bio.PDB import PDBParser

from pls.features.external_v4_loader import load_external_v4
from pls.features.structure_v4_schema import adapt_v4_features


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}
_V4 = None


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def load_records(csv_root: Path) -> list[dict]:
    records = []
    seen_names = set()
    for upstream_split, filename in (("train", "train.csv"), ("validation", "valid.csv"), ("test", "test.csv")):
        with (csv_root / filename).open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                name = row["name"]
                if name in seen_names:
                    raise ValueError(f"duplicate PDBSol name: {name}")
                seen_names.add(name)
                sequence = row["aa_seq"].strip().upper()
                records.append({"name": name, "sequence": sequence,
                                "sequence_sha256": sequence_sha256(sequence),
                                "upstream_split": upstream_split})
    return records


def init_worker(source_root: str) -> None:
    global _V4
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _V4, _ = load_external_v4(Path(source_root))


def extract_one(task: tuple[dict, str, str, bool]) -> dict:
    record, pdb_root_string, output_root_string, overwrite = task
    started = time.monotonic()
    pdb_path = Path(pdb_root_string) / f"{record['name']}.ef.pdb"
    output_path = Path(output_root_string) / record["sequence_sha256"][:2] / f"{record['sequence_sha256']}.pt"
    base = {key: record[key] for key in ("name", "sequence_sha256", "upstream_split")}
    base.update({"pdb_path": str(pdb_path), "output_path": str(output_path)})
    if output_path.is_file() and not overwrite:
        return {**base, "status": "skipped", "seconds": 0.0}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            features = _V4.extract_complete_features_v4(pdb_path, mode="complex", normalization_stats=None)
        structure = PDBParser(QUIET=True).get_structure(record["name"], pdb_path)
        ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == "CA"]
        raw_plddt = torch.tensor([atom.get_bfactor() for atom in ca_atoms], dtype=torch.float32)
        adapted = adapt_v4_features(features, raw_plddt)
        structure_sequence = "".join(AA3_TO_1.get(name, "X") for name in features["residue_names"])
        payload = {
            **adapted,
            "spatial_scalar_raw_features": features["spatial_scalar_raw_features"].float(),
            "residue_names": features["residue_names"],
            "residue_indices": features["residue_indices"],
            "chain_ids": features["chain_ids"],
            "label_sequence": record["sequence"],
            "structure_sequence": structure_sequence,
            "sequence_exact_match": structure_sequence == record["sequence"],
            "sequence_sha256": record["sequence_sha256"],
            "source_record_id": record["name"],
            "upstream_split": record["upstream_split"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=f".{output_path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {**base, "status": "ok", "n_residues": adapted["n_residues"],
                "label_length": len(record["sequence"]), "sequence_exact_match": payload["sequence_exact_match"],
                "mean_plddt": float(adapted["plddt"].mean()), "seconds": time.monotonic() - started}
    except Exception as error:
        return {**base, "status": "failed", "error_type": type(error).__name__,
                "error": str(error), "seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--pdb-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/home/pc/Code/BIO/protein"))
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    records = load_records(args.csv_root)
    if args.max_files is not None:
        records = records[:args.max_files]
    missing = [r["name"] for r in records if not (args.pdb_root / f"{r['name']}.ef.pdb").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing structures; first={missing[:5]}")
    _, source_hashes = load_external_v4(args.source_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {"schema": "PLS_structure_v4_raw_90d_v1", "records": len(records), "workers": args.workers,
              "csv_root": str(args.csv_root.resolve()), "pdb_root": str(args.pdb_root.resolve()),
              "source_root": str(args.source_root.resolve()), "source_sha256": source_hashes,
              "normalization": "none; fit scalar statistics on strict train only", "created_unix": time.time()}
    (args.output_root / "extraction_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    tasks = [(record, str(args.pdb_root), str(args.output_root), args.overwrite) for record in records]
    counts = {"ok": 0, "skipped": 0, "failed": 0, "sequence_mismatch": 0}
    manifest = args.output_root / "extraction_manifest.jsonl"
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    with manifest.open("w", encoding="utf-8") as log, ctx.Pool(
        args.workers, initializer=init_worker, initargs=(str(args.source_root),), maxtasksperchild=200
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(extract_one, tasks, chunksize=1), 1):
            counts[result["status"]] += 1
            if result.get("sequence_exact_match") is False:
                counts["sequence_mismatch"] += 1
            log.write(json.dumps(result, sort_keys=True) + "\n")
            log.flush()
            if index == 1 or index % 250 == 0 or index == len(tasks):
                rate = index / max(time.monotonic() - started, 1e-9)
                print(json.dumps({"done": index, "total": len(tasks), "rate_files_s": round(rate, 2), **counts}), flush=True)
    summary = {**counts, "records": len(records), "elapsed_seconds": time.monotonic() - started}
    (args.output_root / "extraction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
