"""Calibrate frozen cost-aware conformal policies on fresh SI30 components."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pls.editflow.cost_aware_gating import calibrate_policies, load_landscapes
from pls.editflow.runtime_cost import load_runtime_cost_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.read_text())
    if protocol.get("status") != "frozen_before_fresh_calibration_exact_scores":
        raise ValueError("cost-aware protocol was not frozen before calibration")
    if protocol.get("evaluate_test", False) or protocol.get("test_evaluated") is not False:
        raise ValueError("test evaluation is permanently disabled")
    section = protocol["fresh_calibration"]
    landscapes = load_landscapes(
        section["manifest"], section["fixed_scores"], section["exact_scores"]
    )
    if len(landscapes) != int(section["components"]):
        raise ValueError("fresh calibration component budget mismatch")
    runtime_model = load_runtime_cost_model(protocol["runtime_cost_model"])
    calibrated = calibrate_policies(landscapes, protocol["policies"], runtime_model)
    result = {
        "schema": "PLS_cost_aware_conformal_calibration_v2",
        "status": "frozen_quantiles_from_fresh_calibration_components",
        "protocol": str(arguments.protocol),
        "protocol_sha256": _sha256(arguments.protocol),
        "calibration_manifest": section["manifest"],
        "calibration_manifest_sha256": _sha256(Path(section["manifest"])),
        "fixed_scores_sha256": _sha256(Path(section["fixed_scores"])),
        "exact_scores_sha256": _sha256(Path(section["exact_scores"])),
        "runtime_cost_model_sha256": _sha256(Path(protocol["runtime_cost_model"])),
        "policies": calibrated,
        "test_sequences_queried": 0,
        "test_evaluated": False,
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
