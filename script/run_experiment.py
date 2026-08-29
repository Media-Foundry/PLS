"""Create a timestamped immutable experiment directory and run its trainer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--hip-override", type=int)
    parser.add_argument("--architecture-override", choices=("early", "late", "gated_residual", "film"))
    parser.add_argument("--pooling-override", choices=("mean", "attention", "plddt_gate", "dual_patch"))
    parser.add_argument("--name-suffix", type=str, default="")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.seed_override is not None:
        config["training"]["seed"] = args.seed_override
    if args.hip_override is not None:
        config["training"]["hip_device"] = args.hip_override
    if args.architecture_override is not None:
        config["model"]["architecture"] = args.architecture_override
    if args.pooling_override is not None:
        config["model"]["pooling"] = args.pooling_override
    if args.name_suffix:
        config["experiment_name"] += args.name_suffix
    timestamp = datetime.now().astimezone().strftime("%m-%d-%H-%M")
    run_dir = args.outputs_root / f"{config['experiment_name']}+{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(); (run_dir / "tensorboard").mkdir()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    environment = {
        "created_at": datetime.now().astimezone().isoformat(), "command": sys.argv,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "torch_hip": torch.version.hip,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("biopython", "fair-esm", "numpy", "scikit-learn", "scipy", "tensorboard")},
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    default_trainer = ("pls.training.train_pooled_structure" if "structure_groups" in config["model"]
                       else "pls.training.train_plm_heads")
    trainer_module = config.get("trainer_module", default_trainer)
    command = [sys.executable, "-m", trainer_module, "--config", str(run_dir / "config.json"),
               "--run-dir", str(run_dir)]
    child_environment = os.environ.copy()
    source_path = str((Path.cwd() / "src").resolve())
    child_environment["PYTHONPATH"] = source_path + (os.pathsep + child_environment["PYTHONPATH"]
                                                        if child_environment.get("PYTHONPATH") else "")
    print(run_dir, flush=True)
    with (run_dir / "output.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True,
                                env=child_environment)
    if result.returncode:
        raise SystemExit(f"training failed with exit code {result.returncode}; see {run_dir / 'output.log'}")


if __name__ == "__main__":
    main()
