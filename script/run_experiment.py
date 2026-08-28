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
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    timestamp = datetime.now().astimezone().strftime("%m-%d-%H-%M")
    run_dir = args.outputs_root / f"{config['experiment_name']}+{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(); (run_dir / "tensorboard").mkdir()
    shutil.copy2(args.config, run_dir / "config.json")
    environment = {
        "created_at": datetime.now().astimezone().isoformat(), "command": sys.argv,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "torch": torch.__version__, "torch_hip": torch.version.hip,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("biopython", "fair-esm", "numpy", "scikit-learn", "scipy", "tensorboard")},
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    command = [sys.executable, "-m", "pls.training.train_plm_heads", "--config", str(run_dir / "config.json"),
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
