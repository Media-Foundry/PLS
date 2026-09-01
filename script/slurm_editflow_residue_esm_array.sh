#!/usr/bin/env bash
#SBATCH --job-name=pls_residue_esm
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/residue_esm_%A_%a.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/residue_esm_%A_%a.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
shard_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
artifact_root="$stage_root/artifacts/oracles/pls_editflow_poc_v1"
entities="$stage_root/benchmark/generated/pls_editflow_entities_poc_v1.csv"
offsets="$artifact_root/structure_v4_compact/offsets.npy"
status="$artifact_root/structure_v4_raw/status.npy"
pca="$stage_root/artifacts/features/esm2_residue_train_pca_256.npz"

export PYTHONPATH="$stage_root/repo/src:$stage_root/repo"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
cd "$stage_root/repo"

"$python_bin" -m pls.features.extract_esm2_residue \
    --entities "$entities" --offsets "$offsets" --structure-status "$status" \
    --output "$artifact_root/residue_esm2_raw" --shard-count 2 \
    --shard-index "$shard_index" --cuda-slurm --token-budget 4096

"$python_bin" preparation/project_esm2_residue_pca.py \
    --entities "$entities" --offsets "$offsets" --structure-status "$status" \
    --source "$artifact_root/residue_esm2_raw" --pca "$pca" \
    --output "$artifact_root/residue_esm2_pca" --shard-count 2 \
    --shard-index "$shard_index" --cuda-slurm --residue-budget 65536

"$python_bin" - "$artifact_root/residue_esm2_pca/shard_${shard_index}_environment.json" <<'PY'
import importlib.metadata
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
record = {
    "python": sys.version,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("fair-esm", "numpy", "torch")
    },
    "test_evaluated": False,
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
