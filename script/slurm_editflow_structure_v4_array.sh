#!/usr/bin/env bash
#SBATCH --job-name=pls_v4_extract
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --array=0-7%2
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/v4_%A_%a.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/v4_%A_%a.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
shard_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 33554432 )); then
    echo "Refusing V4 extraction with less than 32 GiB host memory available" >&2
    exit 3
fi

export PYTHONPATH="$stage_root/repo/src:$stage_root/repo"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$stage_root/repo"
exec "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$stage_root/benchmark/generated/pls_editflow_oracle_manifest_poc_v1.json" \
    --plan "$stage_root/benchmark/generated/pls_editflow_oracle_query_plan_poc_v1.json" \
    --shard-index "$shard_index" \
    --pdb-root "$stage_root/artifacts/oracles/pls_editflow_poc_v1/esmfold" \
    --parent-raw-root "$stage_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$stage_root/artifacts/oracles/pls_editflow_poc_v1/structure_v4_raw" \
    --source-root "$stage_root/external_v4" \
    --workers 24
