#!/usr/bin/env bash
#SBATCH --job-name=pls_esmfold
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --array=0-7%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/slurm_%A_%a.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/slurm_%A_%a.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
shard_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
visible_device=${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}

case "$visible_device" in
    0|1) ;;
    *)
        echo "Expected one Slurm-assigned A100 ordinal, got CUDA_VISIBLE_DEVICES=$visible_device" >&2
        exit 2
        ;;
esac

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 67108864 )); then
    echo "Refusing ESMFold start with less than 64 GiB host memory available" >&2
    exit 3
fi

export TORCH_HOME="$stage_root/torch"
export PYTHONPATH="$stage_root/repo/src"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

cd "$stage_root/repo"
exec "$python_bin" -m pls.oracles.fold_editflow \
    --manifest "$stage_root/benchmark/generated/pls_editflow_oracle_manifest_poc_v1.json" \
    --plan "$stage_root/benchmark/generated/pls_editflow_oracle_query_plan_poc_v1.json" \
    --output-root "$stage_root/artifacts/oracles/pls_editflow_poc_v1/esmfold" \
    --shard-index "$shard_index" \
    --cuda-device "$visible_device" \
    --chunk-size 64 \
    --num-recycles 3
