#!/usr/bin/env bash
#SBATCH --job-name=pls_oracle_score
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/oracle_score_%j.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/oracle_score_%j.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
visible_device=${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}
if [[ "$visible_device" == *,* ]]; then
    echo "Oracle scoring requires exactly one Slurm-assigned GPU" >&2
    exit 2
fi

export PYTHONPATH="$stage_root/repo/src:$stage_root/repo"
cd "$stage_root"
exec "$python_bin" -m pls.oracles.score_editflow \
    --config "$stage_root/repo/configs/editflow/pls_oracle_score_star_poc_v1.json" \
    --output-root "$stage_root/artifacts/oracles/pls_editflow_poc_v1/scores"
