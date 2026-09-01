#!/usr/bin/env bash
#SBATCH --job-name=gb1_adaptive
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/gb1_adaptive_%j.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/gb1_adaptive_%j.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
visible_device=${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}
if [[ "$visible_device" == *,* ]]; then
    echo "Adaptive GB1 development requires exactly one Slurm-assigned GPU" >&2
    exit 2
fi

export PYTHONPATH="$stage_root/repo/src"
: "${PLS_GIT_REVISION:?Submit with the exact staged Git revision}"
export PLS_GIT_REVISION
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

cd "$stage_root/repo"
exec "$python_bin" script/run_experiment.py \
    --config configs/editflow/gb1_adaptive_path_ucb_star_development_v1.json \
    --outputs-root "$stage_root/outputs"
