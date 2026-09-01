#!/usr/bin/env bash
#SBATCH --job-name=pls_value_student
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/pls_student_%j.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/pls_student_%j.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
visible_device=${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}
if [[ "$visible_device" == *,* ]]; then
    echo "PLS student training requires exactly one Slurm-assigned GPU" >&2
    exit 2
fi
: "${PLS_GIT_REVISION:?Submit with the exact staged Git revision}"

export PYTHONPATH="$stage_root/repo/src:$stage_root/repo"
export PLS_GIT_REVISION
cd "$stage_root"
exec "$python_bin" "$stage_root/repo/script/run_experiment.py" \
    --config "$stage_root/repo/configs/editflow/pls_student_value_star_poc_v1.json" \
    --outputs-root "$stage_root/outputs"
