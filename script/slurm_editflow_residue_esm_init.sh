#!/usr/bin/env bash
#SBATCH --job-name=pls_residue_init
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/residue_init_%j.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/residue_init_%j.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
artifact_root="$stage_root/artifacts/oracles/pls_editflow_poc_v1"
entities="$stage_root/benchmark/generated/pls_editflow_entities_poc_v1.csv"
offsets="$artifact_root/structure_v4_compact/offsets.npy"
status="$artifact_root/structure_v4_raw/status.npy"
pca="$stage_root/artifacts/features/esm2_residue_train_pca_256.npz"

export PYTHONPATH="$stage_root/repo/src:$stage_root/repo"
cd "$stage_root/repo"

"$python_bin" -m pls.features.extract_esm2_residue \
    --entities "$entities" --offsets "$offsets" --structure-status "$status" \
    --output "$artifact_root/residue_esm2_raw" --shard-count 2 --initialize-only

"$python_bin" preparation/project_esm2_residue_pca.py \
    --entities "$entities" --offsets "$offsets" --structure-status "$status" \
    --source "$artifact_root/residue_esm2_raw" --pca "$pca" \
    --output "$artifact_root/residue_esm2_pca" --shard-count 2 --initialize-only
