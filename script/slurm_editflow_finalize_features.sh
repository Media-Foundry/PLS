#!/usr/bin/env bash
#SBATCH --job-name=pls_feature_finalize
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/husrcf/PLS/editflow_oracle_20260901/logs/feature_finalize_%j.out
#SBATCH --error=/data/husrcf/PLS/editflow_oracle_20260901/logs/feature_finalize_%j.err

set -euo pipefail

stage_root=${PLS_STAGE_ROOT:-/data/husrcf/PLS/editflow_oracle_20260901}
python_bin=${PLS_PYTHON:-/home/husrcf/anaconda3/envs/gpsite_full/bin/python}
artifact_root="$stage_root/artifacts/oracles/pls_editflow_poc_v1"
entities="$stage_root/benchmark/generated/pls_editflow_entities_poc_v1.csv"
raw_root="$artifact_root/structure_v4_raw"
status="$raw_root/status.npy"
stats="$stage_root/artifacts/features/pdbsol_structure_v4_train_stats.json"

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 50331648 )); then
    echo "Refusing feature finalization with less than 48 GiB host memory available" >&2
    exit 3
fi

export PYTHONPATH="$stage_root/repo/src:$stage_root/repo"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$stage_root/repo"

"$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$stage_root/benchmark/generated/pls_editflow_oracle_manifest_poc_v1.json" \
    --pdb-root "$artifact_root/esmfold" \
    --parent-raw-root "$stage_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$raw_root" \
    --source-root "$stage_root/external_v4" \
    --workers 32

if [[ ! -f "$artifact_root/structure_v4_compact/metadata.json" ]]; then
    "$python_bin" preparation/build_structure_v4_compact.py \
        --entities "$entities" --raw-root "$raw_root" --status "$status" \
        --stats "$stats" --output "$artifact_root/structure_v4_compact"
fi

if [[ ! -f "$artifact_root/structure_v4_geometry/metadata.json" ]]; then
    "$python_bin" preparation/build_structure_v4_geometry.py \
        --entities "$entities" --raw-root "$raw_root" --status "$status" \
        --compact-root "$artifact_root/structure_v4_compact" \
        --output "$artifact_root/structure_v4_geometry" --neighbors 16
fi

if [[ ! -f "$artifact_root/structure_v4_vectors/metadata.json" ]]; then
    "$python_bin" preparation/build_structure_v4_vectors.py \
        --entities "$entities" --feature-root "$raw_root" \
        --compact-dir "$artifact_root/structure_v4_compact" --status "$status" \
        --output "$artifact_root/structure_v4_vectors" --workers 32
fi

if [[ ! -f "$artifact_root/surface_patch_components/metadata.json" ]]; then
    "$python_bin" preparation/build_surface_patch_components.py \
        --entities "$entities" --compact "$artifact_root/structure_v4_compact" \
        --geometry "$artifact_root/structure_v4_geometry" \
        --structure-stats "$stats" \
        --output "$artifact_root/surface_patch_components"
fi
