#!/usr/bin/env bash
# Exact V4 features and scoring for the split neighborhood campaign.
# Fold completeness was verified separately by verify_neighborhood_scale_folds.py,
# because this campaign ran under two plans and two report families.
set -euo pipefail

repo_root=/media/PM983/Code/PLS
python_bin=/home/pc/anaconda3/envs/BIO/bin/python
artifact_root="$repo_root/artifacts/oracles/pls_editflow_neighborhood_scale_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_neighborhood_scale_v1.json"
entities="$repo_root/benchmark/generated/pls_editflow_entities_neighborhood_scale_v1.csv"
exact="$artifact_root/exact_full"
fixed="$artifact_root/fixed_parent"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
score_config="$repo_root/configs/editflow/pls_oracle_score_neighborhood_scale_exact_v1.json"
log="$artifact_root/logs/scale_postfold.log"

cd "$repo_root"
export PYTHONPATH="$repo_root/src:$repo_root"
note() { echo "[$(date --iso-8601=seconds)] $*"; }

{
note "extracting exact V4 features for every neighborhood node"
numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$manifest" --pdb-root "$exact/esmfold" \
    --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$exact/structure_v4_raw" --source-root /home/pc/Code/BIO/protein --workers 192

note "building compact / geometry / vector / surface caches"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$entities" --raw-root "$exact/structure_v4_raw" \
    --status "$exact/structure_v4_raw/status.npy" --stats "$stats" \
    --output "$exact/structure_v4_compact"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
    --entities "$entities" --raw-root "$exact/structure_v4_raw" \
    --status "$exact/structure_v4_raw/status.npy" --compact-root "$exact/structure_v4_compact" \
    --output "$exact/structure_v4_geometry" --neighbors 16
numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
    --entities "$entities" --feature-root "$exact/structure_v4_raw" \
    --compact-dir "$exact/structure_v4_compact" --status "$exact/structure_v4_raw/status.npy" \
    --output "$exact/structure_v4_vectors" --workers 192
numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
    --entities "$entities" --compact "$exact/structure_v4_compact" \
    --geometry "$exact/structure_v4_geometry" --structure-stats "$stats" \
    --output "$exact/surface_patch_components"

note "confirming the PLM caches still index this entity order"
"$python_bin" - <<PY
import numpy as np, sys
a = np.load("$fixed/structure_v4_compact/offsets.npy")
b = np.load("$exact/structure_v4_compact/offsets.npy")
if a.shape != b.shape or not np.array_equal(a, b):
    sys.exit("offsets differ; reindex the PLM caches before scoring")
print("offsets identical: PLM caches apply unchanged")
PY

note "scoring the exact oracle over the whole campaign"
HIP_VISIBLE_DEVICES=1 "$python_bin" -m pls.oracles.score_editflow \
    --config "$score_config" --output-root "$exact/scores_fp32_a" \
    >> "$artifact_root/logs/score_exact.log" 2>&1
note "exact stage complete"
} 2>&1 | tee -a "$log"
