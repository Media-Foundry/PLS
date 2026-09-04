#!/usr/bin/env bash
# Cached-parent oracle over the EXHAUSTIVE single-mutant neighborhood.
# Zero mutant folds: every mutant reuses its anchor's structure. This stage
# produces the low-fidelity field L over all 10,735 single mutants.
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
# Run name selects the manifest, entities, artifacts and configs.
run=${PLS_NEIGHBORHOOD_RUN:-pls_editflow_neighborhood_pilot_v1}
entities_name="pls_editflow_entities_${run#pls_editflow_}"
tag=${PLS_NEIGHBORHOOD_TAG:-pilot}
artifact_root="$repo_root/artifacts/oracles/${run}"
manifest="$repo_root/benchmark/generated/${run}.json"
entities="$repo_root/benchmark/generated/${entities_name}.csv"
parent_raw="$repo_root/artifacts/features/pdbsol_structure_v4_raw"
fixed="$artifact_root/fixed_parent"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
pca="$repo_root/artifacts/plm/esm2_t33_650M_UR50D_residue_pdbsol/train_pca_256.npz"
score_config="$repo_root/configs/editflow/pls_oracle_score_neighborhood_${tag}_fixed_v1.json"
log="$artifact_root/logs/cached_build.log"
nodes=${PLS_NEIGHBORHOOD_NODES:?set the node count}

mkdir -p "$artifact_root/logs" "$fixed"
export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"
note() { echo "[$(date --iso-8601=seconds)] $*"; }

{
note "materializing fixed-parent V4 tensors, zero mutant folds required"
"$python_bin" preparation/materialize_fixed_parent_structure.py \
    --manifest "$manifest" --exact-raw-root "$parent_raw" \
    --output-root "$fixed/structure_v4_raw"

note "building compact / geometry / vector / surface caches"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$entities" --raw-root "$fixed/structure_v4_raw" \
    --status "$fixed/structure_v4_raw/status.npy" --stats "$stats" \
    --output "$fixed/structure_v4_compact"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
    --entities "$entities" --raw-root "$fixed/structure_v4_raw" \
    --status "$fixed/structure_v4_raw/status.npy" --compact-root "$fixed/structure_v4_compact" \
    --output "$fixed/structure_v4_geometry" --neighbors 16
numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
    --entities "$entities" --feature-root "$fixed/structure_v4_raw" \
    --compact-dir "$fixed/structure_v4_compact" --status "$fixed/structure_v4_raw/status.npy" \
    --output "$fixed/structure_v4_vectors" --workers 192
numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
    --entities "$entities" --compact "$fixed/structure_v4_compact" \
    --geometry "$fixed/structure_v4_geometry" --structure-stats "$stats" \
    --output "$fixed/surface_patch_components"

note "extracting mean ESM2 on authorized GPU 0"
HIP_VISIBLE_DEVICES=0 "$python_bin" -m pls.features.extract_esm2 \
    --entities "$entities" --output-dir "$artifact_root/esm2_mean" \
    --token-budget 8192 --maximum-residues 300 --hip-device 0 --precision float16 \
    >> "$artifact_root/logs/esm2_mean.log" 2>&1
"$python_bin" - <<PY
import numpy as np
status = np.load("$artifact_root/esm2_mean/status.npy")
if status.shape != ($nodes,) or not np.all(status == 1):
    raise SystemExit("mean ESM2 cache is incomplete")
print("mean ESM2 cache complete: $nodes/$nodes")
PY

note "initializing sharded residue ESM2 and PCA caches"
"$python_bin" -m pls.features.extract_esm2_residue \
    --entities "$entities" --offsets "$fixed/structure_v4_compact/offsets.npy" \
    --structure-status "$fixed/structure_v4_raw/status.npy" \
    --output "$artifact_root/residue_esm2_raw" --shard-count 4 --initialize-only
"$python_bin" preparation/project_esm2_residue_pca.py \
    --entities "$entities" --offsets "$fixed/structure_v4_compact/offsets.npy" \
    --structure-status "$fixed/structure_v4_raw/status.npy" \
    --source "$artifact_root/residue_esm2_raw" --pca "$pca" \
    --output "$artifact_root/residue_esm2_pca" --shard-count 4 --initialize-only

note "running residue ESM2 and PCA across authorized GPUs 0-3"
for device in 0 1 2 3; do
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' PYTHONPATH='$repo_root/src:$repo_root' && { '$python_bin' -m pls.features.extract_esm2_residue --entities '$entities' --offsets '$fixed/structure_v4_compact/offsets.npy' --structure-status '$fixed/structure_v4_raw/status.npy' --output '$artifact_root/residue_esm2_raw' --shard-count 4 --shard-index '$device' --hip-device '$device' --token-budget 8192 && '$python_bin' preparation/project_esm2_residue_pca.py --entities '$entities' --offsets '$fixed/structure_v4_compact/offsets.npy' --structure-status '$fixed/structure_v4_raw/status.npy' --source '$artifact_root/residue_esm2_raw' --pca '$pca' --output '$artifact_root/residue_esm2_pca' --shard-count 4 --shard-index '$device' --hip-device '$device' --residue-budget 65536; } >> '$artifact_root/logs/residue_g${device}.log' 2>&1"
    tmux new-session -d -s "pls_nbhd_${tag}_residue_g${device}" "bash -lc \"$command\""
done
while tmux list-sessions 2>/dev/null | grep -q 'pls_nbhd_${tag}_residue_g'; do sleep 20; done

note "scoring the cached-parent oracle over the whole neighborhood"
HIP_VISIBLE_DEVICES=1 "$python_bin" -m pls.oracles.score_editflow \
    --config "$score_config" --output-root "$fixed/scores_fp32_a" \
    >> "$artifact_root/logs/score_fixed.log" 2>&1
note "cached-parent stage complete; zero mutants folded"
} 2>&1 | tee -a "$log"
