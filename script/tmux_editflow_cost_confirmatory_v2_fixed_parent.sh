#!/usr/bin/env bash
# Build ONLY the cached-parent oracle for the fresh cost-aware confirmatory
# components. This stage must never fold a confirmatory mutant: the primary
# endpoint is measured selected-stage ESMFold compute, and folding before the
# gated selection is committed would destroy it.
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_cost_confirmatory_v2"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_cost_confirmatory_v2.json"
entities="$repo_root/benchmark/generated/pls_editflow_entities_cost_confirmatory_v2.csv"
parent_raw="$repo_root/artifacts/features/pdbsol_structure_v4_raw"
fixed="$artifact_root/fixed_parent"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
pca="$repo_root/artifacts/plm/esm2_t33_650M_UR50D_residue_pdbsol/train_pca_256.npz"
score_config="$repo_root/configs/editflow/pls_oracle_score_cost_confirmatory_fixed_v2.json"
log="$artifact_root/logs/fixed_parent_build.log"

stage=${1:-all}

mkdir -p "$artifact_root/logs" "$fixed"
export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

note() { echo "[$(date --iso-8601=seconds)] $*"; }

# Hard guard: this tree must never acquire an exact-fold subtree at this stage.
guard() {
    if [[ -e "$artifact_root/exact" || -e "$artifact_root/exact_selected" || -e "$artifact_root/exact_full" ]]; then
        echo "REFUSING: a confirmatory exact-fold tree already exists; the cached-parent" >&2
        echo "stage must run before any confirmatory mutant is folded." >&2
        exit 1
    fi
}

structure_stage() {
    guard
    note "materializing fixed-parent V4 tensors, zero mutant folds required"
    "$python_bin" preparation/materialize_fixed_parent_structure.py \
        --manifest "$manifest" \
        --exact-raw-root "$parent_raw" \
        --output-root "$fixed/structure_v4_raw"

    note "building compact structure cache"
    numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
        --entities "$entities" --raw-root "$fixed/structure_v4_raw" \
        --status "$fixed/structure_v4_raw/status.npy" --stats "$stats" \
        --output "$fixed/structure_v4_compact"

    note "building kNN-16 geometry"
    numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
        --entities "$entities" --raw-root "$fixed/structure_v4_raw" \
        --status "$fixed/structure_v4_raw/status.npy" --compact-root "$fixed/structure_v4_compact" \
        --output "$fixed/structure_v4_geometry" --neighbors 16

    note "building equivariant vector channels"
    numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
        --entities "$entities" --feature-root "$fixed/structure_v4_raw" \
        --compact-dir "$fixed/structure_v4_compact" --status "$fixed/structure_v4_raw/status.npy" \
        --output "$fixed/structure_v4_vectors" --workers 192

    note "building surface patch components"
    numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
        --entities "$entities" --compact "$fixed/structure_v4_compact" \
        --geometry "$fixed/structure_v4_geometry" --structure-stats "$stats" \
        --output "$fixed/surface_patch_components"
    note "structure stage complete"
}

plm_stage() {
    guard
    note "extracting exact-sequence mean ESM2 on authorized GPU 0"
    HIP_VISIBLE_DEVICES=0 "$python_bin" -m pls.features.extract_esm2 \
        --entities "$entities" --output-dir "$artifact_root/esm2_mean" \
        --token-budget 8192 --maximum-residues 300 --hip-device 0 --precision float16 \
        >> "$artifact_root/logs/esm2_mean.log" 2>&1

    "$python_bin" - <<PY
import numpy as np
status = np.load("$artifact_root/esm2_mean/status.npy")
if status.shape != (2176,) or not np.all(status == 1):
    raise SystemExit("mean ESM2 cache is incomplete")
print("mean ESM2 cache complete: 2176/2176")
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
        tmux new-session -d -s "pls_conf_v2_residue_g${device}" "bash -lc \"$command\""
    done
    while tmux list-sessions 2>/dev/null | grep -q 'pls_conf_v2_residue_g'; do sleep 20; done
    note "PLM stage complete"
}

score_stage() {
    guard
    note "scoring the dense cached-parent oracle for every confirmatory candidate"
    HIP_VISIBLE_DEVICES=1 "$python_bin" -m pls.oracles.score_editflow \
        --config "$score_config" --output-root "$fixed/scores_fp32_a" \
        >> "$artifact_root/logs/score_fixed.log" 2>&1
    note "cached-parent scoring complete; no confirmatory mutant was folded"
}

{
case "$stage" in
    structure) structure_stage ;;
    plm) plm_stage ;;
    score) score_stage ;;
    all) structure_stage; plm_stage; score_stage ;;
    *) echo "unknown stage: $stage" >&2; exit 1 ;;
esac
} 2>&1 | tee -a "$log"
