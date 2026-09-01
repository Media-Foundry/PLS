#!/usr/bin/env bash
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_scale_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_scale_v1.json"
entities="$repo_root/benchmark/generated/pls_editflow_entities_scale_v1.csv"
raw_root="$artifact_root/structure_v4_raw"
compact="$artifact_root/structure_v4_compact"
geometry="$artifact_root/structure_v4_geometry"
vectors="$artifact_root/structure_v4_vectors"
surface="$artifact_root/surface_patch_components"
status="$raw_root/status.npy"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
pca="$repo_root/artifacts/plm/esm2_t33_650M_UR50D_residue_pdbsol/train_pca_256.npz"
log="$artifact_root/logs/postfold_pipeline.log"

mkdir -p "$artifact_root/logs"
exec >> "$log" 2>&1
export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

echo "[$(date --iso-8601=seconds)] waiting for 2560 exact-mutant structures"
until [[ $(find "$artifact_root/esmfold" -maxdepth 1 -name '*.ef.pdb' -type f | wc -l) -ge 2560 ]]; do
    sleep 30
done
echo "[$(date --iso-8601=seconds)] folds complete; extracting exact V4 features"
numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$manifest" --pdb-root "$artifact_root/esmfold" \
    --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$raw_root" --source-root /home/pc/Code/BIO/protein --workers 96

echo "[$(date --iso-8601=seconds)] building compact/GVP/surface caches"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$entities" --raw-root "$raw_root" --status "$status" \
    --stats "$stats" --output "$compact"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
    --entities "$entities" --raw-root "$raw_root" --status "$status" \
    --compact-root "$compact" --output "$geometry" --neighbors 16
numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
    --entities "$entities" --feature-root "$raw_root" --compact-dir "$compact" \
    --status "$status" --output "$vectors" --workers 96
numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
    --entities "$entities" --compact "$compact" --geometry "$geometry" \
    --structure-stats "$stats" --output "$surface"

echo "[$(date --iso-8601=seconds)] extracting exact residue ESM2 on GPUs 0 and 1"
"$python_bin" -m pls.features.extract_esm2_residue \
    --entities "$entities" --offsets "$compact/offsets.npy" --structure-status "$status" \
    --output "$artifact_root/residue_esm2_raw" --shard-count 2 --initialize-only
"$python_bin" preparation/project_esm2_residue_pca.py \
    --entities "$entities" --offsets "$compact/offsets.npy" --structure-status "$status" \
    --source "$artifact_root/residue_esm2_raw" --pca "$pca" \
    --output "$artifact_root/residue_esm2_pca" --shard-count 2 --initialize-only
for device in 0 1; do
    session="pls_scale_residue_g${device}"
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' PYTHONPATH='$repo_root/src:$repo_root' && { '$python_bin' -m pls.features.extract_esm2_residue --entities '$entities' --offsets '$compact/offsets.npy' --structure-status '$status' --output '$artifact_root/residue_esm2_raw' --shard-count 2 --shard-index '$device' --hip-device '$device' --token-budget 8192 && '$python_bin' preparation/project_esm2_residue_pca.py --entities '$entities' --offsets '$compact/offsets.npy' --structure-status '$status' --source '$artifact_root/residue_esm2_raw' --pca '$pca' --output '$artifact_root/residue_esm2_pca' --shard-count 2 --shard-index '$device' --hip-device '$device' --residue-budget 65536; } >> '$artifact_root/logs/residue_g${device}.log' 2>&1"
    tmux new-session -d -s "$session" "bash -lc \"$command\""
done
until ! tmux has-session -t pls_scale_residue_g0 2>/dev/null && ! tmux has-session -t pls_scale_residue_g1 2>/dev/null; do sleep 20; done

echo "[$(date --iso-8601=seconds)] scoring canonical float32 full/matched-sequence oracle on GPU 2"
HIP_VISIBLE_DEVICES=2 "$python_bin" -m pls.oracles.score_editflow \
    --config configs/editflow/pls_oracle_score_matched_ablation_scale_v1.json \
    --output-root "$artifact_root/scores_matched_fp32_a"

echo "[$(date --iso-8601=seconds)] launching four identical-node students on GPUs 0--3"
declare -a student_specs=(
    "plm_potential:0:configs/editflow/pls_student_plm_potential_scale_v1.json"
    "plm_direct:1:configs/editflow/pls_student_plm_direct_delta_scale_v1.json"
    "plm_cycle:2:configs/editflow/pls_student_plm_cycle_delta_scale_v1.json"
    "raw_potential:3:configs/editflow/pls_student_value_scale_v1.json"
)
for spec in "${student_specs[@]}"; do
    IFS=: read -r name device config <<< "$spec"
    session="pls_scale_student_${name}"
    tmux new-session -d -s "$session" \
        "cd '$repo_root' && '$python_bin' script/run_experiment.py --config '$config' --hip-override '$device' >> '$artifact_root/logs/student_${name}.log' 2>&1"
done
while tmux list-sessions 2>/dev/null | rg -q 'pls_scale_student_'; do sleep 20; done
echo "[$(date --iso-8601=seconds)] scaled PLS intervention-student pipeline complete"
