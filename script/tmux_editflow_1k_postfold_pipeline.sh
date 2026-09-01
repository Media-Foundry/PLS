#!/usr/bin/env bash
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_1k_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_1k_v1.json"
entities="$repo_root/benchmark/generated/pls_editflow_entities_1k_v1.csv"
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

echo "[$(date --iso-8601=seconds)] waiting for four successful 1k ESMFold shard reports"
while true; do
    reports=0
    for shard in 0 1 2 3; do
        [[ -f "$artifact_root/esmfold/shard_$(printf '%03d' "$shard")_report.json" ]] && reports=$((reports + 1))
    done
    [[ $reports -eq 4 ]] && break
    if ! tmux list-sessions 2>/dev/null | grep -q 'pls_1k_esmfold_g'; then
        echo "ESMFold sessions exited before all shard reports were written" >&2
        exit 1
    fi
    sleep 30
done
"$python_bin" - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/oracles/pls_editflow_1k_v1/esmfold")
reports = [json.loads((root / f"shard_{index:03d}_report.json").read_text()) for index in range(4)]
if any(report["failed"] for report in reports):
    raise SystemExit("at least one ESMFold shard contains failed queries")
print({"fold_ok": sum(r["ok"] for r in reports), "fold_reused": sum(r["skipped"] for r in reports)})
PY

echo "[$(date --iso-8601=seconds)] folds complete; launching mean ESM2 and exact V4 extraction"
tmux new-session -d -s pls_1k_mean_esm2 \
    "cd '$repo_root' && HIP_VISIBLE_DEVICES=3 PYTHONPATH='$repo_root/src:$repo_root' '$python_bin' -m pls.features.extract_esm2 --entities '$entities' --output-dir '$artifact_root/esm2_mean' --token-budget 8192 --maximum-residues 300 --hip-device 3 --precision float16 >> '$artifact_root/logs/esm2_mean.log' 2>&1"

numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$manifest" --pdb-root "$artifact_root/esmfold" \
    --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$raw_root" --source-root /home/pc/Code/BIO/protein --workers 192

echo "[$(date --iso-8601=seconds)] building compact/GVP/surface caches"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$entities" --raw-root "$raw_root" --status "$status" \
    --stats "$stats" --output "$compact"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
    --entities "$entities" --raw-root "$raw_root" --status "$status" \
    --compact-root "$compact" --output "$geometry" --neighbors 16
numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
    --entities "$entities" --feature-root "$raw_root" --compact-dir "$compact" \
    --status "$status" --output "$vectors" --workers 192
numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
    --entities "$entities" --compact "$compact" --geometry "$geometry" \
    --structure-stats "$stats" --output "$surface"

until ! tmux has-session -t pls_1k_mean_esm2 2>/dev/null; do sleep 20; done
"$python_bin" - <<'PY'
import numpy as np
status = np.load("artifacts/oracles/pls_editflow_1k_v1/esm2_mean/status.npy")
if status.shape != (19584,) or not np.all(status == 1):
    raise SystemExit("mean ESM2 cache is incomplete")
print({"mean_esm2_complete": int(status.sum())})
PY

echo "[$(date --iso-8601=seconds)] extracting exact residue ESM2/PCA on GPUs 0--3"
"$python_bin" -m pls.features.extract_esm2_residue \
    --entities "$entities" --offsets "$compact/offsets.npy" --structure-status "$status" \
    --output "$artifact_root/residue_esm2_raw" --shard-count 4 --initialize-only
"$python_bin" preparation/project_esm2_residue_pca.py \
    --entities "$entities" --offsets "$compact/offsets.npy" --structure-status "$status" \
    --source "$artifact_root/residue_esm2_raw" --pca "$pca" \
    --output "$artifact_root/residue_esm2_pca" --shard-count 4 --initialize-only
for device in 0 1 2 3; do
    session="pls_1k_residue_g${device}"
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' PYTHONPATH='$repo_root/src:$repo_root' && { '$python_bin' -m pls.features.extract_esm2_residue --entities '$entities' --offsets '$compact/offsets.npy' --structure-status '$status' --output '$artifact_root/residue_esm2_raw' --shard-count 4 --shard-index '$device' --hip-device '$device' --token-budget 8192 && '$python_bin' preparation/project_esm2_residue_pca.py --entities '$entities' --offsets '$compact/offsets.npy' --structure-status '$status' --source '$artifact_root/residue_esm2_raw' --pca '$pca' --output '$artifact_root/residue_esm2_pca' --shard-count 4 --shard-index '$device' --hip-device '$device' --residue-budget 65536; } >> '$artifact_root/logs/residue_g${device}.log' 2>&1"
    tmux new-session -d -s "$session" "bash -lc \"$command\""
done
while tmux list-sessions 2>/dev/null | grep -q 'pls_1k_residue_g'; do sleep 20; done

echo "[$(date --iso-8601=seconds)] scoring canonical float32 full/matched-sequence oracle on GPU 2"
HIP_VISIBLE_DEVICES=2 "$python_bin" -m pls.oracles.score_editflow \
    --config configs/editflow/pls_oracle_score_matched_ablation_1k_v1.json \
    --output-root "$artifact_root/scores_matched_fp32_a"

echo "[$(date --iso-8601=seconds)] launching PLM intervention students on GPUs 0--3"
declare -a wave1=(
    "potential:0:configs/editflow/pls_student_plm_potential_1k_v1.json"
    "pair:1:configs/editflow/pls_student_plm_pair_delta_1k_v1.json"
    "residual:2:configs/editflow/pls_student_plm_structural_residual_1k_v1.json"
    "cycle:3:configs/editflow/pls_student_plm_cycle_delta_1k_v1.json"
)
for spec in "${wave1[@]}"; do
    IFS=: read -r name device config <<< "$spec"
    tmux new-session -d -s "pls_1k_student_wave1_${name}" \
        "cd '$repo_root' && '$python_bin' script/run_experiment.py --config '$config' --hip-override '$device' >> '$artifact_root/logs/student_${name}.log' 2>&1"
done
while tmux list-sessions 2>/dev/null | grep -q 'pls_1k_student_wave1_'; do sleep 20; done

echo "[$(date --iso-8601=seconds)] launching raw-potential and parent-delta controls"
declare -a wave2=(
    "raw:0:configs/editflow/pls_student_value_1k_v1.json"
    "direct:1:configs/editflow/pls_student_plm_direct_delta_1k_v1.json"
)
for spec in "${wave2[@]}"; do
    IFS=: read -r name device config <<< "$spec"
    tmux new-session -d -s "pls_1k_student_wave2_${name}" \
        "cd '$repo_root' && '$python_bin' script/run_experiment.py --config '$config' --hip-override '$device' >> '$artifact_root/logs/student_${name}.log' 2>&1"
done
while tmux list-sessions 2>/dev/null | grep -q 'pls_1k_student_wave2_'; do sleep 20; done
echo "[$(date --iso-8601=seconds)] 1k PLS intervention-distillation pipeline complete"
