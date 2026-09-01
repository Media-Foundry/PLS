#!/usr/bin/env bash
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_1k_v1"
full_manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_1k_v1.json"
full_entities="$repo_root/benchmark/generated/pls_editflow_entities_1k_v1.csv"
cal_manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_1k_exact_calibration_v1.json"
cal_entities="$repo_root/benchmark/generated/pls_editflow_entities_1k_exact_calibration_v1.csv"
fixed="$artifact_root/fixed_parent"
exact="$artifact_root/exact_calibration"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
pca="$repo_root/artifacts/plm/esm2_t33_650M_UR50D_residue_pdbsol/train_pca_256.npz"
log="$artifact_root/logs/multifidelity_pipeline.log"

mkdir -p "$artifact_root/logs" "$exact"
exec >> "$log" 2>&1
export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

echo "[$(date --iso-8601=seconds)] waiting for four exact-calibration fold reports"
while true; do
    reports=0
    for shard in 0 1 2 3; do
        [[ -f "$artifact_root/esmfold/shard_$(printf '%03d' "$shard")_report.json" ]] && reports=$((reports + 1))
    done
    [[ $reports -eq 4 ]] && break
    if ! tmux list-sessions 2>/dev/null | grep -q 'pls_1k_calibration_g'; then
        echo "calibration folding exited before all reports were written" >&2
        exit 1
    fi
    sleep 20
done
"$python_bin" - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/oracles/pls_editflow_1k_v1/esmfold")
reports = [json.loads((root / f"shard_{i:03d}_report.json").read_text()) for i in range(4)]
if any(r["failed"] for r in reports) or sum(r["assigned"] for r in reports) != 3072:
    raise SystemExit("invalid exact-calibration folding reports")
print({"exact_new": sum(r["ok"] for r in reports), "exact_reused": sum(r["skipped"] for r in reports)})
PY
[[ -f "$fixed/surface_patch_components/metadata.json" ]] || { echo "fixed-parent cache incomplete" >&2; exit 1; }

echo "[$(date --iso-8601=seconds)] launching full exact-sequence mean ESM2 on GPU 3"
tmux new-session -d -s pls_1k_mean_esm2 \
    "cd '$repo_root' && HIP_VISIBLE_DEVICES=3 PYTHONPATH='$repo_root/src:$repo_root' '$python_bin' -m pls.features.extract_esm2 --entities '$full_entities' --output-dir '$artifact_root/esm2_mean' --token-budget 8192 --maximum-residues 300 --hip-device 3 --precision float16 >> '$artifact_root/logs/esm2_mean.log' 2>&1"

echo "[$(date --iso-8601=seconds)] extracting exact calibration V4 while mean ESM2 runs"
numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$cal_manifest" --pdb-root "$artifact_root/esmfold" \
    --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$exact/structure_v4_raw" --source-root /home/pc/Code/BIO/protein --workers 192
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$cal_entities" --raw-root "$exact/structure_v4_raw" \
    --status "$exact/structure_v4_raw/status.npy" --stats "$stats" \
    --output "$exact/structure_v4_compact"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
    --entities "$cal_entities" --raw-root "$exact/structure_v4_raw" \
    --status "$exact/structure_v4_raw/status.npy" --compact-root "$exact/structure_v4_compact" \
    --output "$exact/structure_v4_geometry" --neighbors 16
numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
    --entities "$cal_entities" --feature-root "$exact/structure_v4_raw" \
    --compact-dir "$exact/structure_v4_compact" --status "$exact/structure_v4_raw/status.npy" \
    --output "$exact/structure_v4_vectors" --workers 192
numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
    --entities "$cal_entities" --compact "$exact/structure_v4_compact" \
    --geometry "$exact/structure_v4_geometry" --structure-stats "$stats" \
    --output "$exact/surface_patch_components"

until ! tmux has-session -t pls_1k_mean_esm2 2>/dev/null; do sleep 20; done
"$python_bin" - <<'PY'
import numpy as np
s = np.load("artifacts/oracles/pls_editflow_1k_v1/esm2_mean/status.npy")
if s.shape != (19584,) or not np.all(s == 1):
    raise SystemExit("full mean ESM2 cache is incomplete")
PY

echo "[$(date --iso-8601=seconds)] extracting full exact-sequence residue ESM2/PCA"
"$python_bin" -m pls.features.extract_esm2_residue \
    --entities "$full_entities" --offsets "$fixed/structure_v4_compact/offsets.npy" \
    --structure-status "$fixed/structure_v4_raw/status.npy" \
    --output "$artifact_root/residue_esm2_raw" --shard-count 4 --initialize-only
"$python_bin" preparation/project_esm2_residue_pca.py \
    --entities "$full_entities" --offsets "$fixed/structure_v4_compact/offsets.npy" \
    --structure-status "$fixed/structure_v4_raw/status.npy" --source "$artifact_root/residue_esm2_raw" \
    --pca "$pca" --output "$artifact_root/residue_esm2_pca" --shard-count 4 --initialize-only
for device in 0 1 2 3; do
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' PYTHONPATH='$repo_root/src:$repo_root' && { '$python_bin' -m pls.features.extract_esm2_residue --entities '$full_entities' --offsets '$fixed/structure_v4_compact/offsets.npy' --structure-status '$fixed/structure_v4_raw/status.npy' --output '$artifact_root/residue_esm2_raw' --shard-count 4 --shard-index '$device' --hip-device '$device' --token-budget 8192 && '$python_bin' preparation/project_esm2_residue_pca.py --entities '$full_entities' --offsets '$fixed/structure_v4_compact/offsets.npy' --structure-status '$fixed/structure_v4_raw/status.npy' --source '$artifact_root/residue_esm2_raw' --pca '$pca' --output '$artifact_root/residue_esm2_pca' --shard-count 4 --shard-index '$device' --hip-device '$device' --residue-budget 65536; } >> '$artifact_root/logs/residue_g${device}.log' 2>&1"
    tmux new-session -d -s "pls_1k_residue_g${device}" "bash -lc \"$command\""
done
while tmux list-sessions 2>/dev/null | grep -q 'pls_1k_residue_g'; do sleep 20; done

echo "[$(date --iso-8601=seconds)] scoring dense fixed-parent oracle"
HIP_VISIBLE_DEVICES=2 "$python_bin" -m pls.oracles.score_editflow \
    --config configs/editflow/pls_oracle_score_fixed_parent_1k_v1.json \
    --output-root "$fixed/scores_fp32_a"

echo "[$(date --iso-8601=seconds)] reindexing PLM caches and scoring sparse exact oracle"
"$python_bin" preparation/reindex_editflow_plm_cache.py \
    --source-entities "$full_entities" --target-entities "$cal_entities" \
    --source-offsets "$fixed/structure_v4_compact/offsets.npy" \
    --target-offsets "$exact/structure_v4_compact/offsets.npy" \
    --source-mean "$artifact_root/esm2_mean" --source-residue-pca "$artifact_root/residue_esm2_pca" \
    --output-mean "$exact/esm2_mean" --output-residue-pca "$exact/residue_esm2_pca"
HIP_VISIBLE_DEVICES=2 "$python_bin" -m pls.oracles.score_editflow \
    --config configs/editflow/pls_oracle_score_exact_calibration_1k_v1.json \
    --output-root "$exact/scores_fp32_a"
echo "[$(date --iso-8601=seconds)] multifidelity oracle construction complete"
