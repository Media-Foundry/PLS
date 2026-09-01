#!/usr/bin/env bash
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_decision_confirmatory_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_decision_confirmatory_v1.json"
entities="$repo_root/benchmark/generated/pls_editflow_entities_decision_confirmatory_v1.csv"
fixed="$artifact_root/fixed_parent"
exact="$artifact_root/exact"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
pca="$repo_root/artifacts/plm/esm2_t33_650M_UR50D_residue_pdbsol/train_pca_256.npz"
log="$artifact_root/logs/postfold.log"

mkdir -p "$artifact_root/logs" "$exact"
exec >> "$log" 2>&1
export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

echo "[$(date --iso-8601=seconds)] waiting for four confirmatory fold reports"
while true; do
    reports=0
    for shard in 0 1 2 3; do
        [[ -f "$exact/esmfold/shard_$(printf '%03d' "$shard")_report.json" ]] && reports=$((reports + 1))
    done
    [[ $reports -eq 4 ]] && break
    if ! tmux list-sessions 2>/dev/null | grep -q 'pls_decision_confirmatory_g'; then
        echo "folding exited before all reports were written" >&2
        exit 1
    fi
    sleep 20
done
"$python_bin" - <<'PY'
import json
from pathlib import Path
root = Path("artifacts/oracles/pls_editflow_decision_confirmatory_v1/exact/esmfold")
reports = [json.loads((root / f"shard_{i:03d}_report.json").read_text()) for i in range(4)]
if any(report["failed"] for report in reports) or sum(report["assigned"] for report in reports) != 1024:
    raise SystemExit("invalid confirmatory folding reports")
print({"new": sum(r["ok"] for r in reports), "cached": sum(r["skipped"] for r in reports)})
PY

echo "[$(date --iso-8601=seconds)] extracting exact V4"
numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$manifest" --pdb-root "$exact/esmfold" \
    --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$exact/structure_v4_raw" --source-root /home/pc/Code/BIO/protein --workers 192
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$entities" --raw-root "$exact/structure_v4_raw" \
    --status "$exact/structure_v4_raw/status.npy" --stats "$stats" --output "$exact/structure_v4_compact"
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

echo "[$(date --iso-8601=seconds)] extracting exact-sequence PLM caches"
HIP_VISIBLE_DEVICES=3 "$python_bin" -m pls.features.extract_esm2 \
    --entities "$entities" --output-dir "$artifact_root/esm2_mean" \
    --token-budget 8192 --maximum-residues 300 --hip-device 3 --precision float16
"$python_bin" -m pls.features.extract_esm2_residue \
    --entities "$entities" --offsets "$fixed/structure_v4_compact/offsets.npy" \
    --structure-status "$fixed/structure_v4_raw/status.npy" \
    --output "$artifact_root/residue_esm2_raw" --shard-count 4 --initialize-only
"$python_bin" preparation/project_esm2_residue_pca.py \
    --entities "$entities" --offsets "$fixed/structure_v4_compact/offsets.npy" \
    --structure-status "$fixed/structure_v4_raw/status.npy" --source "$artifact_root/residue_esm2_raw" \
    --pca "$pca" --output "$artifact_root/residue_esm2_pca" --shard-count 4 --initialize-only
for device in 0 1 2 3; do
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' PYTHONPATH='$repo_root/src:$repo_root' && { '$python_bin' -m pls.features.extract_esm2_residue --entities '$entities' --offsets '$fixed/structure_v4_compact/offsets.npy' --structure-status '$fixed/structure_v4_raw/status.npy' --output '$artifact_root/residue_esm2_raw' --shard-count 4 --shard-index '$device' --hip-device '$device' --token-budget 8192 && '$python_bin' preparation/project_esm2_residue_pca.py --entities '$entities' --offsets '$fixed/structure_v4_compact/offsets.npy' --structure-status '$fixed/structure_v4_raw/status.npy' --source '$artifact_root/residue_esm2_raw' --pca '$pca' --output '$artifact_root/residue_esm2_pca' --shard-count 4 --shard-index '$device' --hip-device '$device' --residue-budget 65536; } >> '$artifact_root/logs/residue_g${device}.log' 2>&1"
    tmux new-session -d -s "pls_decision_residue_g${device}" "bash -lc \"$command\""
done
while tmux list-sessions 2>/dev/null | grep -q 'pls_decision_residue_g'; do sleep 20; done

echo "[$(date --iso-8601=seconds)] scoring fixed and exact oracles"
HIP_VISIBLE_DEVICES=2 "$python_bin" -m pls.oracles.score_editflow \
    --config configs/editflow/pls_oracle_score_decision_confirmatory_fixed_v1.json \
    --output-root "$fixed/scores_fp32_a"
HIP_VISIBLE_DEVICES=2 "$python_bin" -m pls.oracles.score_editflow \
    --config configs/editflow/pls_oracle_score_decision_confirmatory_exact_v1.json \
    --output-root "$exact/scores_fp32_a"

run_dir="outputs/editflow_pls_decision_gating_confirmatory_v1+$(date +%m-%d-%H-%M)"
"$python_bin" -m pls.training.evaluate_editflow_pls_decision_confirmatory \
    --config configs/editflow/pls_decision_gating_confirmatory_v1.json \
    --run-dir "$run_dir" \
    --analysis-json analysis/editflow_pls_decision_gating_confirmatory_v1.json \
    --analysis-md analysis/editflow_pls_decision_gating_confirmatory_v1.md

echo "[$(date --iso-8601=seconds)] confirmatory oracle construction complete"
