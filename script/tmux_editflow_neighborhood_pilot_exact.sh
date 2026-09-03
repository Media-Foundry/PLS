#!/usr/bin/env bash
# Exhaustive EXACT oracle over the whole single-mutant neighborhood.
# This is the gold standard: every one of the 10,735 mutants gets its own
# ESMFold structure, so the true discrete gradient field is measured, not
# approximated. Runs only on authorized GPUs 0-3.
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_neighborhood_pilot_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_neighborhood_pilot_v1.json"
entities="$repo_root/benchmark/generated/pls_editflow_entities_neighborhood_pilot_v1.csv"
plan="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_neighborhood_pilot_v1.json"
plan_report="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_report_neighborhood_pilot_v1.json"
exact="$artifact_root/exact_full"
fixed="$artifact_root/fixed_parent"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
score_config="$repo_root/configs/editflow/pls_oracle_score_neighborhood_pilot_exact_v1.json"
log_root="$artifact_root/logs"
mode=${1:-fold}

export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"
mkdir -p "$exact/esmfold" "$log_root"
note() { echo "[$(date --iso-8601=seconds)] $*"; }

fold_stage() {
    note "planning the exhaustive neighborhood fold set with LPT balancing"
    "$python_bin" preparation/plan_pls_editflow_oracle.py \
        --manifest "$manifest" --shards 4 --output "$plan" --report "$plan_report" \
        --runtime-cost-model "$repo_root/configs/editflow/pls_esmfold_runtime_cost_model_v1.json"
    for device in 0 1 2 3; do
        session="pls_nbhd_fold_g${device}"
        tmux has-session -t "$session" 2>/dev/null && { echo "session exists: $session" >&2; exit 2; }
    done
    for device in 0 1 2 3; do
        session="pls_nbhd_fold_g${device}"
        log="$log_root/esmfold_g${device}.log"
        command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' TORCH_HOME='/home/pc/.cache/torch' PYTHONPATH='$repo_root/src:$repo_root' && '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$exact/esmfold' --shard-index '$device' --hip-device '$device' --chunk-size 64 --num-recycles 3 >> '$log' 2>&1"
        tmux new-session -d -s "$session" "bash -lc \"$command\""
        echo "$session -> shard $device -> $log"
    done
    date --iso-8601=seconds > "$log_root/neighborhood_fold_started_at.txt"
}

postfold_stage() {
    note "verifying the exhaustive fold set is complete"
    "$python_bin" - <<PY
import json, sys
from pathlib import Path
plan = json.loads(Path("$plan").read_text())
reports = []
for shard in range(plan["shard_count"]):
    path = Path("$exact/esmfold") / f"shard_{shard:03d}_report.json"
    if not path.is_file():
        sys.exit(f"missing shard report: {path}")
    reports.append(json.loads(path.read_text()))
assigned = sum(int(r["assigned"]) for r in reports)
failed = sum(int(r["failed"]) for r in reports)
if assigned != len(plan["assignments"]) or failed:
    sys.exit(f"incomplete: assigned={assigned}, failed={failed}")
seconds = sum(float(row["seconds"]) for r in reports for row in r["results"] if row["status"] == "ok")
print(json.dumps({"folds": assigned, "failed": failed,
                  "measured_gpu_seconds": round(seconds, 1)}, indent=2))
PY
    note "extracting exact V4 features for every neighborhood node"
    numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
        --manifest "$manifest" --pdb-root "$exact/esmfold" \
        --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
        --output-root "$exact/structure_v4_raw" --source-root /home/pc/Code/BIO/protein --workers 192
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

    note "scoring the exact oracle over the whole neighborhood"
    HIP_VISIBLE_DEVICES=1 "$python_bin" -m pls.oracles.score_editflow \
        --config "$score_config" --output-root "$exact/scores_fp32_a" \
        >> "$log_root/score_exact.log" 2>&1
    note "exhaustive exact stage complete"
}

{
case "$mode" in
    fold) fold_stage ;;
    postfold) postfold_stage ;;
    *) echo "usage: $0 [fold|postfold]" >&2; exit 1 ;;
esac
} 2>&1 | tee -a "$log_root/exact_build.log"
