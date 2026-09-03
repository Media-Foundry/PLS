#!/usr/bin/env bash
# Retrospective folds of the UNSELECTED confirmatory candidates.
#
# Valid only after the selected-stage deployment report is frozen in Git. These
# folds are not deployment cost: they exist to supply the measured exhaustive
# denominator and the coverage/regret audit. Keep their cost accounted separately.
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_cost_confirmatory_v2"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_cost_confirmatory_v2.json"
plan="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_cost_confirmatory_v2_full.json"
plan_report="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_report_cost_confirmatory_v2_full.json"
cost_report="$repo_root/analysis/editflow_pls_cost_aware_confirmatory_v2_selected_stage_cost.json"
selected="$artifact_root/exact_selected/esmfold"
full="$artifact_root/exact_full/esmfold"
log_root="$artifact_root/logs"

mode=${1:-dry-run}

export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

note() { echo "[$(date --iso-8601=seconds)] $*"; }

# The selected-stage freeze must already be committed and clean.
"$python_bin" - <<PY
import json, subprocess, sys
from pathlib import Path
report = Path("$cost_report")
if not report.is_file():
    sys.exit("refusing: the selected-stage cost report does not exist")
tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(report)],
                         capture_output=True, text=True)
if tracked.returncode != 0:
    sys.exit("refusing: the selected-stage cost report is not committed")
dirty = subprocess.run(["git", "status", "--porcelain", str(report)],
                       capture_output=True, text=True).stdout.strip()
if dirty:
    sys.exit(f"refusing: the selected-stage freeze is dirty:\n{dirty}")
data = json.loads(report.read_text())
print(json.dumps({
    "selected_stage_frozen": True,
    "selected_folds": data["selected_folds"],
    "unselected_pending": data["unselected_not_yet_folded"],
}, indent=2))
PY

mkdir -p "$full" "$log_root"

note "materializing already-folded selected structures into the full tree"
"$python_bin" - <<PY
import os
from pathlib import Path
source = Path("$selected")
destination = Path("$full")
destination.mkdir(parents=True, exist_ok=True)
linked = skipped = 0
for path in sorted(source.glob("*.ef.pdb")):
    target = destination / path.name
    if target.exists():
        skipped += 1
        continue
    os.link(path, target)
    linked += 1
print({"hardlinked": linked, "already_present": skipped})
PY

note "planning the exhaustive confirmatory fold set"
"$python_bin" preparation/plan_pls_editflow_oracle.py \
    --manifest "$manifest" --shards 4 --output "$plan" --report "$plan_report" \
    --runtime-cost-model "$repo_root/configs/editflow/pls_esmfold_runtime_cost_model_v1.json"

if [[ "$mode" == "dry-run" ]]; then
    for device in 0 1 2 3; do
        echo "--- shard $device"
        "$python_bin" -m pls.oracles.fold_editflow \
            --manifest "$manifest" --plan "$plan" --output-root "$full" \
            --shard-index "$device" --hip-device "$device" \
            --chunk-size 64 --num-recycles 3 --dry-run
    done
    exit 0
fi

if [[ "$mode" != "run" ]]; then
    echo "usage: $0 [dry-run|run]" >&2
    exit 1
fi

for device in 0 1 2 3; do
    session="pls_cost_conf_v2_retro_g${device}"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 2
    fi
done
for device in 0 1 2 3; do
    session="pls_cost_conf_v2_retro_g${device}"
    log="$log_root/esmfold_retrospective_g${device}.log"
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' TORCH_HOME='/home/pc/.cache/torch' PYTHONPATH='$repo_root/src:$repo_root' && '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$full' --shard-index '$device' --hip-device '$device' --chunk-size 64 --num-recycles 3 >> '$log' 2>&1"
    tmux new-session -d -s "$session" "bash -lc \"$command\""
    echo "$session -> shard $device -> $log"
done
date --iso-8601=seconds > "$log_root/retrospective_fold_started_at.txt"
