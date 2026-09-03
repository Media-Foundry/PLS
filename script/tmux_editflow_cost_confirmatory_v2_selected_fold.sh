#!/usr/bin/env bash
# Fold ONLY the gated selected confirmatory mutants, on authorized GPUs 0-3.
#
# The plan is the preregistered selection committed before this stage. Never
# substitute the full confirmatory manifest plan here: the primary endpoint is
# measured selected-stage ESMFold compute, and folding unselected candidates now
# would collapse the numerator and the denominator into the same quantity.
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_cost_confirmatory_v2"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_cost_confirmatory_v2.json"
plan="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_cost_confirmatory_v2_selected.json"
selection="$repo_root/analysis/editflow_pls_cost_aware_confirmatory_v2_selection.json"
output_root="$artifact_root/exact_selected/esmfold"
log_root="$artifact_root/logs"

mode=${1:-dry-run}

export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

# The selection must already be committed: this stage is only valid after the
# preregistration exists in Git history.
"$python_bin" - <<PY
import json, subprocess, sys
from pathlib import Path
plan = json.loads(Path("$plan").read_text())
selection = json.loads(Path("$selection").read_text())
if plan.get("test_evaluated") is not False:
    sys.exit("fold plan is not test-free")
if plan.get("selection_scope") != "gated_selected_candidates_only":
    sys.exit("refusing: this plan is not restricted to the gated selection")
if plan["assignments_sha256"] != selection["fold_plan_assignments_sha256"]:
    sys.exit("fold plan does not match the persisted selection")
if len(plan["assignments"]) != selection["fold_plan_queries"]:
    sys.exit("fold plan size does not match the persisted selection")
tracked = subprocess.run(
    ["git", "ls-files", "--error-unmatch", "$plan"],
    capture_output=True, text=True,
)
if tracked.returncode != 0:
    sys.exit("refusing: the fold plan is not committed; preregister it first")
dirty = subprocess.run(
    ["git", "status", "--porcelain", "$plan", "$selection"],
    capture_output=True, text=True,
).stdout.strip()
if dirty:
    sys.exit(f"refusing: preregistration files have uncommitted changes:\n{dirty}")
print(json.dumps({
    "planned_selected_folds": len(plan["assignments"]),
    "shards": plan["shard_count"],
    "policy": plan["selection_policy_id"],
    "preregistration": "committed and clean",
}, indent=2))
PY

mkdir -p "$output_root" "$log_root"

if [[ "$mode" == "dry-run" ]]; then
    for device in 0 1 2 3; do
        echo "--- shard $device"
        "$python_bin" -m pls.oracles.fold_editflow \
            --manifest "$manifest" --plan "$plan" --output-root "$output_root" \
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
    session="pls_cost_conf_v2_selected_g${device}"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 2
    fi
done
for device in 0 1 2 3; do
    session="pls_cost_conf_v2_selected_g${device}"
    log="$log_root/esmfold_selected_g${device}.log"
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' TORCH_HOME='/home/pc/.cache/torch' PYTHONPATH='$repo_root/src:$repo_root' && '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$output_root' --shard-index '$device' --hip-device '$device' --chunk-size 64 --num-recycles 3 >> '$log' 2>&1"
    tmux new-session -d -s "$session" "bash -lc \"$command\""
    echo "$session -> shard $device -> $log"
done
date --iso-8601=seconds > "$log_root/selected_fold_started_at.txt"
