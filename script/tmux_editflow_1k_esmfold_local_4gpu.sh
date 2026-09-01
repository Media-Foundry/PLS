#!/usr/bin/env bash
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_1k_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_1k_v1.json"
plan="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_1k_v1.json"
output_root="$artifact_root/esmfold"
log_root="$artifact_root/logs"

mkdir -p "$output_root" "$log_root"
for device in 0 1 2 3; do
    session="pls_1k_esmfold_g${device}"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 2
    fi
done

for device in 0 1 2 3; do
    session="pls_1k_esmfold_g${device}"
    log="$log_root/esmfold_g${device}.log"
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' TORCH_HOME='/home/pc/.cache/torch' PYTHONPATH='$repo_root/src' && '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$output_root' --shard-index '$device' --hip-device '$device' --chunk-size 64 --num-recycles 3 >> '$log' 2>&1"
    tmux new-session -d -s "$session" "bash -lc \"$command\""
    echo "$session -> shard $device -> $log"
done
