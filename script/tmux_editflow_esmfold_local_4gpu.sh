#!/usr/bin/env bash
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_poc_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_poc_v1.json"
plan="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_poc_v1.json"
output_root="$artifact_root/esmfold"
log_root="$artifact_root/logs"

mkdir -p "$output_root" "$log_root"
for device in 0 1 2 3; do
    session="pls_esmfold_g${device}"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "tmux session already exists: $session" >&2
        exit 2
    fi
done

for device in 0 1 2 3; do
    first=$device
    second=$((device + 4))
    session="pls_esmfold_g${device}"
    log="$log_root/local_esmfold_g${device}.log"
    command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' TORCH_HOME='/home/pc/.cache/torch' PYTHONPATH='$repo_root/src' && '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$output_root' --shard-index '$first' --hip-device '$device' --chunk-size 64 --num-recycles 3 && '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$output_root' --shard-index '$second' --hip-device '$device' --chunk-size 64 --num-recycles 3"
    tmux new-session -d -s "$session" "bash -lc \"$command >> '$log' 2>&1\""
    echo "$session -> shards $first,$second -> $log"
done
