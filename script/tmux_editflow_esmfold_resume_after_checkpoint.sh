#!/usr/bin/env bash
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
checkpoint=/home/pc/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt
artifact_root="$repo_root/artifacts/oracles/pls_editflow_poc_v1"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_poc_v1.json"
plan="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_poc_v1.json"
output_root="$artifact_root/esmfold"
log_root="$artifact_root/logs"

launch_device() {
    local device=$1
    local first=$device
    local second=$((device + 4))
    local session="pls_esmfold_g${device}"
    local log="$log_root/local_esmfold_g${device}.log"
    if tmux has-session -t "$session" 2>/dev/null; then
        return
    fi
    local command="cd '$repo_root' && export HIP_VISIBLE_DEVICES='$device' TORCH_HOME='/home/pc/.cache/torch' PYTHONPATH='$repo_root/src' && { '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$output_root' --shard-index '$first' --hip-device '$device' --chunk-size 64 --num-recycles 3 && '$python_bin' -m pls.oracles.fold_editflow --manifest '$manifest' --plan '$plan' --output-root '$output_root' --shard-index '$second' --hip-device '$device' --chunk-size 64 --num-recycles 3; } >> '$log' 2>&1"
    tmux new-session -d -s "$session" "bash -lc \"$command\""
}

until [[ -s "$checkpoint" ]]; do sleep 20; done
until find "$output_root" -maxdepth 1 -name '*.ef.pdb' -type f -print -quit | grep -q .; do
    sleep 20
done

launch_device 2
launch_device 3
until ! tmux has-session -t editflow_adaptive_g1 2>/dev/null; do sleep 20; done
launch_device 1
