#!/usr/bin/env bash
set -euo pipefail
cd /media/PM983/Code/PLS
selection=artifacts/features/uesolds_selection
root=artifacts/plm/esm2_t33_650M_UR50D_residue_uesolds
wait_for_sessions() {
  while true; do
    active=0
    for shard in 0 1 2 3; do tmux has-session -t "$1$shard" 2>/dev/null && active=1; done
    (( active == 0 )) && return
    sleep 15
  done
}
wait_for_sessions pls_ues_extract_
PYTHONPATH=src conda run --no-capture-output -n BIO python preparation/audit_sharded_status.py --selection-status "$selection/status.npy" --status-root "$root" --prefix status_shard_ --shard-count 4
for spec in '0 4' '1 5' '2 6' '3 7'; do
  set -- $spec; shard=$1; hip=$2
  tmux new-session -d -s "pls_ues_project_$shard" "cd /media/PM983/Code/PLS && HIP_VISIBLE_DEVICES=$hip conda run --no-capture-output -n BIO python preparation/project_esm2_residue_pca.py --entities benchmark/generated/sequence_entities.csv --offsets $selection/offsets.npy --structure-status $selection/status.npy --source $root --pca $root/train_pca256.npz --output $root --shard-count 4 --shard-index $shard --hip-device $hip > outputs/uesolds_project_shard_$shard.log 2>&1"
done
wait_for_sessions pls_ues_project_
PYTHONPATH=src conda run --no-capture-output -n BIO python preparation/audit_sharded_status.py --selection-status "$selection/status.npy" --status-root "$root" --prefix pca_status_shard_ --shard-count 4
for spec in 'stats_s29 4 20260829 statistics_attention 0.99' 'stats_noema_s30 5 20260830 statistics_attention 0' 'attention_s31 6 20260831 attention 0.99' 'mean_s32 7 20260832 mean 0.99'; do
  set -- $spec; tag=$1; hip=$2; seed=$3; pooling=$4; ema=$5
  tmux new-session -d -s "pls_ues_train_$tag" "cd /media/PM983/Code/PLS && HIP_VISIBLE_DEVICES=$hip conda run --no-capture-output -n BIO python script/run_experiment.py --config configs/experiments/uesolds_residue_esm_v1.json --hip-override $hip --seed-override $seed --pooling-override $pooling --ema-decay-override $ema --name-suffix _$tag > outputs/uesolds_train_$tag.launch.log 2>&1"
done
