# PLS sequence-student proof of concept

The comparison uses the same 272 train oracle nodes, 136 validation oracle
nodes, sequence-student architecture, seed, optimizer, and validation
`edge_rmse` selection rule. Only the graph-Sobolev edge-loss weight changes from
zero to one. The teacher is the canonical float32 full PLS oracle. Test use is
zero.

| Objective | Value Pearson | Value R2 | Edge RMSE | Edge Spearman | Sign accuracy | Top-5 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Value KD | **0.6447** | -0.1110 | 0.6686 | **0.1171** | **0.5391** | **0.3500** |
| Value + edge KD | 0.5984 | **-0.0855** | **0.6685** | 0.0777 | 0.5312 | 0.3250 |

Both checkpoints select epoch 1. Edge KD changes edge RMSE by only `-0.00016`
and is worse on rank, sign, and top-k metrics. This is a null result for naive
graph-Sobolev training on the PLS oracle, consistent with the GB1 same-node
control. The student can correlate with teacher values across held-out proteins
while remaining near chance on held-out mutation directions; the next method
must address local-field coverage or representation, not merely increase the
edge-loss coefficient.
