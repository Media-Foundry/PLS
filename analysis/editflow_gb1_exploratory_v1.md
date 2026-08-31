# GB1 EditFlow exploratory result v1

This report is an exploratory proof of concept, not a final GB1 benchmark claim.
It uses the measured Wu et al. GB1 fitness table, one wild-type anchor, one model
seed schedule, and repeated evaluation on the full measured landscape.  No PLS
test split was accessed.

## Static same-node-budget control

Value-KD and naive EditFlow used the same ordered list of 640 queried nodes
(the original, order-sensitive run identity is
`9d1b4ae06acd05090c405657006b1bcb58f6e7b8bf28ac6f2aca1d898f62f51c`)
and the same 752 closed edges.

| objective | measured-node R2 | edge Spearman | radius-4 regret |
| --- | ---: | ---: | ---: |
| Value-KD | 0.3334 | 0.3551 | 2.2619 |
| Value + naive edge loss | 0.3361 | 0.3359 | 2.2619 |

The naive edge objective does not improve this control.  This negative result is
consistent with treating path-aware acquisition, rather than the mere presence of
an edge loss, as the algorithmic hypothesis.

### Exact-node replay after active acquisition

The final queried-node manifests from both active runs were replayed with the
same five model seeds and either value-only or value-plus-edge training.  Thus,
within each pair, objective weighting is the only changed factor.

| queried-node set | objective | R2 | edge Spearman | regret k=1 / k=2 / k=3 / k=4 |
| --- | --- | ---: | ---: | --- |
| Path-OLD | Value-KD | 0.6012 | 0.3344 | 0.0100 / 0.0000 / 0.5029 / 1.8840 |
| Path-OLD | Value + edge | 0.5897 | 0.3341 | 0.0100 / 0.0000 / 0.5029 / 1.8840 |
| uncertainty | Value-KD | 0.7354 | 0.4058 | 0.0100 / 1.3643 / 0.5482 / 1.9044 |
| uncertainty | Value + edge | 0.7472 | 0.3880 | 0.0100 / 1.3643 / 0.5482 / 1.9044 |

The graph-Sobolev objective does not change any selected optimum in this replay.
The current positive signal therefore comes from the acquired node set and its
induced local coverage, not from claiming that edge differences contain new
oracle information.  Edge loss remains an ablation rather than the novelty claim.

## Equal-budget sequential acquisition

Path-aware and uncertainty-only acquisition start from the exact same 80-node
connected set (`909bd94104a47f6c316b001b6b6142f776359ed7843cce46b17064c816ee005c`).
Both train the same five-member edge-distilled ensemble and purchase exactly the
same number of unique measured-node oracle queries.

| queries | acquisition | R2 | edge Spearman | regret k=1 | k=2 | k=3 | k=4 | closed edges |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | shared initial set | -0.5617 | 0.1981 | 3.3158 | 0.6162 | 5.6021 | 5.4920 | 95 |
| 160 | Path-OLD | 0.3109 | 0.3556 | 0.0000 | 1.4893 | 2.3735 | 4.6699 | 631 |
| 160 | uncertainty | 0.1318 | 0.2587 | 3.3158 | 1.7822 | 4.0055 | 4.6699 | 434 |
| 320 | Path-OLD | 0.3938 | 0.2236 | 0.0100 | 0.0000 | 0.8842 | 3.3712 | 1,791 |
| 320 | uncertainty | 0.5377 | 0.3647 | 0.0100 | 1.4993 | 1.2114 | 3.1307 | 1,206 |
| 640 | Path-OLD | 0.5897 | 0.3341 | 0.0100 | 0.0000 | 0.5029 | 1.8840 | 3,905 |
| 640 | uncertainty | 0.7472 | 0.3880 | 0.0100 | 1.3643 | 0.5482 | 1.9044 | 3,094 |

At 640 queries, uncertainty-only has substantially better global prediction
fidelity, while Path-OLD has lower exact measured-fitness regret at radii 2, 3,
and 4 and ties at radius 1.  This is evidence for the paper's motivating
distinction between global value fidelity and local design quality, but one seed
and one anchor are insufficient for an inferential claim.

## Acquisition behavior

The path-aware rounds selected 60, 63, and 43 nodes directly by path occupancy
times ensemble edge uncertainty.  Strict one-hop uncertainty filling supplied the
remaining 20, 97, and 277 nodes.  Prospective nodes never became frontier sources
within the same acquisition round.

## Required confirmatory protocol

- Freeze anchor partitions, seeds, budgets, metrics, and hyperparameters before
  confirmatory evaluation.
- Repeat across anchors and seeds and report paired confidence intervals.
- Compare methods under equal unique-node query budgets and retain exact queried
  node manifests.
- Keep measured-only results primary; report the 10,639 imputed variants only as
  a separate sensitivity analysis.
- Do not use any PLS test split during method development or selection.

## Archived runs

- `outputs/editflow_gb1_path_acquisition_poc_v1_s831_r1+08-31-17-24`
- `outputs/editflow_gb1_uncertainty_acquisition_poc_v1_s831_r1+08-31-17-24`
- `outputs/editflow_gb1_value_kd_poc_v1_q640_s831_r1+08-31-17-06`
- `outputs/editflow_gb1_edge_poc_v1_q640_s831_r1+08-31-17-06`
- `outputs/editflow_gb1_path_nodes_value_replay_poc_v1_s831+08-31-17-32`
- `outputs/editflow_gb1_uncertainty_nodes_value_replay_poc_v1_s831+08-31-17-32`
