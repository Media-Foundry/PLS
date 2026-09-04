# Complete-neighborhood campaign under the frozen analysis protocol

- anchors: 8
- candidates: 10,735
- protocol: `configs/editflow/pls_neighborhood_scale_analysis_protocol_v1.json` (frozen_before_any_scale_exact_fold)
- test sequences queried: 0

## Policies at matched exact budget, judged on the loss distribution

| Budget | Policy | Mean regret | CVaR95 | Max | P(R=0) | Exact-best recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `cached_parent` | 0.3341 | 1.4481 | 1.4481 | 0.625 | 0.6250 |
| 1 | `sequence_only` | 1.8695 | 6.1860 | 6.1860 | 0.000 | 0.0000 |
| 1 | `one_backward_gradient` | 1.8905 | 6.4331 | 6.4331 | 0.000 | 0.0000 |
| 1 | `random_control` | 1.9698 | 6.4850 | 6.8212 | 0.000 | 0.0000 |
| 2 | `cached_parent` | 0.2820 | 1.4481 | 1.4481 | 0.750 | 0.7500 |
| 2 | `sequence_only` | 1.6858 | 6.0904 | 6.0904 | 0.000 | 0.0000 |
| 2 | `one_backward_gradient` | 1.8516 | 6.4331 | 6.4331 | 0.000 | 0.0000 |
| 2 | `hybrid_union_m1` | 0.3341 | 1.4481 | 1.4481 | 0.625 | 0.6250 |
| 2 | `random_control` | 1.6773 | 6.3432 | 6.4544 | 0.006 | 0.0063 |
| 4 | `cached_parent` | 0.2162 | 0.9215 | 0.9215 | 0.750 | 0.7500 |
| 4 | `sequence_only` | 1.3057 | 5.9254 | 5.9254 | 0.000 | 0.0000 |
| 4 | `one_backward_gradient` | 1.5576 | 6.1653 | 6.1653 | 0.000 | 0.0000 |
| 4 | `hybrid_union_m1` | 0.2162 | 0.9215 | 0.9215 | 0.750 | 0.7500 |
| 4 | `hybrid_union_m2` | 0.1643 | 0.8082 | 0.8082 | 0.750 | 0.7500 |
| 4 | `random_control` | 1.5320 | 6.2126 | 6.3919 | 0.000 | 0.0000 |
| 8 | `cached_parent` | 0.0875 | 0.6997 | 0.6997 | 0.875 | 0.8750 |
| 8 | `sequence_only` | 1.1209 | 5.8305 | 5.8305 | 0.125 | 0.1250 |
| 8 | `one_backward_gradient` | 1.4386 | 5.7867 | 5.7867 | 0.000 | 0.0000 |
| 8 | `hybrid_union_m1` | 0.2162 | 0.9215 | 0.9215 | 0.750 | 0.7500 |
| 8 | `hybrid_union_m2` | 0.1643 | 0.8082 | 0.8082 | 0.750 | 0.7500 |
| 8 | `hybrid_union_m4` | 0.1188 | 0.8082 | 0.8082 | 0.750 | 0.7500 |
| 8 | `random_control` | 1.4262 | 6.1408 | 6.2303 | 0.013 | 0.0125 |
| 16 | `cached_parent` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 16 | `sequence_only` | 1.1172 | 5.8305 | 5.8305 | 0.125 | 0.1250 |
| 16 | `one_backward_gradient` | 1.3212 | 5.7867 | 5.7867 | 0.000 | 0.0000 |
| 16 | `hybrid_union_m1` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 16 | `hybrid_union_m2` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 16 | `hybrid_union_m4` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 16 | `hybrid_union_m8` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 16 | `random_control` | 1.2887 | 6.0370 | 6.1576 | 0.019 | 0.0187 |
| 32 | `cached_parent` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 32 | `sequence_only` | 0.9962 | 5.8305 | 5.8305 | 0.250 | 0.2500 |
| 32 | `one_backward_gradient` | 1.2267 | 5.7867 | 5.7867 | 0.000 | 0.0000 |
| 32 | `hybrid_union_m1` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 32 | `hybrid_union_m2` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 32 | `hybrid_union_m4` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 32 | `hybrid_union_m8` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 32 | `random_control` | 1.1268 | 5.9593 | 6.0247 | 0.044 | 0.0437 |
| 64 | `cached_parent` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 64 | `sequence_only` | 0.9257 | 5.5271 | 5.5271 | 0.250 | 0.2500 |
| 64 | `one_backward_gradient` | 1.0032 | 5.7867 | 5.7867 | 0.125 | 0.1250 |
| 64 | `hybrid_union_m1` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 64 | `hybrid_union_m2` | 0.0427 | 0.3417 | 0.3417 | 0.875 | 0.8750 |
| 64 | `hybrid_union_m4` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 64 | `hybrid_union_m8` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 64 | `random_control` | 1.0117 | 5.8815 | 5.9449 | 0.031 | 0.0312 |
| 128 | `cached_parent` | 0.0300 | 0.2399 | 0.2399 | 0.875 | 0.8750 |
| 128 | `sequence_only` | 0.2870 | 0.8728 | 0.8728 | 0.250 | 0.2500 |
| 128 | `one_backward_gradient` | 0.8635 | 5.7867 | 5.7867 | 0.250 | 0.2500 |
| 128 | `hybrid_union_m1` | 0.0300 | 0.2399 | 0.2399 | 0.875 | 0.8750 |
| 128 | `hybrid_union_m2` | 0.0300 | 0.2399 | 0.2399 | 0.875 | 0.8750 |
| 128 | `hybrid_union_m4` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 128 | `hybrid_union_m8` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 128 | `random_control` | 0.8071 | 5.6127 | 5.9276 | 0.100 | 0.1000 |
| 256 | `cached_parent` | 0.0215 | 0.1717 | 0.1717 | 0.875 | 0.8750 |
| 256 | `sequence_only` | 0.1520 | 0.5314 | 0.5314 | 0.500 | 0.5000 |
| 256 | `one_backward_gradient` | 0.7060 | 5.3914 | 5.3914 | 0.375 | 0.3750 |
| 256 | `hybrid_union_m1` | 0.0215 | 0.1717 | 0.1717 | 0.875 | 0.8750 |
| 256 | `hybrid_union_m2` | 0.0215 | 0.1717 | 0.1717 | 0.875 | 0.8750 |
| 256 | `hybrid_union_m4` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 256 | `hybrid_union_m8` | 0.0178 | 0.1421 | 0.1421 | 0.875 | 0.8750 |
| 256 | `random_control` | 0.4541 | 3.5872 | 5.3915 | 0.244 | 0.2437 |

## Exploratory failure detection

ROC AUC for predicting the catastrophic event from a label-free signal.
`roc_auc_negated` is the same signal with the sign flipped, which settles
which direction of a signal, if any, carries the information.

| Budget | Event | Signal | Positives | ROC AUC | Negated | Avg precision |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2 | `rank_gt_K` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 2 | `regret_gt_0.02` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 2 | `regret_gt_0.05` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 2 | `regret_gt_0.1` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 2 | `regret_gt_0.2` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 4 | `rank_gt_K` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 4 | `regret_gt_0.02` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 4 | `regret_gt_0.05` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 4 | `regret_gt_0.1` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 4 | `regret_gt_0.2` | `top32_dispersion` | 2/8 | 0.000 | 1.000 | 0.196 |
| 8 | `rank_gt_K` | `effective_support_tau_mad` | 1/8 | 1.000 | 0.000 | 1.000 |
| 8 | `rank_gt_K` | `top32_dispersion` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `rank_gt_K` | `top8_dispersion` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `rank_gt_K` | `top_gap_struct_minus_seq` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `rank_gt_K` | `top_margin` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `rank_gt_K` | `top_margin_over_mad` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.02` | `effective_support_tau_mad` | 1/8 | 1.000 | 0.000 | 1.000 |
| 8 | `regret_gt_0.02` | `top32_dispersion` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.02` | `top8_dispersion` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.02` | `top_gap_struct_minus_seq` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.02` | `top_margin` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.02` | `top_margin_over_mad` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.05` | `effective_support_tau_mad` | 1/8 | 1.000 | 0.000 | 1.000 |
| 8 | `regret_gt_0.05` | `top32_dispersion` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.05` | `top8_dispersion` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.05` | `top_gap_struct_minus_seq` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.05` | `top_margin` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.05` | `top_margin_over_mad` | 1/8 | 0.000 | 1.000 | 0.125 |
| 8 | `regret_gt_0.1` | `effective_support_tau_mad` | 1/8 | 1.000 | 0.000 | 1.000 |
| 8 | `regret_gt_0.1` | `top32_dispersion` | 1/8 | 0.000 | 1.000 | 0.125 |

Detectors here are exploratory and require a fresh confirmatory batch.
