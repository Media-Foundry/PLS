# Complete-neighborhood campaign under the frozen analysis protocol

- anchors: 64
- candidates: 201,399
- protocol: `configs/editflow/pls_neighborhood_scale_analysis_protocol_v1.json` (frozen_before_any_scale_exact_fold)
- test sequences queried: 0

## Policies at matched exact budget, judged on the loss distribution

| Budget | Policy | Mean regret | CVaR95 | Max | P(R=0) | Exact-best recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `cached_parent` | 0.4003 | 1.7169 | 2.2047 | 0.406 | 0.4062 |
| 1 | `sequence_only` | 1.5141 | 6.7505 | 8.7763 | 0.109 | 0.1094 |
| 1 | `one_backward_gradient` | 2.1842 | 8.4502 | 9.2583 | 0.000 | 0.0000 |
| 1 | `random_control` | 2.2946 | 8.9590 | 9.4868 | 0.000 | 0.0000 |
| 2 | `cached_parent` | 0.2702 | 1.5951 | 2.2047 | 0.500 | 0.5000 |
| 2 | `sequence_only` | 1.3962 | 6.7505 | 8.7763 | 0.109 | 0.1094 |
| 2 | `one_backward_gradient` | 2.0018 | 8.4090 | 9.2522 | 0.016 | 0.0156 |
| 2 | `hybrid_union_m1` | 0.3535 | 1.6200 | 2.2047 | 0.406 | 0.4062 |
| 2 | `random_control` | 2.1351 | 8.6502 | 9.4156 | 0.001 | 0.0008 |
| 4 | `cached_parent` | 0.2057 | 1.1108 | 1.3577 | 0.531 | 0.5312 |
| 4 | `sequence_only` | 1.2508 | 6.6542 | 8.7763 | 0.156 | 0.1562 |
| 4 | `one_backward_gradient` | 1.7955 | 8.3710 | 9.2522 | 0.016 | 0.0156 |
| 4 | `hybrid_union_m1` | 0.2519 | 1.4600 | 2.2047 | 0.500 | 0.5000 |
| 4 | `hybrid_union_m2` | 0.2670 | 1.5951 | 2.2047 | 0.500 | 0.5000 |
| 4 | `random_control` | 2.0178 | 8.6914 | 9.3282 | 0.002 | 0.0023 |
| 8 | `cached_parent` | 0.1714 | 0.9498 | 1.3316 | 0.547 | 0.5469 |
| 8 | `sequence_only` | 1.1188 | 6.6542 | 8.7763 | 0.203 | 0.2031 |
| 8 | `one_backward_gradient` | 1.6754 | 8.3320 | 9.1312 | 0.016 | 0.0156 |
| 8 | `hybrid_union_m1` | 0.1733 | 0.9645 | 1.3316 | 0.547 | 0.5469 |
| 8 | `hybrid_union_m2` | 0.1936 | 1.0327 | 1.3316 | 0.531 | 0.5312 |
| 8 | `hybrid_union_m4` | 0.2014 | 1.1108 | 1.3577 | 0.531 | 0.5312 |
| 8 | `random_control` | 1.8671 | 8.2067 | 9.3235 | 0.002 | 0.0016 |
| 16 | `cached_parent` | 0.1446 | 0.9498 | 1.3316 | 0.609 | 0.6094 |
| 16 | `sequence_only` | 1.0259 | 6.6447 | 8.7763 | 0.234 | 0.2344 |
| 16 | `one_backward_gradient` | 1.4875 | 7.3502 | 8.8567 | 0.047 | 0.0469 |
| 16 | `hybrid_union_m1` | 0.1446 | 0.9498 | 1.3316 | 0.609 | 0.6094 |
| 16 | `hybrid_union_m2` | 0.1527 | 0.9498 | 1.3316 | 0.594 | 0.5938 |
| 16 | `hybrid_union_m4` | 0.1539 | 0.9498 | 1.3316 | 0.578 | 0.5781 |
| 16 | `hybrid_union_m8` | 0.1570 | 0.9498 | 1.3316 | 0.562 | 0.5625 |
| 16 | `random_control` | 1.7215 | 8.0357 | 9.2656 | 0.006 | 0.0063 |
| 32 | `cached_parent` | 0.1236 | 0.8683 | 1.3316 | 0.641 | 0.6406 |
| 32 | `sequence_only` | 0.8595 | 5.1524 | 6.5289 | 0.234 | 0.2344 |
| 32 | `one_backward_gradient` | 1.1721 | 5.6719 | 6.6122 | 0.094 | 0.0938 |
| 32 | `hybrid_union_m1` | 0.1236 | 0.8683 | 1.3316 | 0.641 | 0.6406 |
| 32 | `hybrid_union_m2` | 0.1236 | 0.8683 | 1.3316 | 0.641 | 0.6406 |
| 32 | `hybrid_union_m4` | 0.1224 | 0.8640 | 1.3316 | 0.625 | 0.6250 |
| 32 | `hybrid_union_m8` | 0.1169 | 0.8640 | 1.3316 | 0.625 | 0.6250 |
| 32 | `random_control` | 1.5747 | 7.5842 | 9.2737 | 0.012 | 0.0117 |
| 64 | `cached_parent` | 0.1075 | 0.7942 | 1.3316 | 0.672 | 0.6719 |
| 64 | `sequence_only` | 0.7756 | 5.0342 | 6.5289 | 0.281 | 0.2812 |
| 64 | `one_backward_gradient` | 1.0741 | 5.2200 | 6.6122 | 0.109 | 0.1094 |
| 64 | `hybrid_union_m1` | 0.1075 | 0.7942 | 1.3316 | 0.672 | 0.6719 |
| 64 | `hybrid_union_m2` | 0.1075 | 0.7942 | 1.3316 | 0.672 | 0.6719 |
| 64 | `hybrid_union_m4` | 0.1035 | 0.7931 | 1.3316 | 0.672 | 0.6719 |
| 64 | `hybrid_union_m8` | 0.1006 | 0.7931 | 1.3316 | 0.672 | 0.6719 |
| 64 | `random_control` | 1.3284 | 6.8732 | 9.1216 | 0.024 | 0.0242 |
| 128 | `cached_parent` | 0.0841 | 0.7418 | 1.2336 | 0.719 | 0.7188 |
| 128 | `sequence_only` | 0.6583 | 4.9104 | 6.5289 | 0.391 | 0.3906 |
| 128 | `one_backward_gradient` | 0.8271 | 4.4451 | 6.6122 | 0.172 | 0.1719 |
| 128 | `hybrid_union_m1` | 0.0841 | 0.7418 | 1.2336 | 0.719 | 0.7188 |
| 128 | `hybrid_union_m2` | 0.0841 | 0.7418 | 1.2336 | 0.719 | 0.7188 |
| 128 | `hybrid_union_m4` | 0.0810 | 0.7176 | 1.2336 | 0.719 | 0.7188 |
| 128 | `hybrid_union_m8` | 0.0782 | 0.7176 | 1.2336 | 0.719 | 0.7188 |
| 128 | `random_control` | 1.1068 | 5.8530 | 8.8802 | 0.038 | 0.0383 |
| 256 | `cached_parent` | 0.0698 | 0.6868 | 1.2336 | 0.766 | 0.7656 |
| 256 | `sequence_only` | 0.5702 | 4.2829 | 5.6276 | 0.438 | 0.4375 |
| 256 | `one_backward_gradient` | 0.7179 | 4.4276 | 6.5865 | 0.281 | 0.2812 |
| 256 | `hybrid_union_m1` | 0.0698 | 0.6868 | 1.2336 | 0.766 | 0.7656 |
| 256 | `hybrid_union_m2` | 0.0698 | 0.6868 | 1.2336 | 0.766 | 0.7656 |
| 256 | `hybrid_union_m4` | 0.0698 | 0.6868 | 1.2336 | 0.766 | 0.7656 |
| 256 | `hybrid_union_m8` | 0.0698 | 0.6868 | 1.2336 | 0.766 | 0.7656 |
| 256 | `random_control` | 0.8977 | 5.3435 | 6.6928 | 0.092 | 0.0922 |

## Exploratory failure detection

ROC AUC for predicting the catastrophic event from a label-free signal.
`roc_auc_negated` is the same signal with the sign flipped, which settles
which direction of a signal, if any, carries the information.

| Budget | Event | Signal | Positives | ROC AUC | Negated | Avg precision |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `regret_gt_0.02` | `struct_seq_top8_disjoint` | 35/64 | 0.808 | 0.192 | 0.854 |
| 1 | `rank_gt_K` | `struct_seq_top8_disjoint` | 38/64 | 0.789 | 0.211 | 0.856 |
| 1 | `regret_gt_0.05` | `struct_seq_top8_disjoint` | 34/64 | 0.789 | 0.211 | 0.842 |
| 2 | `regret_gt_0.2` | `struct_seq_top8_disjoint` | 22/64 | 0.784 | 0.216 | 0.666 |
| 256 | `regret_gt_0.02` | `top32_dispersion` | 14/64 | 0.220 | 0.780 | 0.148 |
| 256 | `regret_gt_0.05` | `top32_dispersion` | 14/64 | 0.220 | 0.780 | 0.148 |
| 256 | `regret_gt_0.1` | `top8_dispersion` | 12/64 | 0.223 | 0.777 | 0.129 |
| 256 | `regret_gt_0.02` | `top8_dispersion` | 14/64 | 0.227 | 0.773 | 0.149 |
| 256 | `regret_gt_0.05` | `top8_dispersion` | 14/64 | 0.227 | 0.773 | 0.149 |
| 1 | `regret_gt_0.1` | `struct_seq_top8_disjoint` | 31/64 | 0.773 | 0.227 | 0.796 |
| 1 | `regret_gt_0.02` | `struct_seq_top32_disjoint` | 35/64 | 0.771 | 0.229 | 0.778 |
| 1 | `regret_gt_0.2` | `struct_seq_top8_disjoint` | 29/64 | 0.771 | 0.229 | 0.751 |
| 256 | `regret_gt_0.1` | `top32_dispersion` | 12/64 | 0.234 | 0.766 | 0.129 |
| 2 | `regret_gt_0.2` | `struct_seq_top32_disjoint` | 22/64 | 0.764 | 0.236 | 0.575 |
| 1 | `regret_gt_0.1` | `struct_seq_top32_disjoint` | 31/64 | 0.763 | 0.237 | 0.748 |
| 1 | `rank_gt_K` | `struct_seq_top32_disjoint` | 38/64 | 0.763 | 0.237 | 0.795 |
| 2 | `regret_gt_0.1` | `struct_seq_top32_disjoint` | 28/64 | 0.760 | 0.240 | 0.661 |
| 1 | `regret_gt_0.05` | `struct_seq_top32_disjoint` | 34/64 | 0.758 | 0.242 | 0.762 |
| 2 | `rank_gt_K` | `struct_seq_top32_disjoint` | 32/64 | 0.757 | 0.243 | 0.691 |
| 1 | `regret_gt_0.2` | `struct_seq_top32_disjoint` | 29/64 | 0.757 | 0.243 | 0.721 |
| 2 | `regret_gt_0.02` | `struct_seq_top8_disjoint` | 31/64 | 0.753 | 0.247 | 0.748 |
| 2 | `regret_gt_0.05` | `struct_seq_top8_disjoint` | 31/64 | 0.753 | 0.247 | 0.748 |
| 256 | `rank_gt_K` | `top32_dispersion` | 15/64 | 0.248 | 0.752 | 0.163 |
| 2 | `rank_gt_K` | `struct_seq_top8_disjoint` | 32/64 | 0.751 | 0.249 | 0.751 |
| 2 | `regret_gt_0.02` | `struct_seq_top32_disjoint` | 31/64 | 0.750 | 0.250 | 0.678 |
| 2 | `regret_gt_0.05` | `struct_seq_top32_disjoint` | 31/64 | 0.750 | 0.250 | 0.678 |
| 128 | `regret_gt_0.1` | `top8_dispersion` | 14/64 | 0.251 | 0.749 | 0.154 |
| 16 | `regret_gt_0.02` | `struct_seq_top8_disjoint` | 24/64 | 0.745 | 0.255 | 0.619 |
| 16 | `regret_gt_0.05` | `struct_seq_top8_disjoint` | 24/64 | 0.745 | 0.255 | 0.619 |
| 256 | `rank_gt_K` | `top8_dispersion` | 15/64 | 0.256 | 0.744 | 0.165 |

Detectors here are exploratory and require a fresh confirmatory batch.
