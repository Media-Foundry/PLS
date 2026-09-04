# Decision regret and label-free reliability over complete neighborhoods

8 pilot anchors. Every signal below costs zero mutant folds; the
exact field scores the outcome and never feeds a signal.

## Regret, not exact-best identity

| Budget | Mean regret | Median | Max | P(R<=0) | P(R<=0.02) | P(R<=0.05) | P(R<=0.10) | P(R<=0.20) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3341 | 0.0000 | 1.4481 | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 |
| 2 | 0.2820 | 0.0000 | 1.4481 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| 4 | 0.2162 | 0.0000 | 0.9215 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| 8 | 0.0875 | 0.0000 | 0.6997 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| 16 | 0.0427 | 0.0000 | 0.3417 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| 32 | 0.0427 | 0.0000 | 0.3417 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| 64 | 0.0427 | 0.0000 | 0.3417 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| 128 | 0.0300 | 0.0000 | 0.2399 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |
| 256 | 0.0215 | 0.0000 | 0.1717 | 0.875 | 0.875 | 0.875 | 0.875 | 1.000 |

## Per-anchor: does the miss matter?

| Anchor | L | Cached rank of exact best | Exact best gain | Regret at K=8 | Regret at K=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 82 | 8 | 1.6907 | 0.0000 | 0.0000 |
| 1 | 54 | 1 | 0.6346 | 0.0000 | 0.0000 |
| 2 | 70 | 2 | 6.2773 | 0.0000 | 0.0000 |
| 3 | 54 | 1 | 1.2550 | 0.0000 | 0.0000 |
| 4 | 70 | 1 | 2.1156 | 0.0000 | 0.0000 |
| 5 | 86 | 1 | 0.5584 | 0.0000 | 0.0000 |
| 6 | 69 | 1016 | 0.7423 | 0.6997 | 0.3417 |
| 7 | 80 | 1 | 1.1457 | 0.0000 | 0.0000 |

## Label-free signals against the outcome

Spearman between each zero-fold statistic and log10 of the cached rank of the
exact best. Eight points: read the sign and the magnitude, nothing finer.

| Signal | Spearman with log rank |
| --- | ---: |
| `struct_seq_top8_disjoint` | -0.5809 |
| `top32_dispersion` | -0.5455 |
| `median_absolute_deviation` | -0.4364 |
| `effective_support_tau_mad` | +0.3546 |
| `struct_seq_top32_disjoint` | -0.3451 |
| `top_gap_struct_minus_seq` | -0.3273 |
| `struct_seq_rank_disagreement` | -0.3000 |
| `top_margin` | -0.2455 |
| `top8_dispersion` | -0.2455 |
| `top_margin_over_mad` | -0.1364 |
| `beneficial_fraction` | +0.0273 |

## Do cheap union budgets rescue the miss?

| Union | Anchors hitting exact best | Mean regret |
| --- | ---: | ---: |
| `top6_cached_plus_top2_seq` | 0.7500 | 0.1643 |
| `top4_cached_plus_top4_seq` | 0.7500 | 0.1188 |
| `top12_cached_plus_top4_seq` | 0.8750 | 0.0178 |
| `top24_cached_plus_top8_seq` | 0.8750 | 0.0178 |
