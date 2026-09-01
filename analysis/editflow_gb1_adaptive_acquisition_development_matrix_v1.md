# GB1 standard-acquisition development matrix v1

These are descriptive development results on multiple starting points in the
same four-site GB1 landscape. Anchors are not treated as independent biological
replicates, and no population-level significance test is reported. All methods
share the same initial-query protocol, unique-node budgets, candidate universe,
and oracle cost model. No PLS test split was evaluated.

## Budget-curve summary

Normalized AUC is the trapezoidal mean regret over budgets 80--640; lower is
better. The local-design column is novel-design regret at radius 2.

| acquisition | final R2 | final edge Spearman | novel k=2 AUC | acquired k=2 AUC | campaign k=2 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 0.272119 | 0.358450 | 2.803489 | 3.552650 | 2.554857 |
| greedy | 0.425563 | 0.356686 | 2.075718 | 2.559138 | 1.693615 |
| ucb | 0.417364 | 0.348462 | 1.950178 | 2.484534 | 1.507433 |
| thompson | 0.406282 | 0.348285 | 2.041621 | 2.405596 | 1.504196 |
| occupancy | 0.453331 | 0.350160 | 1.866244 | 1.226168 | 0.999814 |
| adaptive_path_ucb | 0.474266 | 0.355584 | 2.211444 | 1.344271 | 1.103699 |

## Final 640-query regret

| acquisition | radius | acquired | novel design | campaign | novel available |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 1 | 1.165121 | 0.256732 | 0.234290 | 16/16 |
| random | 2 | 2.930256 | 2.352326 | 1.866010 | 16/16 |
| random | 3 | 3.543170 | 3.886505 | 3.113651 | 16/16 |
| random | 4 | 4.225344 | 5.583884 | 4.198020 | 16/16 |
| greedy | 1 | 0.563538 | 0.220396 | 0.217641 | 16/16 |
| greedy | 2 | 1.473313 | 1.812162 | 1.015048 | 16/16 |
| greedy | 3 | 1.042601 | 1.945872 | 0.749717 | 16/16 |
| greedy | 4 | 1.111867 | 2.558027 | 1.067067 | 16/16 |
| ucb | 1 | 0.367194 | 0.352216 | 0.222324 | 16/16 |
| ucb | 2 | 1.499477 | 1.711937 | 0.826012 | 16/16 |
| ucb | 3 | 1.028079 | 1.668716 | 0.789630 | 16/16 |
| ucb | 4 | 0.969797 | 1.698577 | 0.723329 | 16/16 |
| thompson | 1 | 0.550707 | 0.374316 | 0.285237 | 16/16 |
| thompson | 2 | 1.721505 | 1.414643 | 0.916475 | 16/16 |
| thompson | 3 | 1.155697 | 1.902976 | 0.688409 | 16/16 |
| thompson | 4 | 1.164941 | 2.524500 | 1.067834 | 16/16 |
| occupancy | 1 | 0.000000 | 0.004712 | 0.000000 | 15/16 |
| occupancy | 2 | 0.077050 | 0.877365 | 0.077050 | 16/16 |
| occupancy | 3 | 0.887689 | 1.472589 | 0.631422 | 16/16 |
| occupancy | 4 | 1.960902 | 3.074454 | 1.863692 | 16/16 |
| adaptive_path_ucb | 1 | 0.008344 | 0.044063 | 0.008344 | 16/16 |
| adaptive_path_ucb | 2 | 0.173878 | 1.526063 | 0.146339 | 16/16 |
| adaptive_path_ucb | 3 | 0.730049 | 1.652249 | 0.552004 | 16/16 |
| adaptive_path_ucb | 4 | 1.183938 | 1.826340 | 0.993975 | 16/16 |

## Interpretation boundary

Occupancy-only is strongest for very local acquisition/campaign behavior, while
UCB is more competitive at the full radius. This is development evidence for an
exploitation--coverage trade-off, not evidence that an adaptive policy already
wins. Method selection remains confined to GB1 before an untouched landscape is
opened.
