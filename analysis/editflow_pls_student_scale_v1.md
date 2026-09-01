# Scaled PLS intervention-student development result

This development experiment uses 128 train anchors, 32 validation anchors,
2,560 exact single-mutant edges, and 2,720 exact sequence nodes. No test entity
was queried or evaluated. A post-hoc component audit found that the train
anchors cover only 89 unique SI30 components; validation covers 32/32. This
artifact therefore establishes a coverage trend but is not the final large-scale
student experiment.

## Frozen full versus matched sequence-only teacher

| Scope | Edges | mean abs full delta | mean abs sequence delta | mean abs residual delta | delta Spearman | sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2,048 | 0.2114 | 0.0446 | 0.1899 | 0.4311 | 0.6587 |
| Validation | 512 | 0.2918 | 0.0574 | 0.2681 | 0.4279 | 0.6855 |
| All | 2,560 | 0.2275 | 0.0471 | 0.2055 | 0.4309 | 0.6641 |

The full and same-checkpoint sequence-only node logits have Pearson `0.7157`.
The structure/fusion residual remains almost as large as the full local effect,
so the expanded set confirms that the derived-modality oracle is nontrivial.

Two independent float32 scoring passes differ by at most `1.93e-4` in the full
logit (mean absolute difference `4.54e-7`); sequence-only logits are bit-exact.
The maximum replay deviation is below 0.1% of the mean absolute full mutation
effect and does not change the qualitative field conclusions.

## Identical-node student comparison

| Sequence-native method | Value Pearson | Edge Pearson | Edge Spearman | Sign accuracy | Edge RMSE | Top-5 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw-token potential | 0.5568 | 0.1233 | 0.1283 | 0.5469 | 0.6790 | 0.3313 |
| Frozen-PLM potential | 0.5951 | 0.3344 | 0.3297 | 0.6211 | 0.6434 | 0.4500 |
| Parent-state direct delta | anchored diagnostic excluded | 0.0814 | 0.0795 | 0.5547 | 0.6797 | 0.3375 |
| Exact-pair direct delta | anchored diagnostic excluded | 0.1983 | 0.2887 | 0.5664 | 0.6673 | 0.4313 |
| Direct delta + cycle | anchored diagnostic excluded | 0.0787 | 0.0822 | 0.5527 | 0.6799 | 0.3250 |
| **Matched sequence + learned structural residual potential** | **0.7148** | **0.4159** | **0.3727** | **0.6563** | **0.6208** | **0.4813** |
| Frozen matched sequence teacher | 0.7782 | 0.3921 | 0.4279 | 0.6855 | 0.6480 | 0.5313 |

Increasing anchor coverage unlocks the conservative PLM potential: its edge
Spearman rises from `0.1028` in the 16-train-anchor PoC to `0.3297` here.
Direct edge prediction remains weaker. Exact target-parent PLM response helps
the direct model substantially, but does not beat a scalar potential.

The structural-residual parameterization is the strongest learned student by
edge Pearson and RMSE. Relative to the plain PLM potential it raises edge
Pearson by `0.0814` and lowers RMSE by `0.0226`. Relative to the matched sequence
teacher it improves linear correlation/RMSE but loses rank, sign, and top-k
fidelity. The result supports preserving a strong pretrained sequence potential
and learning only the derived-modality correction; it does not yet establish a
better mutation selector.

The next protocol expands to 1,024 unique train SI30 components and 128 unique
validation components. It is label-blind, retains zero test queries, and avoids
interpreting multiple homologous entities as independent coverage.
