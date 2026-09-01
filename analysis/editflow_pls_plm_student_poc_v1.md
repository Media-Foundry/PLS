# Frozen-PLM intervention student proof of concept

This development experiment uses exactly the existing 272 train and 136
validation oracle nodes. The frozen ESM2-650M mean/residue representations are
sequence-only; no ESMFold query or test entity was added. All neural models use
the same seed and validation edge-RMSE selection rule.

| Student | Value Pearson | Edge Pearson | Edge Spearman | Sign accuracy | Edge RMSE | Top-5 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw-token potential | **0.6447** | -0.1002 | **0.1171** | 0.5391 | 0.6686 | 0.3500 |
| Frozen-PLM potential | -0.0321 | **0.6831** | 0.1028 | **0.6016** | **0.5383** | **0.4500** |
| Parent-state direct delta | anchored diagnostic | 0.0330 | 0.0856 | 0.5000 | 0.6599 | 0.3750 |
| Direct delta + cycle | anchored diagnostic | 0.0246 | 0.0743 | 0.4922 | 0.6604 | 0.4000 |

The PLM potential learns a few large mutation effects, improving linear edge
correlation, RMSE, sign accuracy, and top-5 recall, but does not improve the
global mutation ranking. Its absolute-value generalization also collapses. The
direct edit parameterization and 1,900 unlabelled commuting-square constraints
do not help. Anchored value metrics for direct models assume one known teacher
value per validation landscape and are deliberately excluded from the table.

A separate four-fold train-anchor Group-CV ridge diagnostic reaches at most
edge Spearman `0.0881` for exact target-parent residue-PLM differences. Alpha is
selected without validation access. This rules out nonlinear-head overfitting
as the sole explanation.

The correct interpretation is narrower than “PLM priors do not help”: the 272
training nodes comprise only 16 independent proteins, each with a highly
correlated 16-edge star. The next experiment must increase independent anchor
coverage before changing the architecture or cycle weight again.
