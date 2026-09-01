# Fixed-parent offline oracle ablation

This test-free ablation asks whether every single mutant must be refolded to
obtain a useful dense PLS intervention landscape. Each mutant keeps its exact
sequence-specific ESM2 mean and residue tokens, while the anchor's complete
V4/GVP/surface tensor, coordinates, and pLDDT are reused unchanged. It requires
zero new mutant folds and is explicitly an approximate fixed-backbone oracle.

| Scope | Edges | Edge Pearson | Edge Spearman | Sign accuracy | Edge RMSE | Macro Kendall | Top-5 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 2,048 | 0.6925 | 0.6771 | 0.7773 | 0.3192 | 0.5681 | 0.6906 |
| Validation | 512 | **0.7441** | **0.7692** | **0.8281** | 0.4620 | **0.6042** | **0.7125** |
| All | 2,560 | 0.7059 | 0.6967 | 0.7875 | 0.3524 | 0.5753 | 0.6950 |

Node-level exact-versus-fixed Pearson is `0.9967` and Spearman is `0.9964`.
The local result is the decisive observation: even without mutation-aware
structure-channel updates, fixed-parent inference preserves about 70% rank
correlation and nearly 79% mutation signs across the complete exact-refold
reference set. It is much closer to the exact teacher than any trained
sequence-only student evaluated so far.

The approximation attenuates mutation magnitude (mean absolute fixed delta
`0.1460` versus exact `0.2275`) and has rare large node errors, so it cannot
replace exact folding as the final evaluator. It can replace exhaustive folding
as dense supervision. The next protocol should use a multi-fidelity design:

1. dense fixed-parent labels for all training edges;
2. a value-blind, component-covered exact-refold subset for correction;
3. exact-refold validation edges for final field/regret evaluation.

A mutation-aware fixed-backbone cache can improve on this lower bound by
recomputing identity lookup, local sequence, charge, hydropathy, exposure, and
surface-patch channels while retaining parent geometry. Its fidelity must be
measured against the same exact subset rather than assumed.
