# PLS multi-fidelity correction v1

This test-free analysis uses 1,024 component-unique train exact edges for grouped-CV correction and evaluates the prespecified candidates once on 2,048 exact validation edges.

| Method | Edge Pearson | Edge Spearman | Sign | RMSE | Macro Kendall | Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_parent | 0.7481 | 0.6896 | 0.7842 | 0.3414 | 0.5850 | 0.6844 |
| affine | 0.7481 | 0.6896 | 0.7876 | 0.3419 | 0.5850 | 0.6844 |
| ridge_residual | 0.7374 | 0.6191 | 0.7188 | 0.3467 | 0.5185 | 0.6375 |
| nonlinear_residual | 0.7460 | 0.6562 | 0.7568 | 0.3423 | 0.5234 | 0.6469 |

Train grouped-OOF RMSE selected `fixed_parent` before validation reporting.

## Selective refolding

The table uses the selected correction as the unrefolded prediction. Random values are means over the frozen repetitions; oracle discrepancy is non-deployable context.

| Exact fraction | Selective Spearman | Random Spearman | Oracle Spearman | Selective sign | Selective Top-5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.6896 | 0.6896 | 0.6896 | 0.7842 | 0.6844 |
| 1% | 0.6927 | 0.6928 | 0.7016 | 0.7852 | 0.6859 |
| 2% | 0.6951 | 0.6965 | 0.7203 | 0.7876 | 0.6891 |
| 5% | 0.7011 | 0.7068 | 0.7637 | 0.7905 | 0.6922 |
| 10% | 0.7174 | 0.7240 | 0.8276 | 0.7964 | 0.7031 |
| 20% | 0.7506 | 0.7584 | 0.9030 | 0.8159 | 0.7266 |
| 50% | 0.8617 | 0.8538 | 0.9877 | 0.8813 | 0.7953 |
| 100% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The fixed-parent score is conditional on the anchor structure, so these results establish local single-edit fidelity only; they do not define a global scalar potential over multi-step paths.

Repository test queries/evaluations: **0**.
