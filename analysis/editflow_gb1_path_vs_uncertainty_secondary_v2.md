# GB1 Path-OLD paired comparison

Negative paired differences favor path-aware acquisition.
Regret component: `legacy_all_candidates`. Analysis status: `secondary_descriptive`.

| endpoint | path mean | uncertainty mean | paired difference | bootstrap 95% CI | exact p | W/T/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Query-curve regret | 1.532140 | 1.882403 | -0.350263 | [-0.642677, -0.068437] | 0.0359497 | 12/0/4 |
| Final-budget regret | 0.582586 | 1.206575 | -0.623989 | [-0.904790, -0.349668] | 0.000671387 | 13/0/3 |

This comparison was not a prespecified primary analysis. Its p-values
and confidence intervals are descriptive and cannot establish a new
confirmatory claim.
The legacy all-candidate metric is confounded by allowing the student
to select nodes whose oracle labels were already purchased. It is
reported only as historical evidence, not as distillation evidence.
No PLS test split was evaluated.
