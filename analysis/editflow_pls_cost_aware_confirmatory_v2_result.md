# Cost-aware conformal gating: confirmatory v2 result

One-pass evaluation of the preregistered selection against the exhaustive
exact oracle on 128 fresh SI30 components. Every policy and baseline is
charged its **measured** ESMFold seconds, not its query count.

- components: 128, candidates each: 16
- measured exhaustive ESMFold GPU seconds: 5053.8
- test sequences queried: 0

## Primary endpoint

| Quantity | Value |
| --- | ---: |
| measured selected-stage GPU seconds | 3038.4 |
| measured exhaustive GPU seconds | 5053.8 |
| measured GPU cost fraction | 0.6012 |
| exact-best coverage | 0.9297 |
| coverage 95% Clopper-Pearson | [0.8707, 0.9673] |

## Policies and baselines at measured cost

| Method | Coverage | Mean queries | Query fraction | Measured cost fraction | Mean regret | CVaR95 | Max regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `exact_best_runtime_gamma1` | 0.9297 | 9.73/16 | 0.6079 | 0.6012 | 0.0216 | 0.3922 | 0.7263 |
| `tolerant_runtime_gamma1` | 0.9062 | 3.89/16 | 0.2432 | 0.2314 | 0.0574 | 0.6937 | 1.0489 |
| `top_4` | 0.7969 | 4.00/16 | 0.2500 | 0.2518 | 0.0397 | 0.5433 | 0.8568 |
| `top_8` | 0.9062 | 8.00/16 | 0.5000 | 0.5001 | 0.0207 | 0.3570 | 0.8568 |
| `rank_12` | 0.9453 | 12.00/16 | 0.7500 | 0.7503 | 0.0127 | 0.2314 | 0.5965 |
| `exhaustive` | 1.0000 | 16.00/16 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

Coverage for the tolerant policy is its epsilon-optimal event; every other
row reports exact-best inclusion.

## Primary policy by predefined length stratum

Descriptive only. The finite-sample guarantee is marginal over SI30
components and does not hold separately within a bin.

| Length | Components | Coverage | Mean queries | Measured cost fraction | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| <=106 | 24 | 0.9583 | 8.83 | 0.5511 | 0.0303 |
| 107-148 | 29 | 1.0000 | 10.62 | 0.6649 | 0.0000 |
| 149-219 | 43 | 0.9302 | 9.63 | 0.6030 | 0.0203 |
| >=220 | 32 | 0.8438 | 9.72 | 0.5939 | 0.0365 |

## Reading this correctly

The guarantee is component-level marginal risk control under SI30-component
exchangeability. It is not a per-protein certificate, and a miss on any single
protein is compatible with the stated coverage. Report the measured cost
fraction, never the query fraction, as the compute saving.
