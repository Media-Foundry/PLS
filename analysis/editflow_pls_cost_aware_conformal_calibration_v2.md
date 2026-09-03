# Fresh cost-aware conformal calibration v2

Frozen split-conformal quantiles for **Cost-Aware Conformal Gating for Cached
Derived-Modality Oracles**, fitted once on 128 fresh SI30 components that have
zero overlap with the confirmatory manifest and with all 1,216 previously used
oracle components.

- protocol: `configs/editflow/pls_cost_aware_conformal_protocol_v2.json`
- machine-readable quantiles: `analysis/editflow_pls_cost_aware_conformal_calibration_v2.json`
- expected-cost detail: `analysis/editflow_pls_cost_aware_conformal_calibration_v2_expected_cost.json`
- calibration manifest SHA-256: `ed03c0d5261850e2296b4640f0dc24deb707e62ed827b78569c5c062568b0dd9`
- runtime cost model SHA-256: `1f6d6ab1773e6af8daced429e32d71f15c3662ee2c011327648d5b6ae437e7fb`
- test sequences queried: 0; `test_evaluated: false` throughout.

Nothing in the frozen policy family was changed to produce these quantiles.
Epsilon, gamma, the runtime cost model, the score normalization, and the
component manifests are exactly as committed in `33f8e6d`.

## Frozen quantiles

| Policy | Role | alpha | epsilon | gamma | Order statistic | Quantile |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exact_best_runtime_gamma1` | primary | 0.1 | 0 | 1 | 117/128 | 0.3915251122 |
| `tolerant_runtime_gamma1` | secondary | 0.1 | 0.2 | 1 | 117/128 | 0.1390406841 |

Both policies use the direct split-conformal order statistic
`ceil((n + 1) * (1 - alpha))` with `n = 128`, giving rank 117. The legacy
one-order-higher convention used for the frozen v1 threshold was not applied.

## Expected candidate cost on the calibration components

| Policy | Mean exact queries | Query fraction | Predicted GPU-cost fraction |
| --- | ---: | ---: | ---: |
| `exact_best_runtime_gamma1` | 10.52/16 | 0.6577 | 0.6162 |
| `tolerant_runtime_gamma1` | 4.31/16 | 0.2695 | 0.2470 |

Predicted cost uses the frozen label-free runtime model, so the query fraction
and the GPU-cost fraction differ whenever the retained candidates sit on longer
proteins. That gap is the reason protocol v2 measures selected-stage compute
directly in the confirmatory stage instead of replaying query counts.

## In-sample diagnostics, not evidence

These components produced the quantiles, so their coverage is optimistic by
construction and is reported only to confirm the calibration is self-consistent.
Confirmatory coverage comes from the disjoint 128-component manifest and has not
been measured.

| Policy | In-sample epsilon coverage | Mean regret | CVaR95 | Max regret | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exact_best_runtime_gamma1` | 0.9141 | 0.0286 | 0.5071 | 1.1515 | 11/128 |
| `tolerant_runtime_gamma1` | 0.9141 | 0.0525 | 0.6315 | 1.2179 | 33/128 |

## Predefined length strata

Descriptive only. The finite-sample guarantee remains marginal across SI30
components; these bins are not separately valid groups. They exist to expose
cost-aware undercoverage on long proteins before the confirmatory stage.

### `exact_best_runtime_gamma1`

| Length | Components | Mean queries | Predicted cost fraction | In-sample coverage | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| <=106 | 20 | 9.60 | 0.6079 | 0.9500 | 0.0147 |
| 107-148 | 27 | 13.96 | 0.8699 | 0.9630 | 0.0063 |
| 149-219 | 42 | 10.38 | 0.6318 | 0.9048 | 0.0288 |
| >=220 | 39 | 8.77 | 0.5556 | 0.8718 | 0.0508 |

### `tolerant_runtime_gamma1`

| Length | Components | Mean queries | Predicted cost fraction | In-sample coverage | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| <=106 | 20 | 3.75 | 0.2377 | 0.9000 | 0.0328 |
| 107-148 | 27 | 6.19 | 0.3851 | 0.9259 | 0.0318 |
| 149-219 | 42 | 4.31 | 0.2638 | 0.9048 | 0.0589 |
| >=220 | 39 | 3.31 | 0.2106 | 0.9231 | 0.0700 |

## What happens next

These quantiles are now frozen inputs. The confirmatory stage must build only
cached-parent features and scores for the 128 confirmatory components, apply
these quantiles, and commit the selected-candidate manifest **before** any
confirmatory mutant is refolded. Folding confirmatory mutants ahead of that
commit would destroy the measured selected-stage cost that is the primary
endpoint.
