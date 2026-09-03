# Cost-aware conformal gating: confirmatory v2 selection

Frozen quantiles applied once to the 128 fresh confirmatory SI30 components,
using only the cached-parent oracle and label-free covariates. **No confirmatory
mutant has been folded.** This document and its artifacts are the preregistration
that makes selected-stage ESMFold compute a measured primary endpoint.

- protocol SHA-256: `406d1ff12dcf8c832e70ab8495f89e8bd0216536ab7f5079e3cda7ea16c4d6b7`
- calibration artifact SHA-256: `5ef3d3742842aa7d218931b1c8fc915bad573cc1c5e49e5edcdee00447e6b466`
- confirmatory manifest SHA-256: `dda1f9f7b5acff8cc17f734d4c1ba921f88a936516f90bbac5ae3631d8b03195`
- cached-parent scores SHA-256: `f94824a66398721010ce3f017029eba37738b508d08e1d72e85bd6cbe5c85ff6`
- component overlap with calibration: 0
- exact confirmatory scores read: false
- test sequences queried: 0

## Selection under the frozen policies

| Policy | Role | Quantile | Selected | Mean queries | Query fraction | Predicted GPU-cost fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exact_best_runtime_gamma1` | primary | 0.3915251122 | 1245/2048 | 9.73/16 | 0.6079 | 0.6009 |
| `tolerant_runtime_gamma1` | secondary | 0.1390406841 | 498/2048 | 3.89/16 | 0.2432 | 0.2305 |

The primary policy retains 9.73 of 16 candidates on average, closely matching
both the calibration-set projection of 10.52/16 and the frozen v1 confirmatory
result of 9.70/16. The predicted cost fraction sits below the query fraction
because the cost scale tightens the margin on longer, more expensive candidates.

## Frozen fold plan for the primary policy

- plan: `benchmark/generated/pls_editflow_oracle_query_plan_cost_confirmatory_v2_selected.json`
- assignments SHA-256: `00f2a7dab3da2cf7a428d8605cf7c54f0a20e753a28192655b72254042434c29`
- selected mutants to fold: 1245
- shards: 4 across authorized physical GPUs 0--3
- maximum-to-minimum predicted shard cost ratio: 1.001326

| Shard | Queries | Predicted GPU seconds |
| ---: | ---: | ---: |
| 0 | 312 | 772.3 |
| 1 | 311 | 771.3 |
| 2 | 311 | 771.3 |
| 3 | 311 | 771.3 |

## Per-anchor selection spread, primary policy

| Statistic | Value |
| --- | ---: |
| minimum selected candidates | 1 |
| median selected candidates | 12.0 |
| maximum selected candidates | 16 |
| components selecting all 16 | 34 |
| components selecting exactly 1 | 18 |

Selected count by predefined length stratum, descriptive only:

| Length | Components | Mean selected | Predicted cost fraction |
| --- | ---: | ---: | ---: |
| <=106 | 24 | 8.83 | 0.5552 |
| 107-148 | 29 | 10.62 | 0.6649 |
| 149-219 | 43 | 9.63 | 0.5982 |
| >=220 | 32 | 9.72 | 0.5944 |

## Mandatory next steps

1. Commit and push this selection before folding anything.
2. Fold and score **only** the planned selected mutants on GPUs 0--3, recording
   real per-query GPU seconds and selected-stage wall time.
3. Freeze the selected-only deployment report.
4. Only then fold the unselected candidates for retrospective coverage, regret,
   and cost denominators.

Coverage and regret cannot be computed yet and must not be estimated from the
cached-parent scores. The finite-sample guarantee stays marginal over SI30
components and does not certify any individual protein.
