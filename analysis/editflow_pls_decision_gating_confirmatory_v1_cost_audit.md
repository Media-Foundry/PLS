# Confirmatory decision-gating cost and tail-risk audit

This is a deterministic post-hoc replay of the frozen candidate sets; no oracle was queried and the frozen threshold was not recomputed.

| Method | Queries | ESMFold GPU-s | GPU saving | Replayed 4-GPU wall | Wall saving | Coverage | Mean regret | Failure mean | CVaR95 | Max regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conformal_margin | 621 | 2236.8 | 26.9% | 693.0s | 22.2% | 0.9375 | 0.0271 | 0.4336 | 0.4336 | 1.3061 |
| fixed_top_m_m4 | 256 | 754.1 | 75.4% | 285.9s | 67.9% | 0.8281 | 0.1107 | 0.6441 | 1.5954 | 4.6084 |
| fixed_top_m_m8 | 512 | 1523.4 | 50.2% | 492.3s | 44.7% | 0.9375 | 0.0952 | 1.5226 | 1.5226 | 4.5960 |
| component_safe_conformal_rank_set_m12 | 768 | 2305.4 | 24.7% | 700.5s | 21.4% | 0.9844 | 0.0204 | 1.3061 | 0.3265 | 1.3061 |
| exhaustive_exact_m16 | 1024 | 3060.3 | 0.0% | 890.9s | 0.0% | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

ESMFold GPU-seconds are measured per sequence. Selected-method wall times are counterfactual replays on the original four shard assignments; exhaustive wall time alone was directly measured.
Exact-sequence PLM extraction and fixed-parent scoring are shared upfront work and therefore are not counted as savings. Selected-only exact geometry/patch/scoring wall time was not instrumented in v1.

PLS test queries/evaluations: **0**.
