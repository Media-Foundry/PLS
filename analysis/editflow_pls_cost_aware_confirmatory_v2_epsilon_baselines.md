# Confirmatory v2 supplement: baselines on the epsilon-optimal event

The main result table charges Top-M with exact-best inclusion while the
tolerant policy is charged with epsilon-optimal inclusion at `epsilon=0.2`.
That makes them incomparable. This supplement rescores the fixed-budget
baselines on the tolerant policy's own risk event. It is descriptive and
changes no frozen number.

| Method | Coverage at epsilon=0.2 | 95% Clopper-Pearson | Measured cost fraction | Mean regret | CVaR95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `top_4` | 0.9375 | [0.8806, 0.9726] | 0.2518 | 0.0397 | 0.5433 |
| `top_8` | 0.9688 | [0.9219, 0.9914] | 0.5001 | 0.0207 | 0.3570 |
| `rank_12` | 0.9766 | [0.9330, 0.9951] | 0.7503 | 0.0127 | 0.2314 |
| `tolerant_runtime_gamma1` | 0.9062 | [0.8420, 0.9506] | 0.2314 | 0.0574 | 0.6937 |

Read the cost column as measured ESMFold seconds, never query count.
