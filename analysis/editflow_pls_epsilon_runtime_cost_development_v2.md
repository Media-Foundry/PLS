# Epsilon-optimal and cost-aware conformal development v1

Exploratory five-fold cross-fitting uses only the old 128 train anchors from 89 SI30 components. Calibration uses the maximum anchor nonconformity per component. The held-out confirmatory 64 components and the PLS test split were not used for method tuning.

| Epsilon | Cost gamma | Anchor epsilon cov. | Component epsilon cov. | Exact coverage | Mean queries | Query fraction | Cost fraction | Mean regret | CVaR95 | Max regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.00 | 0.9062 | 0.8989 | 0.9062 | 9.26 | 0.5786 | 0.6630 | 0.0153 | 0.2475 | 0.6087 |
| 0.00 | 0.50 | 0.9219 | 0.9213 | 0.9219 | 9.92 | 0.6201 | 0.6605 | 0.0148 | 0.2547 | 0.6087 |
| 0.00 | 1.00 | 0.9375 | 0.9213 | 0.9375 | 10.90 | 0.6812 | 0.6514 | 0.0081 | 0.1444 | 0.3181 |
| 0.05 | 0.00 | 0.9141 | 0.9101 | 0.8750 | 7.75 | 0.4844 | 0.5706 | 0.0211 | 0.3293 | 0.6087 |
| 0.05 | 0.50 | 0.8906 | 0.8652 | 0.8438 | 7.91 | 0.4941 | 0.5136 | 0.0235 | 0.3259 | 0.6087 |
| 0.05 | 1.00 | 0.9375 | 0.9101 | 0.8984 | 9.02 | 0.5635 | 0.5302 | 0.0141 | 0.2325 | 0.3496 |
| 0.10 | 0.00 | 0.9141 | 0.8876 | 0.8203 | 6.55 | 0.4092 | 0.4959 | 0.0376 | 0.4820 | 0.6716 |
| 0.10 | 0.50 | 0.9062 | 0.8764 | 0.7969 | 5.88 | 0.3677 | 0.3848 | 0.0340 | 0.4242 | 0.6716 |
| 0.10 | 1.00 | 0.9297 | 0.8989 | 0.8203 | 7.16 | 0.4478 | 0.4145 | 0.0236 | 0.3259 | 0.6087 |
| 0.20 | 0.00 | 0.9141 | 0.8876 | 0.7891 | 5.70 | 0.3560 | 0.4231 | 0.0437 | 0.5208 | 0.6716 |
| 0.20 | 0.50 | 0.9297 | 0.9101 | 0.7891 | 5.38 | 0.3359 | 0.3505 | 0.0344 | 0.4276 | 0.6716 |
| 0.20 | 1.00 | 0.9453 | 0.9213 | 0.7891 | 6.04 | 0.3774 | 0.3452 | 0.0257 | 0.3259 | 0.6087 |

`gamma=0` is the ordinary margin score. Positive gamma tightens candidate sets for more expensive anchors. New runtime-cost protocols use a label-free monotone model and a reference cost frozen before conformal calibration. This matrix is development evidence, not a new confirmatory result.

PLS test queries/evaluations: **0**.
