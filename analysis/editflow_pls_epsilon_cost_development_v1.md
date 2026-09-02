# Epsilon-optimal and cost-aware conformal development v1

Exploratory five-fold cross-fitting uses only the old 128 train anchors from 89 SI30 components. Calibration uses the maximum anchor nonconformity per component. The held-out confirmatory 64 components and the PLS test split were not used for method tuning.

| Epsilon | Cost gamma | Anchor epsilon cov. | Component epsilon cov. | Exact coverage | Mean queries | Query fraction | Length^2 cost fraction | Mean regret | CVaR95 | Max regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.00 | 0.9062 | 0.8989 | 0.9062 | 9.26 | 0.5786 | 0.6975 | 0.0153 | 0.2475 | 0.6087 |
| 0.00 | 0.50 | 0.9297 | 0.9213 | 0.9297 | 9.87 | 0.6167 | 0.6928 | 0.0143 | 0.2538 | 0.6087 |
| 0.00 | 1.00 | 0.9453 | 0.9326 | 0.9453 | 10.70 | 0.6689 | 0.6852 | 0.0087 | 0.1588 | 0.3181 |
| 0.05 | 0.00 | 0.9141 | 0.9101 | 0.8750 | 7.75 | 0.4844 | 0.6060 | 0.0211 | 0.3293 | 0.6087 |
| 0.05 | 0.50 | 0.8906 | 0.8652 | 0.8594 | 8.00 | 0.5000 | 0.5568 | 0.0229 | 0.3259 | 0.6087 |
| 0.05 | 1.00 | 0.9062 | 0.8764 | 0.8594 | 8.14 | 0.5088 | 0.5022 | 0.0225 | 0.3259 | 0.6087 |
| 0.10 | 0.00 | 0.9141 | 0.8876 | 0.8203 | 6.55 | 0.4092 | 0.5256 | 0.0376 | 0.4820 | 0.6716 |
| 0.10 | 0.50 | 0.9141 | 0.8876 | 0.8047 | 6.19 | 0.3867 | 0.4381 | 0.0335 | 0.4242 | 0.6716 |
| 0.10 | 1.00 | 0.9219 | 0.8876 | 0.8281 | 6.46 | 0.4038 | 0.3999 | 0.0246 | 0.3307 | 0.6087 |
| 0.20 | 0.00 | 0.9141 | 0.8876 | 0.7891 | 5.70 | 0.3560 | 0.4473 | 0.0437 | 0.5208 | 0.6716 |
| 0.20 | 0.50 | 0.9297 | 0.9101 | 0.7969 | 5.62 | 0.3516 | 0.3968 | 0.0341 | 0.4276 | 0.6716 |
| 0.20 | 1.00 | 0.9453 | 0.9213 | 0.8203 | 5.92 | 0.3701 | 0.3694 | 0.0262 | 0.3307 | 0.6087 |

`gamma=0` is the ordinary margin score. Positive gamma tightens candidate sets for longer, more expensive anchors using a label-free length proxy. This matrix is development evidence, not a new confirmatory result.

PLS test queries/evaluations: **0**.
