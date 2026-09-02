# Cost-aware cached-oracle conformal protocol v2

The method is frozen before any fresh calibration exact score exists. It uses
no PLS test entity.

## Frozen policies

| Role | Epsilon | Runtime gamma | Alpha | Target |
| --- | ---: | ---: | ---: | --- |
| Primary | 0.0 | 1.0 | 0.1 | exact high-fidelity optimum |
| Secondary | 0.2 | 1.0 | 0.1 | regret at most 0.2 raw teacher logit |

The tolerant endpoint remains secondary because `epsilon=0.2` is approximately
`1.021` times the development exact mutation-effect IQR. It is a user-specified
oracle-score tolerance, not a measured change in experimental solubility.

## Independent components

| Stage | SI30 components | Anchors/component | Mutations/anchor | Prior overlap |
| --- | ---: | ---: | ---: | ---: |
| Method development | 89 | variable | 16 | historical |
| Fresh calibration | 128 | 1 | 16 | 0 |
| Fresh confirmatory | 128 | 1 | 16 | 0 |

Fresh calibration and confirmatory components also have zero overlap with each
other. Both manifests are label-blind, train-only, and test-free.

## Runtime model and execution

The frozen monotone cost model uses 5,814 marginal ESMFold timings from 20
homogeneous ROCm shards. Group-report cross-fit Spearman is `0.9339`; median
absolute percentage error is `2.97%`. The 2,048 calibration mutations are
balanced into four 512-query shards with identical predicted marginal cost
(`1382.78` seconds per shard).

Confirmatory exact work is intentionally not planned exhaustively yet. After
fresh quantiles are frozen, the primary candidate set is persisted before any
confirmatory mutant refold. Only selected candidates are folded and timed for
the deployment endpoint. Unselected candidates are folded later solely for the
retrospective coverage/regret audit.

PLS test queries/evaluations: **0**.
