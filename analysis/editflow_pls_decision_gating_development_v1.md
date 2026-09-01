# Decision-focused cached-oracle development v1

No PLS test entity was loaded, queried, scored, or evaluated.

## Top-M exact verification on the frozen 128-component validation report

| M | Exact fraction | True-best inclusion | Zero regret | Mean regret | P90 regret |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0625 | 0.4922 | 0.4922 | 0.1338 | 0.2940 |
| 2 | 0.1250 | 0.6641 | 0.6641 | 0.0790 | 0.2438 |
| 4 | 0.2500 | 0.8438 | 0.8438 | 0.0404 | 0.0350 |
| 8 | 0.5000 | 0.9531 | 0.9531 | 0.0073 | 0.0000 |

## Protein-specific shrinkage-slope diagnostic on old exhaustive train anchors

| Exact probes/anchor | Held-out edges | RMSE | Pearson | Sign |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2048 | 0.3192 | 0.6925 | 0.7773 |
| 1 | 1920 | 0.3225 | 0.6812 | 0.7797 |
| 2 | 1792 | 0.3247 | 0.6932 | 0.7790 |
| 4 | 1536 | 0.3333 | 0.6910 | 0.7702 |

A positive per-anchor scalar cannot change mutation ordering; this diagnostic tests magnitude attenuation only.

## Train-only simultaneous certified gating

| Envelope | Simultaneous coverage | Decision accuracy | Mean exact queries | Exact fraction | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| component_max | 0.9141 | 0.9922 | 14.95 | 0.9341 | 0.0002 |
| bonferroni_edge | 0.9375 | 0.9922 | 15.07 | 0.9419 | 0.0002 |

## Train-only conformal exact-best candidate set

Component-safe cross-fit coverage is `0.9609` with mean `12.00` exact queries per 16-mutation neighborhood. The frozen future set size is `12`. This is marginal risk control, not a deterministic per-anchor certificate.

## Train-only conformal decision-margin set

The variable-size margin set reaches cross-fit coverage `0.9219` with mean `10.07` exact queries. Its frozen future margin threshold is `0.232594`. This is marginal component-level risk control, not a deterministic per-anchor certificate.

The certified-gating analysis is cross-fitted development evidence on train components only. It does not reuse the current validation set for method selection.
