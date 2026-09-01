# GB1 prequential-calibrated path development result

This is development evidence on the same 16 GB1 starting points used by the
standard acquisition matrix. Anchors share one four-site landscape and are not
independent biological replicates. No external landscape or PLS test entity was
opened.

Each purchased frontier node contributes edge errors computed from predictions
made before its label was purchased. Those errors calibrate later-round ensemble
edge uncertainty with an additive 90% envelope. This is a prequential empirical
calibration diagnostic under adaptive selection, not a marginal conformal
coverage theorem.

## Result

| Method | final R2 | final edge Spearman | novel k=2 AUC | acquired k=2 AUC | campaign k=2 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCB | 0.4174 | 0.3485 | 1.9502 | 2.4845 | 1.5074 |
| Occupancy-only | 0.4533 | 0.3502 | **1.8662** | **1.2262** | **0.9998** |
| Adaptive path/UCB | **0.4743** | 0.3556 | 2.2114 | 1.3443 | 1.1037 |
| Prequential calibrated path | 0.4437 | **0.3671** | 1.9816 | 1.2656 | 1.0272 |

The local-design budget curve does not improve: radius-2 novel AUC is worse than
occupancy-only by `0.1154` and worse than UCB by `0.0314`. At the final 640-query
point it is competitive—novel regret `0.8684` and campaign regret `0.0255`,
versus occupancy-only `0.8774` and `0.0770`—but a late endpoint cannot rescue the
curve-level null result.

Calibration behaves as expected but largely saturates. Mean additive correction
is `1.20` after the first purchased batch and `1.49` after the second. The path
share therefore falls automatically from `77.6/80` to `94.1/160` and
`102.5/320`, with UCB filling the remaining frontier budget. Empirical coverage
on accumulated adaptively selected residuals is about 0.92 and 0.90. Thus the
method becomes a support-limited occupancy/UCB hybrid rather than producing a
new regret win.

The useful positive signal is the highest final global edge Spearman (`0.3671`),
but this is secondary and does not justify opening an untouched landscape. The
next algorithmic change should target candidate-path coverage or relative/path-
conditional calibration; another additive global quantile or fixed mixture is
not supported.
