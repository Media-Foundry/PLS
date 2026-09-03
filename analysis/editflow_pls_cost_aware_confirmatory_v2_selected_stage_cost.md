# Confirmatory v2: measured selected-stage compute

The gated selection was committed before this campaign ran, so the numerator
of the primary endpoint is directly measured rather than replayed. The
denominator is still open: the unselected candidates have not been folded.

- policy: `exact_best_runtime_gamma1`
- preregistered fold plan SHA-256: `00f2a7dab3da2cf7a428d8605cf7c54f0a20e753a28192655b72254042434c29`
- selected folds: 1245, failures: 0
- unselected candidates still unfolded: 803
- campaign: 2026-09-03T20:25:19+08:00 to 2026-09-03T20:39:01+08:00
- test sequences queried: 0

## Measured selected-stage cost

| Quantity | Value |
| --- | ---: |
| ESMFold GPU seconds, total | 3038.4 |
| ESMFold GPU seconds, warm-up excluded | 3021.8 |
| selected-stage wall seconds, 4 GPUs | 816.8 |
| mean seconds per fold | 2.441 |
| median seconds per fold | 1.830 |
| maximum seconds per fold | 6.960 |
| shard wall imbalance ratio | 1.0147 |

## Frozen runtime model against measured cost

| Quantity | Value |
| --- | ---: |
| predicted selected GPU seconds | 3086.2 |
| measured over predicted, total | 0.9845 |
| measured over predicted, marginal | 0.9791 |
| median absolute percentage error, marginal | 5.44% |

The model was fitted to marginal inference seconds with the first successful
sequence per process removed, so the total includes one warm-up fold per
shard and is expected to run above prediction.

## What is still open

Predicted cost fraction is 0.6009 against a query
fraction of 0.6079. The measured fraction cannot be stated
until the unselected candidates are folded retrospectively. Coverage, regret,
and every risk endpoint likewise remain uncomputed. Do not quote a measured
saving from this document alone.
