# Frozen ESMFold runtime cost model v1

The label-free model uses 5,814 marginal timings from 20 homogeneous ROCm ESMFold shards. The first successful sequence per process is excluded as warm-up; startup remains a separate deployment cost.

| Diagnostic | Value |
| --- | ---: |
| Group-report CV Spearman | 0.9339 |
| Median absolute error | 0.0641 s |
| Median absolute percentage error | 2.97% |
| P90 absolute percentage error | 9.39% |
| Frozen reference cost | 1.5793 s |

The model is used only to allocate candidate-evaluation cost. It uses no oracle scores and accesses no PLS test entity.
