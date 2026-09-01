# Confirmatory decision-margin miss audit

The frozen margin set missed the exact best mutation for 4 of 64 new anchors.
This post-evaluation diagnostic does not alter the method or threshold.

| Anchor | Length | Queries | Exact-best fixed rank | Fixed margin gap | Regret |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 183 | 1 | 8 | 1.1722 | 0.1913 |
| 48 | 110 | 8 | 13 | 0.4531 | 1.3061 |
| 60 | 71 | 2 | 3 | 0.2771 | 0.0872 |
| 61 | 210 | 1 | 2 | 0.3074 | 0.1499 |

Anchor 48 is the important failure: the exact-best mutation was ranked 13th by
the cached oracle and changed by `+1.4661` under exact refolding while its cached
delta was `-0.3133`. It is also the sole miss of the frozen rank-12 baseline.
Anchor 21 instead has an unusually overconfident cached top mutation: the gap
between the cached top and the eventual exact best is `1.1722`, so a constant
decision-margin set collapses to one query and fails. These examples show why
the observed 93.75% coverage is a marginal risk statement and cannot certify an
individual protein.

No PLS test entity was queried or evaluated.
