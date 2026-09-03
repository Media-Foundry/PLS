# Exhaustive single-mutant neighborhood pilot

- anchors: 8 fresh SI30 components, lengths 54-86
- candidates: 10735 complete single-mutant neighborhoods
- measured exhaustive ESMFold: 12746.9 GPU-seconds
- test sequences queried: 0

## Three links

| Link | Mean Spearman | Median | Min | Sign agreement |
| --- | ---: | ---: | ---: | ---: |
| A. one-backward gradient to cached-parent field | 0.2284 | 0.2341 | 0.1389 | 0.5739 |
| B. cached-parent field to exact field | 0.7111 | 0.7261 | 0.4354 | 0.7592 |
| C. gradient to exact field | 0.1699 | 0.1912 | -0.0232 | 0.5761 |
| D. sequence-only ablation to exact field | 0.4420 | 0.4144 | 0.1521 | 0.6604 |
| E. cached-parent to sequence-only | 0.5551 | 0.6221 | 0.0862 | 0.7043 |

## Exact-best recall at a fixed fold budget

Every row folds `budget` candidates out of the complete neighborhood and asks
whether the exact best single mutant is among them.

| Budget | Cached-parent | 95% CI | Sequence-only | Gradient | Random | Cached mean regret | Cost fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6250 | [0.24, 0.91] | 0.0000 | 0.0000 | 0.0000 | 0.3341 | 0.0008 |
| 2 | 0.7500 | [0.35, 0.97] | 0.0000 | 0.0000 | 0.0063 | 0.2820 | 0.0015 |
| 4 | 0.7500 | [0.35, 0.97] | 0.0000 | 0.0000 | 0.0000 | 0.2162 | 0.0029 |
| 8 | 0.8750 | [0.47, 1.00] | 0.1250 | 0.0000 | 0.0125 | 0.0875 | 0.0058 |
| 16 | 0.8750 | [0.47, 1.00] | 0.1250 | 0.0000 | 0.0187 | 0.0427 | 0.0116 |
| 32 | 0.8750 | [0.47, 1.00] | 0.2500 | 0.0000 | 0.0437 | 0.0427 | 0.0235 |
| 64 | 0.8750 | [0.47, 1.00] | 0.2500 | 0.1250 | 0.0312 | 0.0427 | 0.0473 |

## Per-anchor link B

| Anchor | L | Candidates | Spearman | Sign agreement |
| --- | ---: | ---: | ---: | ---: |
| 0 | 82 | 1558 | 0.8094 | 0.7407 |
| 1 | 54 | 1026 | 0.5981 | 0.7680 |
| 2 | 70 | 1330 | 0.8590 | 0.8850 |
| 3 | 54 | 1026 | 0.8549 | 0.8372 |
| 4 | 70 | 1330 | 0.6428 | 0.6564 |
| 5 | 86 | 1634 | 0.6116 | 0.6756 |
| 6 | 69 | 1311 | 0.4354 | 0.6545 |
| 7 | 80 | 1520 | 0.8773 | 0.8559 |

## Where the exact best mutation sits in each cheap ranking

| Anchor | L | Candidates | Cached-parent rank | Sequence-only rank | Gradient rank | Exact gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 82 | 1558 | 8 | 217 | 121 | 1.6907 |
| 1 | 54 | 1026 | 1 | 221 | 57 | 0.6346 |
| 2 | 70 | 1330 | 2 | 302 | 1072 | 6.2773 |
| 3 | 54 | 1026 | 1 | 31 | 595 | 1.2550 |
| 4 | 70 | 1330 | 1 | 6 | 461 | 2.1156 |
| 5 | 86 | 1634 | 1 | 1614 | 1120 | 0.5584 |
| 6 | 69 | 1311 | 1016 | 509 | 1304 | 0.7423 |
| 7 | 80 | 1520 | 1 | 296 | 142 | 1.1457 |

The cached-parent ranking is bimodal: it puts the exact best mutation first
or near-first on most anchors and fails almost completely on one. Read the
recall column with its interval; eight anchors is a pilot, not an estimate.
