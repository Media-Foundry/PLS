# GB1 random-frontier development baseline v1

This is a development result on 16 previously used GB1 v1 anchors, not a blind
or confirmatory evaluation. The run used the same initial queried-node protocol,
student architecture, training seeds, candidate universe, and budgets as the v1
Path/uncertainty experiments. Every reported regret uses experimentally measured
GB1 nodes only.

At the final 640-query budget, mean global R-squared was 0.272119. The separated
design regrets were:

| radius | acquired regret | novel-design regret | campaign regret |
| ---: | ---: | ---: | ---: |
| 1 | 1.165121 | 0.256732 | 0.234290 |
| 2 | 2.930256 | 2.352326 | 1.866010 |
| 3 | 3.543170 | 3.886505 | 3.113651 |
| 4 | 4.225344 | 5.583884 | 4.198020 |

All 16 anchors retained at least one unqueried feasible candidate at every edit
radius, so novel-design regret was defined for every cell. The complete budget
curve and per-anchor values are stored in the run artifacts. No PLS test split
was evaluated.
