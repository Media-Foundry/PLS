# Does the optimum escape the parent's structural basin?

Post-hoc mechanism diagnosis. It needs the mutant's own exact structure and
therefore can never be a pre-fold detector feature. No new folds.

Displacement is the exact mutant's C-alpha coordinates against the parent's,
Kabsch-superposed on the identity alignment.

| Anchor | L | Cached rank of best | Best RMSD | Best TM | Best RMSD pct in own nbhd | Nbhd median RMSD | Nbhd median TM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 82 | 8 | 0.183 | 0.9969 | 0.016 | 0.507 | 0.9800 |
| 1 | 54 | 1 | 1.755 | 0.8397 | 0.891 | 0.245 | 0.9902 |
| 2 | 70 | 2 | 0.169 | 0.9967 | 0.426 | 0.194 | 0.9957 |
| 3 | 54 | 1 | 0.150 | 0.9962 | 0.160 | 0.367 | 0.9807 |
| 4 | 70 | 1 | 0.405 | 0.9816 | 0.078 | 1.160 | 0.8860 |
| 5 | 86 | 1 | 0.871 | 0.9817 | 0.953 | 0.254 | 0.9944 |
| **6** | 69 | **1016** | 2.425 | 0.6296 | 0.445 | 2.608 | 0.6434 |
| 7 | 80 | 1 | 2.645 | 0.6434 | 0.719 | 1.932 | 0.8095 |

## Catastrophic against reliable anchors

| Quantity | Catastrophic | Reliable |
| --- | ---: | ---: |
| `rmsd` | 2.4254 | 0.8826 |
| `tm_score` | 0.6296 | 0.9195 |
| `maximum_deviation` | 4.6554 | 3.4820 |
| `mean_plddt_change` | 0.0488 | -0.0088 |
| `min_plddt_change` | 0.0171 | -0.0276 |
| `exact_best_rmsd_percentile` | 0.4453 | 0.4632 |
| `neighborhood_rmsd_median` | 2.6082 | 0.6656 |
| `neighborhood_rmsd_p90` | 6.9425 | 2.7443 |
| `neighborhood_tm_median` | 0.6434 | 0.9481 |

## Reading

The literal hypothesis is not supported. A reliable anchor's optimum moved
further from its parent than the catastrophic anchor's (RMSD 2.645 against
2.425, TM 0.6434 against 0.6296) and was still ranked first, and in both
classes the optimum sits at an ordinary percentile of its own neighborhood.

What separates the classes is the neighborhood as a whole. The catastrophic
anchor is the only one whose **median** mutant loses the fold. The parent
structure stops being a valid proxy for the entire neighborhood, not just for
the optimum.

One catastrophic anchor out of eight. This is a lead, not a finding.
