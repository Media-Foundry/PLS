# Does the optimum escape the parent's structural basin?

Post-hoc mechanism diagnosis. It needs the mutant's own exact structure and
therefore can never be a pre-fold detector feature. No new folds.

Displacement is the exact mutant's C-alpha coordinates against the parent's,
Kabsch-superposed on the identity alignment, over EVERY mutant in every
neighborhood.

| Anchor | L | Mutants | Cached rank of best | Best RMSD | Best TM | Best RMSD pct | Nbhd median RMSD | Nbhd median TM | Frac TM<0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 82 | 1558 | 8 | 0.183 | 0.9969 | 0.028 | 0.531 | 0.9786 | 0.053 |
| 1 | 54 | 1026 | 1 | 1.755 | 0.8397 | 0.883 | 0.251 | 0.9901 | 0.058 |
| 2 | 70 | 1330 | 2 | 0.169 | 0.9967 | 0.408 | 0.204 | 0.9953 | 0.014 |
| 3 | 54 | 1026 | 1 | 0.150 | 0.9962 | 0.135 | 0.375 | 0.9796 | 0.012 |
| 4 | 70 | 1330 | 1 | 0.405 | 0.9816 | 0.069 | 1.216 | 0.8742 | 0.316 |
| 5 | 86 | 1634 | 1 | 0.871 | 0.9817 | 0.948 | 0.265 | 0.9942 | 0.003 |
| **6** | 69 | 1311 | **1016** | 2.425 | 0.6296 | 0.455 | 2.594 | 0.6527 | 0.588 |
| 7 | 80 | 1520 | 1 | 2.645 | 0.6434 | 0.712 | 1.869 | 0.8176 | 0.273 |

## Catastrophic against reliable anchors

| Quantity | Catastrophic | Reliable |
| --- | ---: | ---: |
| `rmsd` | 2.4254 | 0.8826 |
| `tm_score` | 0.6296 | 0.9195 |
| `maximum_deviation` | 4.6554 | 3.4820 |
| `mean_plddt_change` | 0.0488 | -0.0088 |
| `rmsd_percentile_in_neighborhood` | 0.4546 | 0.4546 |
| `tm_percentile_in_neighborhood` | 0.4561 | 0.5613 |
| `neighborhood_rmsd_median` | 2.5941 | 0.6729 |
| `neighborhood_rmsd_p90` | 6.7663 | 2.7371 |
| `neighborhood_tm_median` | 0.6527 | 0.9471 |
| `neighborhood_tm_p10` | 0.2703 | 0.7475 |
| `neighborhood_fraction_tm_below_0.7` | 0.5881 | 0.1042 |
| `neighborhood_fraction_tm_below_0.9` | 0.9443 | 0.2589 |
| `neighborhood_mean_plddt_change` | -0.0472 | -0.0093 |

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
