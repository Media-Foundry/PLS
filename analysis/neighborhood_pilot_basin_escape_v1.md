# Does the optimum escape the parent's structural basin?

Post-hoc mechanism diagnosis. It needs the mutant's own exact structure and
therefore can never be a pre-fold detector feature. No new folds.

Displacement is the exact mutant's C-alpha coordinates against the parent's,
Kabsch-superposed on the identity alignment, over EVERY mutant in every
neighborhood.

| Anchor | L | Mutants | Cached rank of best | Best TM | Best RMSD pct | Nbhd median TM | Nbhd TM p10 | Frac TM<0.7 | Frac TM<0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 82 | 1558 | 8 | 0.9969 | 0.028 | 0.9786 | 0.8667 | 0.053 | 0.049 |
| 1 | 54 | 1026 | 1 | 0.8397 | 0.883 | 0.9901 | 0.7949 | 0.058 | 0.038 |
| 2 | 70 | 1330 | 2 | 0.9967 | 0.408 | 0.9953 | 0.8850 | 0.014 | 0.002 |
| 3 | 54 | 1026 | 1 | 0.9962 | 0.135 | 0.9796 | 0.9102 | 0.012 | 0.008 |
| 4 | 70 | 1330 | 1 | 0.9816 | 0.069 | 0.8742 | 0.2400 | 0.316 | 0.237 |
| 5 | 86 | 1634 | 1 | 0.9817 | 0.948 | 0.9942 | 0.9641 | 0.003 | 0.003 |
| **6** | 69 | 1311 | **1016** | 0.6296 | 0.455 | 0.6527 | 0.2703 | 0.588 | 0.275 |
| 7 | 80 | 1520 | 1 | 0.6434 | 0.712 | 0.8176 | 0.5718 | 0.273 | 0.045 |

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
| `neighborhood_fraction_tm_below_0.5` | 0.2746 | 0.0546 |
| `neighborhood_fraction_tm_below_0.7` | 0.5881 | 0.1042 |
| `neighborhood_fraction_tm_below_0.9` | 0.9443 | 0.2589 |
| `neighborhood_mean_plddt_change` | -0.0472 | -0.0093 |

## Reading

The literal hypothesis is not supported. A reliable anchor's optimum moved
further from its parent than the catastrophic anchor's (RMSD 2.645 against
2.425, TM 0.6434 against 0.6296) and was still ranked first, and in both
classes the optimum sits at an ordinary percentile of its own neighborhood.

What separates the classes is the neighborhood as a whole. TM < 0.7 is used
here as a stringent operational marker of substantial structural drift; the
classical approximate same-fold transition is nearer TM = 0.5, so both are
reported. The parent structure stops being a good proxy across the entire
neighborhood, not just for the optimum.

One catastrophic anchor out of eight. This is a lead, not a finding.
