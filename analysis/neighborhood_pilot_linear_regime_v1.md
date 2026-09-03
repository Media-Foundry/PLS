# How far the cached-parent oracle stays linear in sequence space

8 anchors, 32 sampled substitutions each, zero mutant folds.

A step of `epsilon` along `E_a - E_w` at one position, compared against the
one-backward prediction `epsilon * d[i,a]`. `epsilon = 1` is a real substitution.

| epsilon | Samples | Median ratio | Correlation | Mean predicted | Mean actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 256 | 0.8626 | 0.8460 | 0.00028 | 0.00035 |
| 0.05 | 256 | 1.0170 | 0.9797 | 0.00140 | 0.00150 |
| 0.10 | 256 | 1.0582 | 0.9776 | 0.00280 | 0.00317 |
| 0.20 | 256 | 1.1852 | 0.8916 | 0.00560 | 0.00769 |
| 0.50 | 256 | 1.1830 | 0.2485 | 0.01401 | 0.09361 |
| 1.00 | 256 | 1.2416 | 0.1484 | 0.02801 | 0.17889 |

Read this as the reason the gradient cannot propose mutations: it is a correct
derivative, and a substitution simply lands far outside the radius where that
derivative predicts anything.
