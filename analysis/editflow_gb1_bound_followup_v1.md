# GB1 bound-aware follow-up v1

This is an exploratory post-hoc arm, not a confirmatory result. Negative
paired differences favor bound-aware acquisition.

| endpoint | comparison | bound mean | baseline mean | difference | 95% CI | W/T/L |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| query_curve | bound_minus_path | 1.615822 | 1.636749 | -0.020927 | [-0.053580, 0.007252] | 6/7/3 |
| query_curve | bound_minus_uncertainty | 1.615822 | 1.683996 | -0.068174 | [-0.284744, 0.157380] | 10/0/6 |
| final_640 | bound_minus_path | 0.587609 | 0.601494 | -0.013884 | [-0.117582, 0.083631] | 6/7/3 |
| final_640 | bound_minus_uncertainty | 0.587609 | 0.849958 | -0.262349 | [-0.507342, -0.019431] | 12/0/4 |

All p-values in the JSON artifact are descriptive. No PLS test split was evaluated.
