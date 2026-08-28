# Strict Biopython SI30 split

The strict component-level split was generated on 2026-08-28 with seed 20260828
and target fractions 75% train, 10% validation and 15% test. The 18,954-entity
giant component is fixed wholly to train.

| split | entities | observations | eSOL observations |
|---|---:|---:|---:|
| train | 99,518 | 109,047 | 2,009 |
| validation | 13,302 | 14,506 | 267 |
| test | 19,961 | 21,755 | 403 |

The allocator balances entity and observation counts, dataset source, binary
source-specific positive/negative labels, and ten bins of the continuous eSOL
target. Components are indivisible. Similar or repeated observations are not
deleted.

All 3,575,788,456 cross-split pairs were checked against the cached exhaustive
matrix. There are zero cross-split threshold-edge violations and the maximum
cross-split SI is 0.29992044, strictly below 0.30.

Generated manifests remain excluded from Git under the project data policy. Their
frozen hashes are recorded in `benchmark/strict_si30_split_report.json`; the
independent matrix validation is recorded in
`benchmark/strict_si30_validation_report.json`.
