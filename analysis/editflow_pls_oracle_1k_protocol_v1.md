# PLS 1k-component intervention landscape protocol

This label-blind development protocol selects 1,024 train and 128 validation
anchors, with exactly one anchor per SI30 component. It contains 18,432 exact
single-mutant edges and 19,584 exact sequence nodes. It queries no test sequence
and uses no target value during selection.

The previous 128-entity scale covered only 89 train components. One existing
entity from each reusable component is prioritized in the new deterministic
selection, preserving 89 train and 32 validation anchor landscapes. This reuses
1,936 exact mutant folds; 16,496 new exact mutants remain.

Four LPT shards run on authorized local physical GPUs 0--3. After exact-cache
reuse they contain 4,130, 4,125, 4,146, and 4,095 new folds. Their remaining
length-squared costs differ by less than 1%, so the cache does not materially
unbalance execution. Based on the completed 128-anchor campaign, folding is
estimated at roughly 3--4 wall hours and the final complete artifact tree at
15--20 GiB. The workspace had about 2.4 TiB free at launch.

`pls_1k_postfold` is a resumable background pipeline. After four successful fold
reports it runs exact V4 extraction, compact/GVP/vector/surface construction,
mean and residue ESM2 caches, one coherent float32 full/matched-sequence teacher,
and six identical-node student controls: raw potential, PLM potential,
parent-state direct delta, exact-pair delta, cycle delta, and structural-residual
potential. Checkpoints and per-node predictions remain excluded from Git.
