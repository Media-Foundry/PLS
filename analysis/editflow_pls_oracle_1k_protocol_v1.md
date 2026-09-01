# PLS 1k-component intervention landscape protocol

This label-blind development protocol selects 1,024 train and 128 validation
anchors, with exactly one anchor per SI30 component. It contains 18,432 exact
single-mutant edges and 19,584 exact sequence nodes. It queries no test sequence
and uses no target value during selection.

The previous 128-entity scale covered only 89 train components. One existing
entity from each reusable component is prioritized in the new deterministic
selection, preserving 89 train and 32 validation anchor landscapes.

The initial exhaustive plan would have required 16,496 additional mutant folds.
A complete 2,560-edge exact-reference ablation showed that a zero-refold
fixed-parent oracle already preserves exact-refold edge Spearman `0.6967` and
mutation signs `78.8%`. The active protocol is therefore multi-fidelity:

- dense fixed-parent structure plus exact mutant PLM features on all 18,432 edges;
- one exact mutation per train component (1,024 train edges);
- all 16 exact mutations per validation component (2,048 validation edges).

The exact calibration/evaluation subset contains 3,072 folds; 822 were already
cached when frozen, leaving 2,250 new folds. This reduces the remaining exact
fold workload by 86.4%. Four balanced shards run on authorized local GPUs 0--3.
The final fixed, exact-subset, and PLM artifacts are still estimated at roughly
15--20 GiB because residue ESM2 dominates storage rather than PDB files.

`pls_1k_multifidelity` is a resumable background pipeline. After the sparse
exact-fold reports it builds exact calibration V4 caches, full fixed-parent
V4/GVP/surface caches, exact-sequence mean/residue ESM2 once, and two coherent
float32 scores: dense fixed-parent and sparse exact-refold. Subset PLM tensors
are copied by exact sequence hash rather than recomputed. Checkpoints and
per-node predictions remain excluded from Git.
