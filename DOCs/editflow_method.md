# EditFlow: path-aware oracle distillation for discrete scientific design

## Scope

EditFlow studies a budgeted-oracle setting rather than classical offline black-box
optimization. Experimental labels are fixed and unavailable outside the original
dataset, while an expensive computational oracle can be queried at a measured
cost. In PLS, that oracle combines a protein language model, ESMFold-derived
geometry, GVP message passing, surface-patch tokens, and confidence-aware fusion.

The target deployment model consumes only a sequence and is cheap enough for
large-scale edit search. PLS test entities remain permanently frozen: oracle
construction, edge acquisition, student training, hyperparameter selection, and
optimization evaluation use training/validation anchors only.

The method claim is deliberately narrower than Sobolev training, Jacobian
matching, MatchOpt, HodgeRank, or conservative model-based optimization. Edge
labels contain no oracle information beyond their queried endpoint values. The
contribution target is the combination of:

1. path-dependent regret bounds on a discrete edit graph;
2. graph-Sobolev distillation under exactly matched queried-node budgets;
3. regret-aware acquisition of costly oracle queries along likely design paths;
4. validation on both a derived-modality PLS oracle and a measured GB1 landscape.

## Edit graph and losses

Let `G=(V,E)` be a sequence edit graph rooted at an anchor `x0`. Each directed
edge `(u,v)` is one allowed edit. A scalar teacher `T` and student `f` induce edge
fields

```text
g_T(u,v) = T(v) - T(u)
g_f(u,v) = f(v) - f(u).
```

For `delta=f-T` and graph incidence matrix `B`, the weighted squared edge loss is

```text
L_edge = ||W^(1/2) B delta||_2^2 = delta^T L_W delta.
```

Thus `L_value + lambda * L_edge` is a discrete graph-Sobolev objective. It changes
how the same queried node values are used; it does not receive additional oracle
information. Every comparison must therefore use the identical queried-node set,
not merely the same number of edges or nominal query count.

The first controlled objective is intentionally small:

```text
L = lambda_value * L_value
  + lambda_edge * L_edge
  + lambda_conservative * L_conservative.
```

Listwise/ranking losses are ablations rather than required components.

## Path-dependent regret guarantee

For edge discrepancy

```text
epsilon_e = |g_f(e) - g_T(e)|,
```

define the minimum cumulative discrepancy from anchor `x0` to `x`:

```text
D_f^T(x) = min_{P:x0 -> x} sum_{e in P} epsilon_e.
```

Let `x_T*` maximize `T` over the allowed design set. If the returned design
`x_hat` is at most `eta` suboptimal for the student objective, then

```text
T(x_T*) - T(x_hat) <= D_f^T(x_T*) + D_f^T(x_hat) + eta.
```

Proof: along any path, signed edge errors telescope to
`delta(x)-delta(x0)`, so their absolute value is bounded by the accumulated
absolute edge discrepancy. Apply this once to `x_T*`, once to `x_hat`, and use
`f(x_T*)-f(x_hat) <= eta`. If every path has at most `k` edges and every edge
error is bounded by `epsilon`, the familiar `2*k*epsilon + eta` result follows.

Absolute value calibration is not required for local design: adding a constant to
`f` leaves every edge field and every argmax unchanged. This motivates reporting
optimization fidelity independently of value R-squared.

For two edits with teacher margin `m`, uniform edge error at most `epsilon`
preserves their order whenever `m > 2*epsilon`; an edit's sign is preserved when
`|g_T(e)| > epsilon`. These statements map directly to mutation sign accuracy,
top-k recall, NDCG, and anchor-macro Kendall tau.

## Query acquisition

Teacher cost is charged per unique queried node. A queried edge is available only
when both endpoint values have been purchased. The initial acquisition rule is

```text
A(e) = pi(e) * U(e),
```

where `pi(e)` is occupancy under current beam/local-search trajectories and
`U(e)` is ensemble uncertainty in the edge discrepancy. Edge scores are aggregated
onto unqueried frontier nodes, because querying a node reveals its teacher value
and may close several incident edges. Later variants may target the largest
contribution to the current top-candidate regret bound.

Required baselines under the same queried-node sets are random frontier sampling,
uncertainty-only sampling, occupancy-only sampling, value-KD, and a MatchOpt-style
adaptation.

## Two evaluation worlds

### PLS derived-modality oracle

Freeze one coherent full teacher and one coherent sequence-only teacher. Distill
raw logits, not calibrated probabilities. The structural residual is

```text
R_struct(x) = T_full(x) - T_seq(x).
```

Every oracle record stores teacher revisions, sequence hashes, fold/feature
revisions, confidence, failures, wall time, and accelerator time. Independent
metric-specific PLS ensembles must not be combined into a fictitious single
teacher.

### GB1 measured landscape

Use the complete four-site GB1 edit graph as an experimental landscape. Preserve
the distinction between approximately 149k measured variants and the roughly
10k variants imputed by the source publication. Primary experimental-regret
claims use measured nodes; imputed values are a separately labelled sensitivity
analysis. Query acquisition hides values according to a precommitted protocol;
the full table is used only by the evaluator, never by the acquisition algorithm.

## Leakage and budget invariants

- All mutants, trajectories, and artifacts for a PLS anchor inherit the anchor's
  strict-SI30 component and split.
- No PLS test sequence, mutant, structure, score, calibration value, or aggregate
  is queried or inspected.
- Value-KD and EditFlow receive exactly the same unique teacher-queried node IDs.
- Model selection uses validation anchors; reported GB1 protocols precommit their
  hidden evaluation sets or simulate repeated fixed seeds without adaptive reuse.
- Query budgets report unique nodes, oracle failures, retry cost, wall time, and
  GPU time rather than counting constructed edges as independent information.

## Required artifacts

```text
artifacts/oracles/<revision>/manifest.json
artifacts/landscapes/<revision>/nodes.parquet
artifacts/landscapes/<revision>/edges.parquet
artifacts/landscapes/<revision>/query_manifest.json
artifacts/landscapes/<revision>/split_audit.json

outputs/<experiment>+MM-DD-HH-MM/
  config.json
  environment.json
  oracle_manifest.json
  queried_nodes.json
  value_metrics.json
  edge_metrics.json
  ranking_metrics.json
  regret_metrics.json
  optimization_rollouts.json
  query_budget.json
  tensorboard/
  checkpoints/
```

Checkpoints, prediction arrays, and raw oracle caches remain excluded from Git.
Configs, manifests without restricted data, logs, aggregate validation metrics,
and TensorBoard histories may be versioned.

## Prior-work boundary

- MatchOpt: <https://proceedings.mlr.press/v235/hoang24a.html>
- Sobolev Training: <https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html>
- Jacobian Matching: <https://proceedings.mlr.press/v80/srinivas18a.html>
- HodgeRank: <https://doi.org/10.1007/s10107-010-0419-x>
- Conservative Objective Models: <https://proceedings.mlr.press/v139/trabucco21a.html>
- GB1 experimental landscape: <https://doi.org/10.7554/eLife.16965>

