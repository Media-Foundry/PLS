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

Development evidence has narrowed the primary question. Global edge metrics and
several path-acquisition heuristics did not reliably reduce design regret, while
PLS students could fit oracle values without preserving mutation directions.
The active method question is therefore how to distill an expensive scientific
oracle for sequence interventions rather than predictions. Path acquisition is
retained as a downstream evaluation setting, not assumed to be the headline
algorithm.

The method claim is deliberately narrower than Sobolev training, Jacobian
matching, MatchOpt, HodgeRank, or conservative model-based optimization. Edge
labels contain no oracle information beyond their queried endpoint values. The
contribution target is the combination of:

1. path-dependent regret bounds on a discrete edit graph;
2. graph-Sobolev distillation under exactly matched queried-node identities;
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
information. Objective/loss comparisons must therefore use the identical
queried-node set, not merely the same number of edges or nominal query count.
Acquisition comparisons instead hold the initial node set, candidate universe,
unique-node budget, and cost model fixed; their purchased node identities are
expected to differ.

The first controlled objective is intentionally small:

```text
L = lambda_value * L_value
  + lambda_edge * L_edge.
```

Listwise/ranking losses are ablations rather than required components.
Conservatism currently belongs to the deployment/search objective; a
conservative student-training loss has not been implemented and is not claimed.

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
`f(x_T*)-f(x_hat) <= eta`. If every feasible design has an allowed path from
`x0` of length at most `k`, and every edge error on those paths is bounded by
`epsilon`, then `D(x) <= k*epsilon` and the familiar
`2*k*epsilon + eta` result follows. The graph may contain arbitrarily long or
cyclic paths; the result requires existence of a short allowed path, not that
every possible path is short.

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
`U(e)` is ensemble uncertainty in the predicted edit effect. Edge scores are aggregated
onto unqueried frontier nodes, because querying a node reveals its teacher value
and may close several incident edges. Later variants may target the largest
contribution to the current top-candidate regret bound.

Ensemble uncertainty is not a proved upper bound on the unknown field error
`epsilon_e`. Accordingly, `pi(e) * U(e)` is a heuristic proxy, not an estimator
that is claimed to minimize the theorem's regret upper bound. Calibrating a
high-probability edge-error envelope on cross-fitted closed edges is a future
algorithmic hypothesis, not an implemented result.

For heterogeneous computational oracles, acquisition accounting uses measured or
predicted positive node costs and ranks the current proxy value per unit cost.
The repository now contains a deterministic cost-aware frontier primitive with
an exact cumulative-cost cap. This is a greedy engineering baseline, not an exact
knapsack solution and not yet wired to the PLS folding campaign; it must not be
described as expected regret-bound reduction until that estimator is implemented.

The bound-aware variant makes that last step explicit. Each ensemble member and
the conservative ensemble objective proposes an optimizer endpoint. Among
beam-discovered routes to every unique proposal, it retains the route minimizing
the cumulative ensemble edge uncertainty, an empirical proxy for `D_f^T(x)`.
Edges are queried in proportion to their uncertainty contribution and occupancy
across those shortest-bound routes. This is distinct from weighting every beam
prefix equally and is evaluated as a separate acquisition ablation.

Required acquisition baselines under the same initial set, candidate universe,
cost model, and unique-node budget include random frontier, greedy predicted
fitness, uncertainty-only, occupancy-only, UCB, Thompson sampling, and a
MatchOpt-style adaptation. Value-KD versus graph-Sobolev objective ablations use
identical final queried-node identities.

## Three non-confounded design regrets

For a feasible set `X`, purchased node set `Q_B`, and student `f`, report all of:

```text
R_acquired = max_{x in X} T(x) - max_{x in X intersect Q_B} T(x)

x_new = argmax_{x in X minus Q_B} f(x)
R_novel = max_{x in X minus Q_B} T(x) - T(x_new)

R_campaign = max_{x in X} T(x)
             - max(max_{x in X intersect Q_B} T(x), T(x_new)).
```

`R_acquired` measures what the queries directly discovered, `R_novel` measures
surrogate generalization to an unpurchased design, and `R_campaign` measures the
end-to-end scientific campaign. The previous all-candidate student regret mixed
direct acquisition with distillation whenever the student selected a queried
optimum and is retained only as historical exploratory evidence.

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

The first sequence-native parameterizations are deliberately separated:

```text
potential:          f_theta(x)
direct edit:        h_theta(parent, position, target)
exact-pair edit:    h_theta(parent, mutant)
residual potential: T_seq(x) + r_theta(x).
```

The residual potential uses the same checkpoint's sequence branch as a frozen
base and learns only `T_full - T_seq`. Unlike independently trained full and
sequence teachers, this difference has a controlled structure/fusion
interpretation. Direct edit heads are evaluated with edge metrics; value metrics
constructed by adding a known validation-parent teacher value are labelled
anchored diagnostics and never compared with fully sequence-native potential
values.

For commuting edits at distinct positions, a direct field may additionally use

```text
h_i(x) + h_j(x_i) = h_j(x) + h_i(x_j)
```

as a zero-extra-query cycle-consistency regularizer. This constraint encourages
a conservative field but is not presumed beneficial without empirical evidence.

### GB1 measured landscape

Use the complete four-site GB1 edit graph as an experimental development
landscape. Preserve
the distinction between approximately 149k measured variants and the roughly
10k variants imputed by the source publication. Primary experimental-regret
claims use measured nodes; imputed values are a separately labelled sensitivity
analysis. Query acquisition hides values according to a precommitted protocol;
the full table is used only by the evaluator, never by the acquisition algorithm.
Because the complete table has already been repeatedly inspected through
evaluation, GB1 is not a blind confirmatory test. External landscapes require a
frozen, zero-tuning transfer protocol.

The intended external landscape is FLIP2 TrpB, whose official release documents
228,298 measured variants across ten sub-landscapes and provides one-to-many,
two-to-many, and by-position splits under CC-BY 4.0:
<https://flip.protein.properties/>. No FLIP2 test row has been downloaded,
parsed, scored, or evaluated in this repository. Data ingestion remains deferred
until the acquisition method, baseline set, cost model, and admissible non-test
evaluation protocol are frozen consistently with the permanent test prohibition.

## Leakage and budget invariants

- All mutants, trajectories, and artifacts for a PLS anchor inherit the anchor's
  strict-SI30 component and split.
- No PLS test sequence, mutant, structure, score, calibration value, or aggregate
  is queried or inspected.
- Value-KD and EditFlow objective comparisons receive exactly the same unique
  teacher-queried node IDs.
- Acquisition comparisons share initial IDs and exact cost budgets, while their
  subsequent queried IDs differ by design.
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

## Evidence ledger (2026-08-31)

The first same-node-budget control found no benefit from adding the graph-Sobolev
edge objective to Value-KD. Exact-node ensemble replay after active acquisition
also produced identical selected optima with and without the edge loss. The edge
objective is therefore an ablation and mathematical interpretation, not an
empirically supported novelty claim at this stage.

An audit of the original all-candidate regret found a decisive confound: on the
single-WT exploratory run, path acquisition purchased node 143317 before its
reported radius-1 zero regret and node 143304 before its radius-2 zero regret.
Those outcomes demonstrate acquisition success, but do not by themselves
demonstrate student landscape generalization. All new runs therefore report
acquired, held-out novel-design, and campaign regret separately. Historical
all-candidate regret remains labelled exploratory and is not used as a
distillation claim.

The frozen 16-anchor GB1 comparison used Value-KD for both acquisition methods.
Its prespecified primary endpoint—mean exact regret across budgets 160/320/640
and radii 1/2/3/4—was not significant: path-aware 1.6367 versus uncertainty-only
1.6840, paired difference -0.0472, bootstrap 95% CI [-0.2801, 0.1832], exact
sign-flip p=0.704. The prespecified secondary final-640 endpoint favored
path-aware acquisition (0.6015 versus 0.8500; difference -0.2485; bootstrap 95%
CI [-0.4829, -0.0372]; exact p=0.044), but this does not rescue the null primary
endpoint.

Mechanistically, the benefit was local: at 640 queries, path-aware acquisition
reduced mean regret at edit radii 1 and 2 but increased it at radius 4. It also
had slightly lower global R-squared. The first path round spent an average 77.6
of 80 queries on path-targeted nodes, exposing an exploitation/coverage tradeoff.
The post-hoc shortest-bound follow-up was effectively tied with ordinary path
acquisition. A fixed 50/50 path-versus-frontier-uncertainty portfolio was tested
as an explicitly exploratory ablation and is not part of the confirmatory claim.

That 50/50 portfolio produced a large post-hoc improvement on the v1 anchors,
but it did not replicate on 16 newly frozen v2 anchors. The v2 prespecified
primary query-curve regret was 1.4659 for hybrid versus 1.5321 for pure path
(difference -0.0662, bootstrap 95% CI [-0.2065, 0.0760], exact p=0.385; 9 wins
and 7 losses). At the secondary 640-query endpoint, hybrid was numerically worse
(0.6484 versus 0.5826). A universal fixed exploration fraction is therefore not
an established contribution. Anchor-adaptive allocation based on optimizer-path
concentration or ensemble disagreement remains a new hypothesis requiring a
separate unseen-anchor protocol.

The same-protocol GB1 development baseline matrix now includes random, greedy,
UCB, Thompson sampling, and occupancy-only acquisition, with acquired,
novel-design, and campaign regret reported separately across all budgets. At
radius 2, occupancy-only had the lowest normalized budget-curve novel-design
regret (1.8662) and campaign regret (0.9998); UCB was more competitive at radius
4. These are descriptive starting-point comparisons within one shared four-site
landscape, not independent biological replicates or population-level evidence.
They motivate adaptive exploitation/coverage allocation but do not establish
that an adaptive policy wins.

The development branch implements two validity-guarded primitives for that next
step. Path concentration is summarized by normalized occupancy entropy,
effective support, and endpoint consensus. A deterministic candidate policy
uses those diagnostics to allocate more path budget only when paths are
concentrated and ensemble endpoints agree, with UCB retaining the exploration
budget. Separately, an additive split-conformal edge-error envelope is available,
but it explicitly requires held-out or cross-fitted edge errors. In-sample
closed-edge residuals are not evidence for coverage, and raw ensemble standard
deviation is still not described as a theorem-valid error upper bound.

The first adaptive path/UCB candidate has now completed on all 16 GB1 v1
development anchors. It achieved the highest final global R-squared among the
six standard/development policies (0.4743), and improved radius-4 campaign
regret over occupancy-only (0.9940 versus 1.8637), but it did not improve the
local radius-2 objective: normalized novel-design regret AUC was 2.2114 versus
1.8662 for occupancy-only and 1.9502 for UCB. Its realized path fraction varied
only from 0.369 to 0.559 (mean 0.487), so the entropy/endpoint formula behaved
much like a softened 50/50 portfolio. This is a null result for the proposed
adaptive allocation, not a method win. It strengthens the case for calibrated,
design-relevant edge-error reduction rather than another fixed-mixture sweep.

The uncertainty-only v2 run completed after the prespecified hybrid-versus-path
comparison. A secondary descriptive comparison on the historical all-candidate
metric favored path over uncertainty (query-curve means 1.5321 versus 1.8824;
final-640 means 0.5826 versus 1.2066). This comparison was not a prespecified
primary endpoint and the metric is now known to mix acquisition with student
generalization, so it cannot establish the method claim. Final checkpoints are
being re-audited under the three-regret decomposition.

For the PLS world, the test-free proof-of-concept contains 24 strict
train/validation anchors, 384 single-mutant edges, and 408 unique sequence
queries. All 384 exact mutant folds and V4/GVP/surface artifacts completed with
zero failures on authorized local GPUs 0--3; parent structure artifacts were
never substituted for mutants. The previously queued star Slurm chain was
cancelled while pending and remains only as a verified fallback staging copy.

Canonical oracle scoring now uses float32 inference. Two immediate replays
agreed within `2.87e-6` in full logits and exactly in the same-checkpoint
sequence-only ablation; BF16 replay drift had reached `0.053` and is only
exploratory. The matched ablation exposes substantial structure/fusion signal:
across 384 mutation edges, mean absolute full delta is `0.2090`, matched
sequence-only delta is `0.0487`, residual delta is `0.1911`, delta Spearman is
`0.3053`, and signs agree on only `61.5%` of edges.
This is not solely a low-confidence artifact: among 17 anchors with parent mean
pLDDT at least 0.7, mean absolute residual delta remains `0.1485` versus
sequence-only delta `0.0361`. The five medium-confidence anchors are less stable
and contain the largest outlier, so confidence-stratified metrics are mandatory.

The first controlled PLS student comparison is also null for naive edge loss.
With identical nodes, architecture, seed, and edge-RMSE model selection, Value
KD obtains edge Spearman `0.1171` and sign accuracy `0.5391`; adding unit-weight
graph-Sobolev loss obtains `0.0777` and `0.5312`. Both select epoch 1. This
strengthens the cross-world conclusion that the edge objective alone is not the
method contribution. Large teacher deltas on medium-confidence structures remain
a required sensitivity analysis before biological interpretation.

The next 128-train-anchor scaled experiment confirms that independent-protein
coverage matters. Frozen-PLM potential edge Spearman rises from `0.1028` to
`0.3297`; raw-token potential reaches only `0.1283`. Parent-only direct delta and
cycle delta remain null (`0.0795` and `0.0822`), while an exact parent-mutant PLM
pair improves to `0.2887` but still trails the scalar potential. The strongest
learned parameterization is the matched sequence teacher plus a learned
structural residual potential: edge Pearson `0.4159`, Spearman `0.3727`, sign
accuracy `0.6563`, and RMSE `0.6208`. It improves magnitude fidelity over both
the plain PLM potential and matched sequence teacher, but does not beat the
matched sequence teacher's rank/sign/top-k fidelity (`0.4279`, `0.6855`, and
`0.5313`). See `analysis/editflow_pls_student_scale_v1.md`.

A component audit found that those 128 train entities represent only 89 unique
SI30 components. The next protocol therefore selects exactly 1,024 unique train
components and 128 unique validation components, with 16 exact mutations per
anchor (18,432 edges; 19,584 nodes). Selection is label-blind, test query count
is zero, and one representative from each reusable prior component is
prioritized. This protocol tests biological coverage rather than merely scaling
correlated observations.

An exact-reference ablation then changes the cost conclusion. Reusing the
parent's complete structural tensor while retaining exact mutant ESM2 tokens
requires zero mutant folds yet reaches exact-refold edge Spearman `0.6967`, sign
accuracy `0.7875`, and anchor-macro top-5 recall `0.6950` across all 2,560
reference edges. Validation-only values are `0.7692`, `0.8281`, and `0.7125`.
Thus exhaustive mutant folding is not justified for dense supervision. The 1k
protocol uses the fixed-parent oracle on every edge, one value-blind exact edge
per train component for correction, and all exact validation edges for final
field evaluation. This is a multi-fidelity oracle design; fixed-parent scores
must not be labelled exact ESMFold outputs. See
`analysis/editflow_pls_fixed_parent_oracle_scale_v1.md`.

The first prequentially calibrated path acquisition is also a development null
on the primary local curve. It calibrates later-round uncertainty only from
pre-query predictions on frontier edges whose targets were subsequently
purchased, so it consumes no extra labels and never inspects unpurchased target
fitness. Radius-2 novel-design regret AUC is `1.9816`, between UCB (`1.9502`)
and the prior adaptive candidate (`2.2114`) but worse than occupancy-only
(`1.8662`). It has the highest final edge Spearman (`0.3671`) and a strong final
campaign endpoint, yet these secondary observations do not constitute a method
win. The additive corrections are large enough that the policy becomes a
support-limited occupancy/UCB hybrid; future calibration must be path-conditional
or improve candidate-path coverage rather than apply another global quantile.
