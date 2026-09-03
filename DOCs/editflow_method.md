# Cost-Aware Conformal Gating for Cached Derived-Modality Oracles

The earlier names EditFlow, Path-OLD, OLD, and intervention distillation refer
to historical stages of this same branch. They are retained only so that older
commits, configs, module paths, and `outputs/` directories remain readable. They
are not the current method name, and the sections they named are now framing and
negative controls rather than the headline claim.

## Scope

This work studies a budgeted-oracle setting rather than classical offline
black-box optimization. Experimental labels are fixed and unavailable outside the
original dataset, while an expensive computational oracle can be queried at a
measured cost. In PLS, that oracle combines a protein language model,
ESMFold-derived geometry, GVP message passing, surface-patch tokens, and
confidence-aware fusion.

PLS test entities remain permanently frozen: oracle construction, candidate
scoring, gating calibration, student training, hyperparameter selection, and
optimization evaluation use training/validation anchors only.

Development evidence has narrowed the primary question twice. First, global edge
metrics and several path-acquisition heuristics did not reliably reduce design
regret, while students could fit oracle values without preserving mutation
directions. Second, for local single edits the expensive modality can frequently
be cached from the parent instead of recomputed per mutant, which converts the
problem from reconstructing the oracle into deciding which candidates must be
evaluated exactly at all. The active method question is therefore how much
expensive computation a decision actually requires under an explicit risk
constraint, not how faithfully a cheap student reproduces the oracle. Path
acquisition and graph-Sobolev distillation are retained as regret framing and as
evaluated negative results, not as the headline algorithm.

The method claim is deliberately narrower than Sobolev training, Jacobian
matching, MatchOpt, HodgeRank, or conservative model-based optimization. Edge
labels contain no oracle information beyond their queried endpoint values. The
contribution target is the combination of:

1. the cached derived-modality oracle: recomputing the cheap modality for a
   mutant while reusing the parent's expensive predicted structure preserves
   local intervention ranking at a small fraction of the fold cost, together
   with the general observation that near-perfect node agreement does not imply
   near-perfect intervention agreement;
2. conformal decision gating: component-level marginally valid decision sets
   over the cached oracle, controlling exact or epsilon-optimal decision regret
   instead of reconstruction error;
3. a cost-aware, epsilon-optimal formulation driven by a frozen label-free
   runtime model, with direct evidence that query count materially overstates
   real accelerator savings;
4. a frozen selection/calibration/confirmation protocol in which selected-stage
   compute is measured rather than replayed, validated on a derived-modality PLS
   oracle, with the discrete edit graph supplying the path-dependent regret
   framing and a measured GB1 landscape supplying acquisition-side negative
   controls.

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

The full 1k-component multi-fidelity run has now completed. It used 3,072 exact
mutant queries (822 cached and 2,250 newly folded), with zero fold failures and
zero test queries, then scored 18,432 dense fixed-parent edges and the exact
subset with the same frozen checkpoint in float32. On the broader 128-component
validation set, fixed-parent retains edge Pearson `0.7481`, Spearman `0.6896`,
sign accuracy `0.7842`, macro Kendall `0.5850`, and top-5 recall `0.6844`.
This supports a strong local cached-backbone oracle, not a global potential:
the approximation remains conditional on the chosen parent structure.

The first prespecified correction baselines are null. Five-fold component-grouped
OOF RMSE on 1,024 train edges selects raw fixed-parent over an affine mapping,
a ridge residual using PLM/local-structure features, and a nonlinear residual.
On the one-pass exact validation report, affine preserves rank but slightly
worsens RMSE (`0.3419`), while ridge/nonlinear lower Spearman to `0.6191/0.6562`.
An ExtraTrees discrepancy model achieves only `0.2013` Spearman against absolute
validation refolding error. Selective exact refolding is no better than random
through the 20% budget point; it becomes slightly better only at 50%. Therefore
neither sparse residual correction nor selective refolding is an established
method contribution. The unmodified cached-backbone approximation is the
positive result. See `analysis/editflow_pls_multifidelity_correction_v1.md`.

## Cached derived-modality decisions

The resulting method object is not a globally corrected fixed-backbone score.
For an expensive derived modality `G(x)`, define

```text
T_H(x')     = F(h(x'), G(x'))
T_L(x' | x) = F(h(x'), G(x)).
```

`T_L` is a local cached-context oracle conditioned on its anchor, not a global
sequence potential. Near-perfect node agreement does not imply intervention
agreement: on the 1k exact subset node Pearson/Spearman are `0.9976/0.9978`,
while validation edge Spearman is `0.6896`. The scientific question is therefore
whether the cached oracle already determines the downstream decision, not how
well a regression globally reconstructs `T_H`.

A direct Top-M verification baseline on the frozen 128-component validation
report gives exact-best inclusion/zero-regret rates `0.4922`, `0.6641`, `0.8438`,
and `0.9531` at M=`1,2,4,8`; mean exact regret falls from `0.1338` at one query
to `0.0073` at eight. This is a decision funnel result, not a calibration model.

Two train-only certification baselines expose the cost of stronger guarantees.
Cross-fitted simultaneous absolute-error envelopes attain roughly 91--94%
family coverage and 99.2% decision accuracy, but require about 15 of 16 exact
queries. A component-safe conformal exact-best rank set uses 12 queries for
96.1% cross-fit coverage. A variable decision-margin set calibrates
`max Delta_L - Delta_L(exact argmax)` at the SI30-component maximum; it reaches
92.2% cross-fit coverage with 10.07/16 mean exact queries. These are marginal
risk-control statements under component exchangeability, not deterministic
per-protein certificates.

The decision-margin threshold `0.2325940132`, alpha `0.1`, endpoints, and
Top-4/8/rank-12/exhaustive baselines were frozen before constructing a new
64-anchor train-only confirmatory landscape. Those anchors use unique SI30
components excluded from every prior PLS oracle manifest. Exact confirmatory
scores were evaluated once with no threshold or endpoint change. The margin set
achieves `0.9375` exact-best coverage (60/64; two-sided 95% Clopper--Pearson
`[0.8476, 0.9827]`) using `9.70/16` exact queries on average (`0.6064` fraction),
with mean regret `0.0271`. Top-4/8 obtain coverage `0.8281/0.9375` at fixed cost
`0.25/0.50`, but mean regret is higher (`0.1107/0.0952`); rank-12 obtains
`0.9844` coverage and `0.0204` mean regret at `0.75` cost. The confirmatory
result supports variable decision-focused reuse, while the wide coverage
interval and four misses prohibit a per-anchor deterministic guarantee.

The method should therefore be called **Cost-Aware Conformal Gating for Cached
Derived-Modality Oracles**, not deterministic certification.  For low-fidelity
candidate scores `L_j` and high-fidelity maximizer `j*`, the calibration score is

```text
S(x) = max_j L_j - L_j*.
C_alpha(x) = {j : L_j >= max_k L_k - q_alpha}.
```

Under exchangeability of the SI30-component calibration units, split conformal
validity gives marginal inclusion of the high-fidelity optimum, and hence zero
decision regret, with probability at least `1-alpha`. It does not give a
deterministic per-protein guarantee. Confirmatory v1 used a frozen threshold
produced by a conservative one-order-higher NumPy quantile convention. That
threshold and result remain unchanged. New protocols directly select the
one-based order statistic `ceil((n+1)*(1-alpha))`.

Tail-risk and measured-cost replay sharpen the confirmatory interpretation.
Margin and Top-8 both miss 4/64 exact optima, but their failure-conditional mean
regrets are `0.4336` and `1.5226`; empirical CVaR95 has the same values, while
maximum regrets are `1.3061` and `4.5960`. Margin therefore reduces failure
severity rather than improving exact-best coverage. Its 621/1,024 query count
suggests a 39.4% saving, but the retained anchors are longer: recorded ESMFold
inference falls only from `3060.3` to `2236.8` GPU-seconds (26.9%), and a replay
on the original four shard assignments estimates wall time `693.0` versus the
measured exhaustive `890.9` seconds (22.2%). Only the exhaustive wall time was
directly measured; the selected wall time is a counterfactual replay. See
`analysis/editflow_pls_decision_gating_confirmatory_v1_cost_audit.md`.

## Train-only epsilon-regret and cost-aware extension

For scientific tasks where near-optimal mutations are interchangeable, define

```text
J_epsilon(x) = {j : H_j >= max_k H_k - epsilon}
S_epsilon(x) = min_{j in J_epsilon(x)} [max_k L_k - L_j].
```

Conformalizing `S_epsilon` controls the marginal event that the exact-verified
set contains an epsilon-optimal decision. A label-free length scale can trade
coverage allocation against folding cost:

```text
a_gamma(x) = (length(x) / median_calibration_length)^(-gamma)
C(x) = {j : (max L - L_j) / a_gamma(x) <= q}.
```

Five-fold development on the old 128 train anchors, grouped into 89 SI30
components and calibrated by the maximum anchor score per component, shows a
useful Pareto direction. At `epsilon=0.2, gamma=1`, component epsilon-coverage
is `0.9213`, the length-squared cost fraction is `0.3694`, and mean regret is
`0.0262`. The exact-best version `epsilon=0, gamma=1` gives component coverage
`0.9326` at cost fraction `0.6852`. These are train-only exploratory results;
the already-seen 64 confirmatory components were not used to select them and
must not be relabelled as confirmation. See
`analysis/editflow_pls_epsilon_cost_development_v1.md`.

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

## Frozen runtime-cost protocol v2

The operational objective is now explicit:

```text
minimize E[sum_{j in C(X)} c(X,j)]
subject to P(R(C;X) <= epsilon) >= 1-alpha.
```

The candidate cost is no longer query count or length squared. A label-free
weighted isotonic model is fit to per-length median marginal inference time from
5,814 historical folds across 20 homogeneous ROCm ESMFold shards. The first
successful sequence in each process is removed as compilation/warm-up, because
candidate-specific gating controls marginal work; process startup is accounted
for separately at deployment. Five-fold cross-validation grouped by complete
runtime report gives Spearman `0.9339`, median absolute error `0.0641` seconds,
and median absolute percentage error `2.97%`. The fitted knots, report hashes,
ESMFold checkpoint hash, recycle count, chunk size, and reference cost are
frozen in `configs/editflow/pls_esmfold_runtime_cost_model_v1.json`.

The score scale is

```text
a_gamma(x) = (c_hat(x) / c0)^(-gamma),
S_epsilon,gamma(x) = min_{j in J_epsilon(x)} (max L - L_j) / a_gamma(x).
```

Both `c_hat` and `c0` are frozen before conformal calibration; calibration
covariates no longer estimate a fold-specific median length. Runtime-cost
development on the old 89 SI30 components gives the following pre-selection
points:

| Policy | Component target coverage | Predicted GPU-cost fraction | Mean regret | CVaR95 |
| --- | ---: | ---: | ---: | ---: |
| epsilon=0, gamma=0 | 0.8989 | 0.6630 | 0.0153 | 0.2475 |
| epsilon=0, gamma=1 | 0.9213 | 0.6514 | 0.0081 | 0.1444 |
| epsilon=0.2, gamma=0 | 0.8876 | 0.4231 | 0.0437 | 0.5208 |
| epsilon=0.2, gamma=1 | 0.9213 | 0.3452 | 0.0257 | 0.3259 |

The `0.2` tolerance is approximately `1.021` development exact mutation-effect
IQRs, so it is a broad oracle-score tolerance rather than a biologically
calibrated solubility change. Protocol v2 therefore freezes `epsilon=0,gamma=1`
as the primary exact-best risk endpoint and `epsilon=0.2,gamma=1` as a secondary
cost--risk frontier point.

Method selection, calibration, and confirmation are disjoint. The old 89
components select the policies. A new label-blind manifest supplies 128 fresh
SI30 components only for final quantiles. A second 128-component manifest has
zero component overlap with calibration and every prior PLS oracle manifest.
The confirmatory order is fixed-parent scoring, label-free candidate selection,
selected-only exact folding and deployment freeze, then retrospective folding
of unselected mutations. This makes selected-stage GPU-seconds and wall time
directly measured rather than replayed. Length quartiles `[106,148,219]` and
parent-pLDDT strata `[0.7,0.8,0.9]` are predefined descriptive diagnostics;
the formal guarantee remains marginal over exchangeable SI30 components.
