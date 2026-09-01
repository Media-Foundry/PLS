# PLS EditFlow oracle proof of concept

This is development evidence from 24 strict train/validation anchors, 384
single-mutant edges, and 408 unique exact-sequence queries. No test entity was
queried or evaluated.

The canonical oracle artifact is
`artifacts/oracles/pls_editflow_poc_v1/scores_matched_fp32_a`. It uses one frozen
v49 checkpoint, float32 inference, exact mutant folds/features, and a matched
sequence ablation from the same checkpoint's sequence projection and sequence
head. An immediate second float32 replay differed by at most `2.87e-6` in the
full logits and was bit-exact in the sequence-only logits. Earlier BF16 scores
are exploratory because repeated full logits differed by as much as `0.053`.

## Matched full versus sequence-only oracle

| Scope | Edges | mean abs full delta | mean abs sequence delta | mean abs structural residual delta | delta Spearman | sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 256 | 0.1844 | 0.0464 | 0.1660 | 0.3489 | 0.6211 |
| Validation | 128 | 0.2583 | 0.0533 | 0.2412 | 0.2431 | 0.6016 |
| All | 384 | 0.2090 | 0.0487 | 0.1911 | 0.3053 | 0.6146 |

Across nodes, full and matched sequence-only logits have Pearson correlation
`0.6391`. The full logit standard deviation is `4.3499`, the sequence-only
standard deviation is `1.0303`, and the matched residual standard deviation is
`3.7755`. Thus the frozen teacher has a large independent structure/fusion
contribution, especially in its local mutation field; this application is not
a nearly sequence-only oracle disguised as a multimodal one.

This does not establish biological mutation-effect validity. A few local deltas
are very large, particularly on medium-confidence validation anchor 23, and require
confidence-stratified sensitivity analysis. The artifact supports a distillation
question, not a claim that the teacher is an experimental mutation oracle.

## Parent-confidence sensitivity

| Parent mean pLDDT | Anchors | Edges | mean abs full delta | mean abs sequence delta | mean abs residual delta | delta Spearman | sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<0.5` | 2 | 32 | 0.1942 | 0.0886 | 0.1371 | 0.7152 | 0.7812 |
| `0.5–0.7` | 5 | 80 | 0.3752 | 0.0757 | 0.3574 | 0.1857 | 0.5000 |
| `>=0.7` | 17 | 272 | 0.1619 | 0.0361 | 0.1485 | 0.3383 | 0.6287 |

The largest instability is concentrated in the medium-confidence group, which
contains anchor 23, rather than monotonically increasing as confidence falls.
Crucially, the high-confidence subset still has a residual delta more than four
times the matched sequence-only delta (`0.1485` versus `0.0361`). The structural
signal therefore survives the confidence sensitivity check, although robust or
confidence-weighted landscape targets remain worth studying.

The 384 exact mutant folds had zero failures. Their measured per-query ESMFold
inference time averaged `2.99 s` (median `2.41 s`, p95 `6.60 s`); summed shard
wall/GPU occupancy including eight model loads was `1571.2 GPU-s`. This is the
costly derived-modality work that a sequence-only student is intended to avoid.
