# PLS: E. coli Protein Solubility Prediction

## 1. Goal and task definition

PLS predicts the solubility of a protein expressed in *E. coli* from its amino-acid
sequence. The system supports both forms of supervision without forcing one dataset
to imitate another:

- **Regression:** continuous solubility in `[0, 1]` (preferred for eSOL-like data).
- **Classification:** soluble/insoluble label, with the threshold stored in dataset
  metadata rather than hard-coded in the model.
- **Optional multi-task learning:** jointly predict the continuous value and class.

Every prediction should include the score, calibrated probability (for
classification), model/version metadata, and confidence/quality flags.

## 2. Design principles

1. Sequence and predicted-structure artifacts are generated once and cached.
2. Every observation preserves its dataset, original record ID, label provenance,
   assay/annotation endpoint and immutable sequence SHA-256.
3. Train/validation/test splits enforce a Biopython pairwise sequence-identity
   (SI) gap of 30%, never using naive random rows.
4. A sequence-only PLM baseline must be reported beside every structural model.
5. ESMFold confidence is an input and a gate: uncertain predicted geometry must not
   be treated as experimental truth.
6. Data preprocessing, feature extraction, training, evaluation, and inference are
   independent CLI stages.

## 3. Proposed architecture

```text
amino-acid sequence
       |
       +------------------------+
       |                        |
       v                        v
 frozen PLM encoder         ESMFold inference
 (ESM-2 initially)          -> PDB/mmCIF + confidence
       |                        |
 residue embeddings             v
       |                  structure featurizer
       |                  - residue graph
       |                  - geometric/global descriptors
       |                        |
       v                        v
 sequence projector       structure encoder
 (attention pooling)      (GVP/GNN + descriptor MLP)
       |                        |
       +-----------+------------+
                   v
          confidence-gated fusion
          [seq, struct, interaction]
                   |
                   v
       regression/classification heads
```

### 3.1 PLM branch

Start with frozen `esm2_t33_650M_UR50D` when hardware permits; use the 150M model
for smoke tests. Extract per-residue embeddings and apply a mask-aware attention
pooler. Mean pooling remains a required baseline. Initially train only the
projector/pooler/head; later experiments may unfreeze the final PLM layers or use
LoRA.

Important: ESMFold itself contains an ESM-2 representation stack. The first
benchmark should still use separately cached sequence embeddings because this
keeps baselines reproducible and lets the PLM be swapped independently. We should
explicitly label the combined system as two views derived from the same sequence,
not as independent biological evidence.

### 3.2 ESMFold structural and surface-context branch

Cache predicted coordinates and residue-level pLDDT. Build one residue graph per
protein:

- nodes: amino-acid identity, PLM residue projection (optional), pLDDT, secondary
  structure, relative solvent accessibility, backbone torsions;
- edges: sequence neighbors plus spatial neighbors within a configurable C-alpha
  cutoff / k-nearest-neighbor graph;
- edge features: distance expansion, sequence separation, relative orientation;
- global descriptors: length, radius of gyration, contact density, exposed
  hydrophobic fraction, charge distribution, secondary-structure fractions,
  disorder/low-confidence fraction and mean/quantiles of pLDDT.

The first structural encoder should be a small GVP-GNN or geometry-aware message
passing network. A descriptor-only MLP is retained as a cheap, interpretable
structural baseline.

The primary innovation target is a **mesh-free surface-patch encoder**, not merely
another C-alpha graph. Surface residues are selected by RSA/SASA, then connected
into geodesic-like patches using spatial adjacency constrained by the residue
graph. Each patch summarizes exposed hydropathy, exposed charge, charge clustering,
area/RSA, packing, curvature proxy, patch size and pLDDT. Attention pooling over
patches learns which exposed regions drive aggregation while retaining explicit
patch-level explanations. This avoids making MSMS/PyMesh/PDB2PQR/APBS/MaSIF a
mandatory inference dependency.

### 3.3 Confidence-gated fusion

Use late fusion first:

`z = concat(z_seq, g(q) * z_struct, z_seq * proj(z_struct), q)`

where `q` contains ESMFold quality summaries and `g(q)` is a learned gate bounded
to `[0, 1]`. Apply **structure dropout** during training so inference can fall back
to sequence-only prediction if folding fails or a sequence is too long. A later
phase may add residue-level cross-attention, but only after late fusion demonstrates
reliable structural gain.

## 4. Data contract

Preferred input is CSV/Parquet with a separate metadata YAML. Minimum fields:

| field | required | meaning |
|---|---:|---|
| `protein_id` | yes | stable unique identifier |
| `sequence` | yes | one-letter amino-acid sequence |
| `solubility` | one target | continuous value and original unit/scale |
| `label` | one target | soluble/insoluble label |
| `source` | yes | dataset/publication/batch |
| `host` | recommended | expression host; retain *E. coli* explicitly |
| `assay` | recommended | cell-free/cell-based and measurement protocol |
| `expression_conditions` | optional | temperature, tag, vector, etc. |
| `replicate_or_batch` | optional | leakage and batch-effect control |

Ingestion will normalize sequences, reject invalid residues according to an
explicit policy, detect duplicate/conflicting labels, and generate a manifest.
No sequence-identity deduplication is performed for model training. Similar protein
variants are retained. Exact sequence duplicates and conflicting measurements are
also retained as separate observations with source and assay provenance; optional
aggregation is an explicit ablation, never a silent preprocessing step.

## 5. Splitting and leakage control

- Do not remove proteins merely because they are identical or similar. Create one
  canonical sequence entity per SHA-256 for similarity computation while retaining
  every source observation that points to that entity.
- Use Biopython pairwise global protein alignment as the canonical similarity
  implementation. Prefer `Bio.Align.PairwiseAligner` rather than the deprecated
  `Bio.pairwise2` interface.
- Define sequence identity reproducibly as
  `identical aligned residue pairs / total alignment columns`, where gap-containing
  columns remain in the denominator and gap-gap columns are ignored. Ambiguous
  residues are identical only on exact character match.
- Build an undirected conflict graph over canonical sequence entities with an edge
  whenever pairwise SI is `>= 0.30`. Assign connected components as indivisible
  groups, then allocate components to train/validation/test while approximately
  preserving target distribution and dataset/source composition. All observations
  attached to an entity inherit the same split.
- Enforce the **Inter_Protein SI 30% Gap** as a hard invariant: the maximum SI for
  every cross-split protein pair must be `< 0.30`. Validate all train-validation,
  train-test and validation-test pairs after allocation and persist an audit table
  containing the closest cross-split neighbors.
- Report a random split only as a literature-comparison diagnostic; it is not the
  primary estimate of generalization.
- Keep proteins from the same experimental batch/source together where possible.
- Fit normalization, calibration, threshold selection, and early stopping using
  training/validation only.
- Record Biopython version, substitution/scoring mode, gap-open and gap-extension
  penalties, SI denominator rule, seed, component assignment and split manifest.

Pairwise alignment is quadratic in protein count. For datasets on the scale of
eSOL this exact audit is feasible and should be cached as a triangular SI matrix.
For substantially larger future datasets, a fast prefilter may nominate candidate
pairs, but final threshold decisions must still be made by the canonical Biopython
implementation.

## 6. Training and evaluation

### Two-endpoint weak-to-strong supervision

PDBSol and eSOL must not be merged as if their labels measured the same endpoint.
Training uses a shared encoder with separate dataset-specific heads:

1. pretrain the shared representation and binary soluble-expression head on
   PDBSol weak labels;
2. fine-tune on quantitative eSOL solubility with regression and derived
   classification heads;
3. optionally alternate PDBSol/eSOL batches with validated loss weights while
   retaining both heads.

Before any weak-label pretraining, bind exact matches and every protein violating
the Biopython SI<30% boundary with eSOL validation or test proteins to the same
held-out split. They remain in the corpus but are unavailable to the training
optimizer for that benchmark. All label-source and PDB-selection shortcuts are
excluded from model inputs. In particular, no feature measuring similarity to a
PDB or *E. coli* reference subset is allowed.

### Losses

- regression: Huber loss (MSE as an ablation);
- classification: BCE with logits; use class weights only when imbalance warrants;
- multi-task: learned uncertainty weighting or validated fixed weights.

### Metrics

- regression: Spearman, Pearson, RMSE, MAE and bootstrap 95% confidence intervals;
- classification: MCC, AUROC, AUPRC, F1, balanced accuracy, Brier score and ECE;
- report metrics by sequence length, homology bin, pLDDT bin, membrane status and
  dataset/source when metadata exists.

### Required experiments

1. physicochemical descriptor baseline;
2. PLM mean-pooling baseline;
3. PLM learned-pooling model;
4. ESMFold descriptor-only model;
5. PLM + structural descriptors;
6. PLM + structure GNN with and without confidence gating;
7. shuffled-structure and structure-dropout controls;
8. random-split versus homology-split comparison.
9. eSOL-only versus PDBSol-pretrained and joint multi-task training;
10. residue graph versus RSA/SASA versus mesh-free surface-patch pooling;
11. surface-patch feature permutation and low-pLDDT masking controls.

### Benchmark tracks

- `legacy_fgnnsol`: exactly reconstruct the public 2,019/268/392 membership for
  direct head-to-head comparison. It must not be described as SI<30% until audited.
- `strict_biopython_si30`: the primary scientific result, regenerated and verified
  using the canonical Inter_Protein SI rule in Section 5.
- `external`: PDBSol external cohorts and other expression datasets used without
  tuning to quantify endpoint/domain shift.

## 7. Repository layout

```text
PLS/
├── DOCs/                 design, dataset cards, experiment notes
├── preparation/          dataset adapters and one-off acquisition recipes
├── benchmark/            frozen split manifests and external benchmark configs
├── script/               thin CLI launchers / batch-job examples
├── configs/              data, feature, model, training and experiment YAMLs
├── data/                 raw/interim/processed manifests (large files ignored)
├── artifacts/
│   ├── plm/              sequence-hash keyed embeddings
│   ├── structures/       ESMFold coordinates and confidence
│   └── graphs/           structure graphs/descriptors
├── src/pls/
│   ├── data/             schemas, validation, split and datamodules
│   ├── features/         PLM, ESMFold and structural feature pipelines
│   ├── models/           encoders, fusion and prediction heads
│   ├── training/         losses, trainer and calibration
│   ├── evaluation/       metrics, bootstrap and reports
│   └── inference/        predictor API and CLI
└── tests/                unit and small integration tests
```

Large artifacts and raw datasets should not enter Git. Each cached artifact stores
its sequence hash, model/checkpoint revision, featurizer version and parameters.

## 8. Staged delivery

### Phase A — data audit and defensible baseline

Adapt eSOL, PDBSol and UESolDS, produce dataset cards, reconstruct the legacy
FGNNSol split, construct and audit a new Biopython SI<30% component split without
discarding similar variants, freeze provenance-rich observation and entity
manifests, then train physicochemical and frozen-PLM baselines.

### Phase B — structural pipeline

Run resumable ESMFold inference, validate structures, extract confidence-aware
descriptors/graphs, and benchmark descriptor-only and late-fusion models.

### Phase C — model refinement and tool packaging

Add geometry-aware GNN and optional cross-attention, calibrate the selected model,
export a versioned checkpoint, and expose batch FASTA/CSV CLI plus a Python API.

## 9. Dataset handoff checklist

When data arrives, determine before implementation:

1. whether each target is continuous, binary, or both, including units and cutoff;
2. whether proteins are native *E. coli* proteins (eSOL-like) or heterologous
   proteins expressed in *E. coli*;
3. assay protocol, expression conditions, tags and batch identifiers;
4. existing train/test splits and the intended comparison papers;
5. sequence IDs, duplicate policy, missing values, licensing and redistribution;
6. expected inference hardware and maximum sequence length.
