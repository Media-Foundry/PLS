# Core PLS frozen pretest state — 2026-09-01

Core predictor development is frozen at code revision
`2ff7afbb3ddddcc78d96038ba106134bd8b8a407`. The machine-readable manifest is
`configs/frozen_pretest_state_2026-09-01.json`; it seals 37 recursively selected
validation reports, 80 selected run/candidate directories, 74 available best
checkpoints, run configs and seeds, feature revisions, split revisions, and the
ESMFold checkpoint.

This is a pretest provenance freeze, not a test evaluation. No test entity,
prediction, target, aggregate, calibration value, or model-selection result was
read or produced. Full immutable split hashes seal test membership without
creating a new test-only list.

## Frozen development metrics

These values are the result of extensive strict-validation development and must
be described as development performance rather than expected unseen-data
performance.

| task | metric-specific recipe | strict-validation value |
| --- | --- | ---: |
| PDBSol | AUROC | 0.9046065494 |
| PDBSol | AUPRC | 0.9161942122 |
| PDBSol | MCC | 0.6425591645 |
| PDBSol | SI30-component OOF Platt Brier | 0.1259196054 |
| UESolDS | AUROC | 0.8076030571 |
| UESolDS | AUPRC | 0.8686877438 |
| UESolDS | MCC | 0.4554435366 |
| UESolDS | SI30-component OOF beta Brier | 0.1744964382 |
| eSOL | Spearman | 0.7720613126 |
| eSOL | Pearson | 0.7642754765 |
| eSOL | RMSE | 0.2096029381 |
| eSOL | MAE | 0.1646387926 |
| eSOL | SI30-component OOF affine RMSE | 0.2094082855 |

The former eSOL affine value `0.2075887243` was fitted and evaluated on the same
267 validation entities. It remains in the manifest as
`validation_fitted_affine_rmse`, not as the generalization-facing affine metric.

## Metric-specific deployment recipes

The canonical recipe file is `configs/validation_selection_v1.json`, SHA-256
`885100e5d2f274aa48eb1461f8bd1b18e7069d8f7a1d027213db75a4a2c10b9d`.
All ensemble weights and component paths are frozen transitively in that file and
its content-addressed selected reports.

PDBSol:

- AUROC report: `latent_endpoint_residual_v35_pdbsol.json`;
- AUPRC report: `pdbsol_spatial_patch_v18_ensembles.json`;
- MCC report: `pdbsol_gvp_multiseed_ensembles.json`;
- MCC threshold: `0.44192126393318176`;
- grouped Platt report:
  `latent_endpoint_residual_v35_pdbsol_platt_si30_grouped.json`;
- AUROC/AUPRC temperature: `1.3404120878780323`;
- MCC temperature: `1.3166465790461879`.

UESolDS:

- AUROC report: `uesolds_hard_pair_v43_auroc.json`;
- AUPRC report: `uesolds_v41_weighted_auprc.json`;
- MCC report: `uesolds_hard_pair_v43_mcc.json`;
- MCC threshold: `0.494721706888232`;
- grouped beta report:
  `uesolds_hard_pair_v43_auroc_beta_si30_grouped.json`.

eSOL uses separate frozen recipes for Spearman, Pearson, RMSE and MAE. The
grouped affine diagnostic is based on the Pearson recipe. Its deployment mapping,
fitted on all validation entities only after OOF reporting, is:

```text
y_affine = 1.1550794511109763 * prediction - 0.0570579937680418
```

Metric-specific recipes must not be compared or selected on test.

## Correctness repairs included in the freeze

- Empty surface-component sets now yield exact zero surface output and bypass
  both patch/residue attention and pooled fusion. The selected PDB surface
  candidate contained no empty validation entity (`0/6429`), so its stored
  validation prediction is unchanged.
- Cross-attention confidence is applied once by default. The older v49 oracle
  checkpoint intentionally used power two; its manifest now states that behavior
  explicitly so old metrics remain reproducible.
- Patch spatial self-edges are disabled by default. The older v49 oracle
  checkpoint explicitly retains them for reproducibility.
- Platt, beta and isotonic cross-fitting use SI30 components as indivisible
  groups. PDBSol has 4,825 validation components and UESolDS 6,363.
- eSOL affine cross-fitting uses 262 SI30 components across 267 validation
  entities.

The full correctness record is
`analysis/core_pls_freeze_correctness_audit_2026-09-01.json`.

## Data, fold and feature seals

| artifact | SHA-256 |
| --- | --- |
| strict entity split | `1dfc29d4d3a5a333235d72f42335b4b7fd6b869c7522529f4690e48fe9ea764c` |
| strict observation split | `987281e064af96f0fccc862017d4cbcb643cb0066081a3a357963a1fccc9c8e7` |
| sequence entity manifest | `7ae6549a4c617a11ebf094c4ff04cdea501d9f0a6ff09aac88ab1ddd1077dd47` |
| ESM-2 mean embedding array | `703a0c96553b039403270a148bd2131bf0dc594bf0c540c57cd880767e8206d7` |
| structure V4 train statistics | `a403024d29fdde67969d8c80d80fb802136abc8cbf503ec77d118d9067f4d1b8` |
| compact structure metadata | `63f9f5dc816493ff89afb20c9e20dccb483384e095544ce6f65d3ab7bffa6b60` |
| geometry metadata | `c3e2307fbd909844cc039789aadcddb2e282a994b8ae16b5fc8cc9076e54bdc4` |
| residue ESM PCA metadata | `98f484010b27a4f9020a8bf98ec6a9e2df3db45c9bc3476e3722cb4f80b65aec` |
| GVP vector metadata | `0f7f6f16775f95904f46f2bbe482456b68e7a766f5d9b7ea0a256c22d1cb734b` |
| surface component metadata | `7a94bc2496a83cd5965caa9fab8632e0563e6a8c5ff552ecf21fa177bf30b5a3` |
| sequence descriptor metadata | `b122244c55d7f020b2c910b192d4e13acc49e29aec0d5a16ff43120e7722d4cb` |
| ESMFold v1 checkpoint | `e9a52579027e77d2d2e0a18218e755821f395730e86624cab9413dc117f5ca62` |

The six selected directories without their own best checkpoint are exported
endpoint candidates under latent-endpoint parent runs, not independently trained
models. Their prediction artifacts and parent provenance are sealed through the
selected report graph; the manifest does not fabricate checkpoint identities for
them.

## Scientific interpretation

The core model is frozen as a strong, expensive scientific oracle. Surface/GVP,
Physchem MoE, hard-pair ranking and latent endpoint candidates are retained as
biological inductive biases or residual-diversity contributors, not claimed as a
new dominant architecture.

The latent endpoint model is scale-unidentified. When endpoint residuals are
enabled it is correctly described as “shared latent + endpoint-specific
residual,” not as a strict monotonic observation of one identifiable intrinsic
solubility property.

## Change-control policy

Permitted changes on the frozen core line are documentation, provenance audits,
test-free reproducibility checks and manuscript work. Any new predictor idea,
ensemble modification, calibration-family choice, threshold search, or feature
change starts a separately named PLS-v2/methods branch and cannot enter this
frozen submission.

The next ML-method contribution is EditFlow/Path-OLD: distilling and optimizing
this expensive oracle under explicit query budgets. Core validation decimals are
no longer an optimization target.
