# Solubility model research notes

## Evidence-backed design direction

The frozen ESM-2 mean baseline establishes sequence performance, but related
methods support explicit multimodal fusion:

- [ProtSolM](https://arxiv.org/abs/2406.19744) combines ESM-2 sequence context,
  roto-translation-equivariant backbone encoding, attention pooling and global
  physicochemical/structural descriptors.
- [FGNNSol](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02262) combines language-
  model residue features, explicit 3D graph features and global physicochemical
  features through fused graph networks.
- [PatchProt](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525051/) shows that local
  surface accessibility and hydrophobic-patch auxiliary tasks can improve hard
  global protein-property predictions.
- [Protein-Sol](https://pmc.ncbi.nlm.nih.gov/articles/PMC5870856/) provides a
  required low-capacity reference using composition, charge, hydropathy, length,
  fold propensity and sequence entropy.

Two distinct local structural feature sources must not be conflated:

1. The ProtSolM release contains 44 protein-level descriptor columns for all
   64,598 PDBSol records: composition, GRAVY, DSSP composition, hydrogen bonds,
   20 cumulative RSA exposure thresholds and pLDDT. Names are unique and match
   the label tables exactly; there are no NaN/Inf values. However, 456
   structure-derived lengths differ from released label sequence lengths.
2. `/home/pc/Code/BIO/protein/extract_complete_features_v4.py` describes a much
   richer residue-level representation: 62 physicochemical scalars, 89 spatial
   scalars and eight 3D vectors per residue, plus coordinates and residue metadata.
   Its 89 spatial channels include a normalized/inverted B-factor confidence
   derivative. A proposed 90th spatial scalar should explicitly retain raw
   `pLDDT/100`; it must not be assumed to exist in the current V4 tensor.

The referenced V4 entrypoint currently fails in its migrated location because its
`preparation.protein` import root is absent. PLS should port the implementation
behind a tested local module with a frozen 90D schema instead of calling the
absolute external path at training time.

The 44D global descriptors should first be evaluated as a PDBSol-only controlled ablation.
Using their availability mask in the joint source-specific model would leak source
identity because full coverage currently exists only for PDBSol. The production
fusion model should consume uniformly generated ESMFold descriptors for every
eligible entity, use confidence gating and structure dropout, and retain a
sequence-only fallback.

## Baseline and adapter experiment

Strict-SI30 test results:

| model | eSOL Spearman | eSOL Pearson | PDBSol AUROC | UESolDS AUROC |
|---|---:|---:|---:|---:|
| ESM-2 mean v1 | 0.6605 | 0.6501 | 0.8428 | 0.7575 |
| task adapters v2 exploratory | 0.6893 | 0.6908 | 0.8071 | 0.7288 |

The adapter model improves quantitative eSOL while degrading both weak binary
endpoints. It is a Pareto point, not a universal replacement. Follow-up experiments
should separate weak pretraining from eSOL adapter fine-tuning and report every
endpoint rather than selecting on one pooled loss.
