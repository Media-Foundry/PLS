# Frozen ESM-2 baseline

The first trainable model uses frozen `esm2_t33_650M_UR50D` residue
representations with exact mean pooling. Sequences longer than ESM-2's 1,022
residue limit are split into deterministic non-overlapping chunks; residue sums
and counts are combined so every residue has equal weight.

Feature extraction is resumable per canonical entity and records the entity
manifest, model checkpoint, contact-regression checkpoint, PyTorch/ROCm version,
precision and pooling policy. The initial official checkpoint SHA-256 is
`ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c`.

The prediction model contains a shared projector and three endpoint-specific
heads:

- UESolDS weak binary expression/solubility;
- PDBSol weak/composite binary solubility;
- eSOL quantitative continuous solubility.

Training samples tasks with inverse-frequency weights so the large weak datasets
do not eliminate the eSOL signal. Binary heads use BCE with logits; eSOL uses
Huber loss. This is a mean-pooling baseline, not the final surface-aware model.

All GPU execution on DiamondHill is pinned to physical device 7. The scripts
refuse CUDA/HIP execution unless `HIP_VISIBLE_DEVICES=7` is explicitly set.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=src python -m pls.features.extract_esm2 \
  --entities benchmark/generated/sequence_entities.csv \
  --output-dir artifacts/plm/esm2_t33_650M_UR50D_mean \
  --token-budget 4096 --device cuda:0 --precision float16

HIP_VISIBLE_DEVICES=7 PYTHONPATH=src python -m pls.training.train_plm_heads \
  --entities benchmark/generated/sequence_entities.csv \
  --observation-split benchmark/generated/strict_si30_observation_split.csv \
  --embedding-dir artifacts/plm/esm2_t33_650M_UR50D_mean \
  --output-dir artifacts/models/esm2_mean_strict_si30 --device cuda:0
```
