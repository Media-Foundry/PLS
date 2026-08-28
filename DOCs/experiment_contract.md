# Experiment output contract

Every training run is created under:

`outputs/<experiment_name>+MM-DD-HH-MM/`

Required contents are:

- `config.json`: exact experiment configuration copy;
- `environment.json`: timestamp, command, Git revision, HIP visibility,
  PyTorch/ROCm and relevant package versions;
- `checkpoints/epoch_NNN.pt`: periodic model and optimizer state;
- `checkpoints/best.pt`: checkpoint selected only on validation metrics;
- `tensorboard/`: TensorBoard event files;
- `history.json`: machine-readable per-epoch metrics;
- `validation_metrics.json` and `test_metrics.json`: endpoint-specific final metrics;
- `output.log`: stdout/stderr, including failures.

Failed runs remain in place as provenance. GPU runs must explicitly set
`HIP_VISIBLE_DEVICES=7`. Generated outputs are excluded from Git; configs, runner,
model code and documentation are versioned.

Dependency installation follows the dry-run and `--no-build-isolation` contract
in the repository root `AGENTS.md`.
