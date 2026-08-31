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

EditFlow runs replace endpoint-only evaluation with optimization-facing artifacts:

- `oracle_manifest.json`: immutable teacher/checkpoint/feature revisions;
- `queried_nodes.json`: unique purchased oracle nodes and their manifest hash;
- `query_budget.json`: node cost, closed-edge count, failures, retries and runtime;
- `value_metrics.json`: scalar teacher approximation;
- `edge_metrics.json` and `ranking_metrics.json`: edit-field fidelity;
- `regret_metrics.json`: teacher- or experiment-evaluated optimization regret;
- `optimization_rollouts.json`: edit paths and acquisition decisions.

Value-KD and EditFlow comparisons must use exactly the same queried node IDs. An
equal edge count or nominal budget is not sufficient evidence of equal oracle
information. For GB1, measured and publication-imputed variants remain explicitly
separated; primary experimental-regret claims use measured variants only.

Failed runs remain in place as provenance. GPU runs must explicitly set
an authorized physical GPU ordinal. Generated outputs are excluded from Git; configs, runner,
model code and documentation are versioned.

Dependency installation follows the dry-run and `--no-build-isolation` contract
in the repository root `AGENTS.md`.
