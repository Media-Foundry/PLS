# PLS agent instructions

## Dependency installation contract

For every future Python package installation:

1. Run a dry-run dependency resolution first and inspect all proposed installs,
   upgrades, downgrades and removals.
2. Confirm the plan does not conflict with the existing PyTorch/ROCm, NumPy,
   Biopython, ESM or other project-critical environment stack.
3. Perform the real installation with `--no-build-isolation`.
4. If the dry run reports conflicts or material core-stack changes, do not
   install; report the conflict and choose an isolated environment or another
   compatible version explicitly.
5. Record installed package versions used by experiments in each run's
   `environment.json`.

Typical pip sequence:

```bash
python -m pip install --dry-run <requirements>
python -m pip install --no-build-isolation <requirements>
```

This contract applies to the primary agent, sub-agents and all automated scripts.

## Permanent test-set freeze

The test split must never be evaluated, inferred, scored, calibrated, inspected, or used for model selection. Training entrypoints must hard-fail when `evaluate_test` is true. There is no command-line override or authorization bypass. Only validation outputs may be produced.
