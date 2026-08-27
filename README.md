# PLS

PLS is an *E. coli* protein-solubility prediction toolkit combining protein
language-model representations with confidence-aware structural features derived
from ESMFold predictions.

The current engineering and evaluation blueprint is in
[`DOCs/architecture.md`](DOCs/architecture.md). The received eSOL, UESolDS and
PDBSol sources normalize to 145,308 provenance-preserving observations over
132,781 canonical sequence entities. Downloaded sources and generated manifests
are deliberately excluded from Git.

Rebuild the union manifests after preparing the source inputs:

```bash
conda run -n BIO python preparation/build_union_manifest.py \
  --uesolds benchmark/generated/uesolds_observations.csv \
  --pdbsol-dir preparation/external/ProtSolM/data/PDBSol \
  --esol benchmark/generated/fgnnsol_esol_manifest.csv \
  --observations-output benchmark/generated/union_observations.csv \
  --entities-output benchmark/generated/sequence_entities.csv
```

Run the small exhaustive sequence-identity correctness and resume tests with:

```bash
conda run -n BIO python -m unittest discover -s tests -v
```

`preparation/si_engine.py` writes independently checksummed triangular blocks and
can resume without recomputing completed blocks. Do not launch the full 8.8-billion
pair run until the provisional alignment scoring policy has been scientifically
frozen and the worker/NUMA benchmark has been completed.
