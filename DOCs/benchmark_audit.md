# Benchmark intake audit

Status as of 2026-08-27. External repositories are pinned in
`preparation/sources.yaml`; downloaded sources are intentionally excluded from the
PLS Git history.

## Confirmed observations

- Official `esol.zip` contains 4,132 database rows. This is not the final labeled
  cohort size: usable quantitative measurements and sequences still need to be
  joined and filtered explicitly.
- FGNNSol contains 2,019 training proteins and one 660-row held-out CSV. Its
  validation/test membership is encoded by repository directories: 268 validation
  and 392 test proteins.
- SurfSol contains 2,002 training and 655 held-out proteins. They are strict
  sequence-level subsets of the matching FGNNSol pools: SurfSol excludes 17
  training and 5 held-out proteins.
- SurfSol does not preserve the FGNNSol 268/392 validation/test boundary in its
  two public CSV files. Comparing results requires reconstructing membership by
  sequence, not assuming its 655-row file is the 392-protein FGNNSol test set.
- ProtSolM contains 58,138/3,230/3,230 PDBSol train/validation/test rows, totaling
  64,598, plus precomputed global structural features and external datasets.
- FGNNSol and SurfSol do not expose a root license file at the pinned revisions.
  Their code/data may be inspected for reproducibility, but redistribution or
  incorporation into PLS needs explicit license clarification.

## Citation correction

The OUP URL ending in `vbag181/8714051` resolves to BioGraphX, not FGNNSol. The
FGNNSol paper should be cited through DOI `10.1021/acs.jcim.5c02262`. Benchmark
claims will be taken from primary papers and pinned repositories rather than from
cross-linked summaries.

## Next audits

1. Generate the normalized FGNNSol 2019/268/392 manifest.
2. Recompute the requested Biopython inter-protein SI<30% invariant instead of
   trusting the description of an upstream split.
3. Quantify whether SurfSol's 22 exclusions are random or correlated with length,
   solubility, fold confidence, or surface-generation failure.
4. Check sequence overlap between all eSOL splits and PDBSol/UESolDS before weak
   pretraining; bind every held-out eSOL homolog to the corresponding held-out
   component without deleting its observation.
5. Audit label definitions and provenance within PDBSol rather than treating all
   binary sources as the same physical endpoint.

## Dataset expansion policy

The project maximizes usable observations instead of creating a globally
nonredundant training set. Similar variants and repeated measurements are retained.
Sequence identity is used to constrain split assignment, not as a deletion rule:
all entities connected by SI>=30% must remain in one split. Dataset and record-level
provenance remain attached to every observation so dataset-specific heads, sample
weights and source-held-out evaluations remain possible.

UESolDS is a promising large source (78,031 entries), but its source labels combine
TargetTrack, DNASU, eSOL and PDB-derived annotations. It will enter as provenance-
aware weak binary supervision, not as interchangeable replicas of the eSOL PURE
continuous assay.

## UESolDS intake

PLM_Sol 1.1 (Zenodo record 12881509) was received and archive-verified. Its FASTA
files contain 70,031 train, 4,000 validation and 4,000 test observations. Across
all files there are 46,450 positive and 31,581 negative labels, matching the paper.

The release does not expose a per-record TargetTrack/DNASU/eSOL/PDB source column.
PLS therefore records the honest available provenance (`UESolDS_PLM_Sol_1.1`, the
original full FASTA header, archive/version/checksum) and marks the subcollection
as unavailable rather than inferring it from identifier spelling.

There are 78,030 unique sequence hashes for 78,031 observations. One exact sequence
occurs in both upstream train and test under different identifiers. We retain both
observations but must bind them to the same newly generated split; the upstream
split cannot be used as a strict SI30 benchmark without repair.

## Current maximum union

Without removing any repeated or similar observations, the received UESolDS,
PDBSol and FGNNSol eSOL cohort provide 145,308 observations over 132,781 unique
sequence entities:

| dataset | observations | unique sequences |
|---|---:|---:|
| UESolDS | 78,031 | 78,030 |
| PDBSol | 64,598 | 64,598 |
| eSOL | 2,679 | 2,679 |

Exact sequence overlaps are 11,908 for UESolDS/PDBSol, 552 for UESolDS/eSOL and
115 for PDBSol/eSOL. Among exact UESolDS/PDBSol overlaps, 1,333 sequence entities
have disagreeing binary labels. These observations remain in the corpus with their
original provenance and dataset-specific targets; they are not collapsed into one
invented consensus label.

The local host has 128 physical AMD EPYC 7763 cores / 256 hardware threads, 1 TiB
RAM and sufficient NVMe space, so the 8,815,330,590 all-pairs comparisons are
feasible without a similarity prefilter. A 2026-08-28 Biopython 1.88 benchmark on
the actual UESolDS length distribution measured about 95k pairs/s with 128 worker
processes and 105k pairs/s with 256. The direct compute estimate is 23.3 hours at
256 workers; the operational budget is 30--40 hours including length skew,
checkpointing, NUMA effects and output.

The primary split will therefore use exhaustive Biopython SI. A packed triangular
float32 matrix is about 35 GB and fits comfortably; threshold edges and nearest-
neighbor audit records are stored alongside it. Faster candidate methods remain
optional comparisons, not part of the scientific guarantee.

The exhaustive run and component audit are now complete. See
[`si_component_audit.md`](si_component_audit.md) for the construct-tag bridge,
giant-component sensitivity, and quantified incompatibility between the legacy
FGNNSol membership and strict SI30 splitting.
