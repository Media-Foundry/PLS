# SI30 component audit

Status: completed 2026-08-28 using the exhaustive Biopython 1.85 run recorded in
`artifacts/si/strict_si30_bio185/run_config.json`.

## Main finding

The 132,781 canonical sequence entities form 84,645 connected components at
SI>=0.30. There are 66,401 singleton components. The largest component contains
18,954 entities, far larger than the next component (35 entities).

This is not evidence that 18,954 proteins are one conventional homologous family.
It is the transitive closure of a sparse threshold graph. The giant component has
149,728 internal threshold edges, and 81,898 (54.7%) lie between SI 0.30 and 0.35.
Its median sequence length is 104 aa versus 251 aa outside the component.

High-degree bridge sequences expose shared expression-construct backbones. The
most prominent candidates contain the maltose-binding protein (MBP) sequence
`KIEEGKLVIWINGDKGYNGLAEVGKK...` fused to unrelated targets; some also contain
GFP or polyhistidine segments. Matching NESG/SGX records occur in both UESolDS
and PDBSol. A strict global-identity graph connects these constructs through the
shared tag/domain and then connects their different target domains transitively.

Known motif counts are:

| motif | giant component | outside giant |
|---|---:|---:|
| MBP core | 257 | 0 |
| GFP core | 28 | 0 |
| six consecutive histidines | 4,686 | 8,274 |

Tags are important accelerators but not the sole cause. At SI 0.30, removing all
nodes containing these known motifs still leaves a largest component of 10,894.
Other multi-domain proteins, short fragments and near-threshold chains also drive
percolation.

## Sensitivity

Threshold sensitivity, using the already computed exact pairwise identities:

| SI threshold | largest component | non-singleton entities |
|---:|---:|---:|
| 0.30 | 18,954 | 66,380 |
| 0.31 | 11,946 | 60,243 |
| 0.35 | 2,626 | 47,061 |
| 0.40 | 1,044 | 38,631 |
| 0.50 | 281 | 29,513 |
| 0.70 | 77 | 19,738 |
| 0.90 | 12 | 12,397 |

Counterfactual edge filters at the fixed 0.30 threshold:

| filter | largest component |
|---|---:|
| none | 18,954 |
| minimum length 30 | 17,421 |
| minimum length 50 | 15,607 |
| minimum length 100 | 7,904 |
| no ambiguous residues | 18,556 |
| no known fusion motif | 10,894 |
| pair length ratio >=0.5 | 17,648 |
| pair length ratio >=0.8 | 12,891 |
| combined conservative filter | 7,071 |

These are diagnostic ablations, not authorization to delete observations or alter
the frozen SI30 rule.

## Legacy benchmark conflict

The giant component alone contains 293 FGNNSol/eSOL entities: 225 legacy train,
25 legacy validation and 43 legacy test. Therefore the public 2,019/268/392 split
cannot satisfy the component-level SI30 invariant. Preserving legacy membership
and enforcing strict SI30 are mutually incompatible.

Keep the two benchmark tracks separate:

1. `legacy_fgnnsol` preserves published membership for literature comparison and
   is explicitly described as non-SI30.
2. `strict_biopython_si30` regenerates component-level membership. The giant
   component should provisionally be placed in train so validation/test are not
   dominated by one construct-bridged component. Allocation must still be checked
   for target and source balance.

A tag-aware target-sequence reconstruction is valuable as a sensitivity track
where record-level provenance supports unambiguous trimming. It must not silently
replace released construct sequences, invent target boundaries, or remove source
observations. Full construct and reconstructed target sequence identities should
remain separately versioned.

## Reproducible outputs

- `artifacts/si/strict_si30_bio185/audit/component_audit.json`
- `artifacts/si/strict_si30_bio185/audit/giant_bridge_candidates.csv`
- script: `preparation/audit_si_components.py`

At audit completion, SHA-256 values were:

- component audit JSON: `d1726bac1af609679dbe30704651ae3057fd01de87e02c2d82f4519e3e7c02dc`
- bridge candidates CSV: `85db43ef11fa6378be57ae9a0b5765bbc46c51c0cd2df2e1865598c030239623`
