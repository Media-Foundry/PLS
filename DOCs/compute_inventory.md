# Compute inventory and exhaustive-SI sharding

Measured on 2026-08-28 with the first 4,096 SHA-256-ordered entities
(8,386,560 unordered pairs), Biopython 1.85, global alignment, match 1,
mismatch 0, gap open/extend 0, and the first optimal traceback.

| host | CPU | NUMA | RAM | accelerator | SI workers | pairs/s |
|---|---|---:|---:|---|---:|---:|
| DiamondHill | 2x EPYC 7763, 128C/256T | 2 | 1 TiB | 8x MI250X logical devices, 64 GiB each | 128 | 97,663 |
| DiamondHill | same | 2 | 1 TiB | same | 256 | 115,387 |
| star | 2x EPYC 7K62, 96C/192T | 2 | 503 GiB | 2x A100 SXM4 80GB | 96 | 67,384 |
| star | same | 2 | 503 GiB | same | 192 | 73,617 |

All four runs produced the identical identity checksum `1520188.888622`.
SMT improves throughput by 18.1% on DiamondHill and 9.3% on star, so the initial
production setting should use all hardware threads unless concurrent host load
changes materially.

## Storage

- DiamondHill workspace: `/media/PM983/Code/PLS`, ext4 NVMe, about 2.8 TiB free.
- star workspace: `/data/husrcf/PLS`, ext4 HDD, about 8.2 TiB free on `/data`.
- The synchronized workspace has 2,276 regular files and 1,121,720,797 file
  bytes on both hosts. Six critical SHA-256 values and all four nested repository
  revisions match.

## Stable logical shards

Use 4,096 logical shards independent of the number of available machines. Upper
triangular block ordinal `k` belongs to logical shard `k % 4096`. This gives about
33 blocks per logical shard with 132,781 entities and block size 256. A machine
can receive any set/range of logical shards, and unfinished shards can be moved to
new resources without renumbering completed work.

The initial measured aggregate is 189,005 pairs/s. Proportional allocation is:

- DiamondHill: shards `0-2500` (2,501 shards, 61.1%);
- star: shards `2501-4095` (1,595 shards, 38.9%).

The pure-compute estimate for 8,815,330,590 pairs is 13.0 hours. Operationally
budget 16--20 hours for length skew, block compression/checksums, HDD writes on
star, other users, final synchronization, and reduction.

For NUMA locality, run two engine processes per host with disjoint halves of the
host's shard allocation. Bind each process to one NUMA node and its local memory;
use 128 workers per DiamondHill NUMA node and 96 workers per star NUMA node. If
this creates memory pressure or interferes with shared workloads, fall back to
64 and 48 physical-core workers respectively.

Each host writes blocks locally. Completed `.npz` artifacts and their
`.complete.json` checksum markers are merged only after computation. Never accept
a data file without its valid completion marker. Run the global union-find and
nearest-neighbor reduction only after every block validates in the merged output.

## Expansion policy

Maintain a small lease manifest outside the immutable run configuration with
logical shard, assigned host, state, attempt, start time, and completion checksum
inventory. New resources receive whole unstarted shards first. Reassign a running
shard only after its old worker is stopped; already completed blocks remain valid
and the new worker resumes the remainder. Measure every new CPU type using
`preparation/benchmark_si.py`, then allocate by observed pairs/s.

Do not start production until the scoring definition, Biopython version, optimal
alignment tie policy, block size, and immutable run configuration have been
scientifically frozen.
