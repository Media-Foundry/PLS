#!/usr/bin/env bash
# Exact features and scoring across ALL confirmatory candidates.
#
# Retrospective only: this supplies the measured exhaustive denominator and the
# coverage/regret audit. It runs after the selected-stage report is frozen.
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_cost_confirmatory_v2"
manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_cost_confirmatory_v2.json"
entities="$repo_root/benchmark/generated/pls_editflow_entities_cost_confirmatory_v2.csv"
plan="$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_cost_confirmatory_v2_full.json"
full="$artifact_root/exact_full"
fixed="$artifact_root/fixed_parent"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
score_config="$repo_root/configs/editflow/pls_oracle_score_cost_confirmatory_exact_full_v2.json"
log="$artifact_root/logs/retrospective_postfold.log"

export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

note() { echo "[$(date --iso-8601=seconds)] $*"; }

{
note "verifying the exhaustive fold set is complete"
"$python_bin" - <<PY
import json, sys
from pathlib import Path
plan = json.loads(Path("$plan").read_text())
root = Path("$full/esmfold")
reports = []
for shard in range(plan["shard_count"]):
    path = root / f"shard_{shard:03d}_report.json"
    if not path.is_file():
        sys.exit(f"missing shard report: {path}")
    reports.append(json.loads(path.read_text()))
assigned = sum(int(r["assigned"]) for r in reports)
failed = sum(int(r["failed"]) for r in reports)
if assigned != len(plan["assignments"]) or failed:
    sys.exit(f"exhaustive fold set incomplete: assigned={assigned}, failed={failed}")
print(json.dumps({
    "exhaustive_folds": assigned,
    "newly_folded_retrospective": sum(int(r["ok"]) for r in reports),
    "reused_from_selected_stage": sum(int(r["skipped"]) for r in reports),
    "failed": failed,
}, indent=2))
PY

note "extracting exact V4 features for every confirmatory node"
numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$manifest" --pdb-root "$full/esmfold" \
    --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$full/structure_v4_raw" --source-root /home/pc/Code/BIO/protein --workers 192

note "building compact, geometry, vector and surface caches"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$entities" --raw-root "$full/structure_v4_raw" \
    --status "$full/structure_v4_raw/status.npy" --stats "$stats" \
    --output "$full/structure_v4_compact"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
    --entities "$entities" --raw-root "$full/structure_v4_raw" \
    --status "$full/structure_v4_raw/status.npy" --compact-root "$full/structure_v4_compact" \
    --output "$full/structure_v4_geometry" --neighbors 16
numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
    --entities "$entities" --feature-root "$full/structure_v4_raw" \
    --compact-dir "$full/structure_v4_compact" --status "$full/structure_v4_raw/status.npy" \
    --output "$full/structure_v4_vectors" --workers 192
numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
    --entities "$entities" --compact "$full/structure_v4_compact" \
    --geometry "$full/structure_v4_geometry" --structure-stats "$stats" \
    --output "$full/surface_patch_components"

note "checking whether the existing PLM caches already index this entity order"
"$python_bin" - <<PY
import numpy as np, sys
from pathlib import Path
a = np.load("$fixed/structure_v4_compact/offsets.npy")
b = np.load("$full/structure_v4_compact/offsets.npy")
if a.shape != b.shape or not np.array_equal(a, b):
    sys.exit("offsets differ; reindex the PLM caches before scoring")
print("offsets identical: the confirmatory PLM caches apply unchanged")
PY

note "scoring every confirmatory candidate against the exact oracle"
HIP_VISIBLE_DEVICES=1 "$python_bin" -m pls.oracles.score_editflow \
    --config "$score_config" --output-root "$full/scores_fp32_a" \
    >> "$artifact_root/logs/score_exact_full.log" 2>&1

note "retrospective exact scoring complete"
} 2>&1 | tee -a "$log"
