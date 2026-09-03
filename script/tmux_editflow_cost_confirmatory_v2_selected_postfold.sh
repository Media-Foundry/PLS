#!/usr/bin/env bash
# Exact features and scoring for the gated selected confirmatory mutants only.
#
# Runs after the selected-only fold campaign. It touches nothing outside the
# selected subset, so the unselected candidates stay unfolded until the
# selected-stage deployment report is frozen.
set -euo pipefail

repo_root=${PLS_REPO_ROOT:-/media/PM983/Code/PLS}
python_bin=${PLS_PYTHON:-/home/pc/anaconda3/envs/BIO/bin/python}
artifact_root="$repo_root/artifacts/oracles/pls_editflow_cost_confirmatory_v2"
full_manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_cost_confirmatory_v2.json"
full_entities="$repo_root/benchmark/generated/pls_editflow_entities_cost_confirmatory_v2.csv"
selection="$repo_root/analysis/editflow_pls_cost_aware_confirmatory_v2_selection.json"
sub_manifest="$repo_root/benchmark/generated/pls_editflow_oracle_manifest_cost_confirmatory_v2_selected.json"
sub_entities="$repo_root/benchmark/generated/pls_editflow_entities_cost_confirmatory_v2_selected.csv"
sub_report="$repo_root/benchmark/generated/pls_editflow_oracle_subset_report_cost_confirmatory_v2_selected.json"
exact="$artifact_root/exact_selected"
fixed="$artifact_root/fixed_parent"
stats="$repo_root/artifacts/features/pdbsol_structure_v4_train_stats.json"
score_config="$repo_root/configs/editflow/pls_oracle_score_cost_confirmatory_exact_selected_v2.json"
log="$artifact_root/logs/selected_postfold.log"

export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"

note() { echo "[$(date --iso-8601=seconds)] $*"; }

{
note "verifying the selected fold campaign completed cleanly"
"$python_bin" - <<PY
import json, sys
from pathlib import Path
root = Path("$exact/esmfold")
plan = json.loads(Path("$repo_root/benchmark/generated/pls_editflow_oracle_query_plan_cost_confirmatory_v2_selected.json").read_text())
reports = []
for shard in range(plan["shard_count"]):
    path = root / f"shard_{shard:03d}_report.json"
    if not path.is_file():
        sys.exit(f"missing shard report: {path}")
    reports.append(json.loads(path.read_text()))
assigned = sum(int(r["assigned"]) for r in reports)
failed = sum(int(r["failed"]) for r in reports)
if assigned != len(plan["assignments"]):
    sys.exit(f"folded {assigned} mutants but the plan holds {len(plan['assignments'])}")
if failed:
    sys.exit(f"{failed} folds failed")
print(json.dumps({
    "selected_folds": assigned,
    "new": sum(int(r["ok"]) for r in reports),
    "cached": sum(int(r["skipped"]) for r in reports),
    "failed": failed,
}, indent=2))
PY

note "building the selected-only subset manifest and entities"
"$python_bin" preparation/subset_pls_editflow_selected_manifest.py \
    --manifest "$full_manifest" --selection "$selection" \
    --policy-id exact_best_runtime_gamma1 \
    --output "$sub_manifest" --entities-output "$sub_entities" --report "$sub_report"

note "extracting exact V4 features for anchors and selected mutants"
numactl --interleave=all "$python_bin" preparation/extract_pls_editflow_structure_v4.py \
    --manifest "$sub_manifest" --pdb-root "$exact/esmfold" \
    --parent-raw-root "$repo_root/artifacts/features/pdbsol_structure_v4_raw" \
    --output-root "$exact/structure_v4_raw" --source-root /home/pc/Code/BIO/protein --workers 192

note "building compact, geometry, vector and surface caches"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_compact.py \
    --entities "$sub_entities" --raw-root "$exact/structure_v4_raw" \
    --status "$exact/structure_v4_raw/status.npy" --stats "$stats" \
    --output "$exact/structure_v4_compact"
numactl --interleave=all "$python_bin" preparation/build_structure_v4_geometry.py \
    --entities "$sub_entities" --raw-root "$exact/structure_v4_raw" \
    --status "$exact/structure_v4_raw/status.npy" --compact-root "$exact/structure_v4_compact" \
    --output "$exact/structure_v4_geometry" --neighbors 16
numactl --interleave=all "$python_bin" preparation/build_structure_v4_vectors.py \
    --entities "$sub_entities" --feature-root "$exact/structure_v4_raw" \
    --compact-dir "$exact/structure_v4_compact" --status "$exact/structure_v4_raw/status.npy" \
    --output "$exact/structure_v4_vectors" --workers 192
numactl --interleave=all "$python_bin" preparation/build_surface_patch_components.py \
    --entities "$sub_entities" --compact "$exact/structure_v4_compact" \
    --geometry "$exact/structure_v4_geometry" --structure-stats "$stats" \
    --output "$exact/surface_patch_components"

note "reindexing the exact-sequence PLM caches onto the selected subset"
"$python_bin" preparation/reindex_editflow_plm_cache.py \
    --source-entities "$full_entities" --target-entities "$sub_entities" \
    --source-offsets "$fixed/structure_v4_compact/offsets.npy" \
    --target-offsets "$exact/structure_v4_compact/offsets.npy" \
    --source-mean "$artifact_root/esm2_mean" \
    --source-residue-pca "$artifact_root/residue_esm2_pca" \
    --output-mean "$exact/esm2_mean" --output-residue-pca "$exact/residue_esm2_pca"

note "scoring the selected exact mutants with the frozen teacher in float32"
HIP_VISIBLE_DEVICES=1 "$python_bin" -m pls.oracles.score_editflow \
    --config "$score_config" --output-root "$exact/scores_fp32_a" \
    >> "$artifact_root/logs/score_exact_selected.log" 2>&1

note "selected-stage exact scoring complete; unselected candidates remain unfolded"
} 2>&1 | tee -a "$log"
