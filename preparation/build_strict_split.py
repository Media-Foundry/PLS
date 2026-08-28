"""Build a deterministic component-level strict SI30 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_tiebreak(seed: int, component: str, split: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{component}:{split}".encode()).digest()[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--entity-output", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=.75)
    parser.add_argument("--validation-fraction", type=float, default=.10)
    parser.add_argument("--test-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    fractions = dict(zip(SPLITS, (args.train_fraction, args.validation_fraction, args.test_fraction)))
    if abs(sum(fractions.values()) - 1) > 1e-12 or any(value <= 0 for value in fractions.values()):
        parser.error("split fractions must be positive and sum to one")

    with args.components.open(newline="", encoding="utf-8") as handle:
        component_rows = list(csv.DictReader(handle))
    component_by_entity = {row["sequence_sha256"]: row["component_root_sha256"] for row in component_rows}
    component_sizes = Counter(component_by_entity.values())
    giant_component = component_sizes.most_common(1)[0][0]

    with args.observations.open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))
    features: dict[str, Counter] = defaultdict(Counter)
    for component, size in component_sizes.items():
        features[component]["entities"] = size
    for row in observations:
        component = component_by_entity[row["sequence_sha256"]]
        source = row["source_dataset"]
        features[component]["observations"] += 1
        features[component][f"source:{source}"] += 1
        if row["target_kind"] == "binary":
            features[component][f"binary:{source}:{row['target_value']}"] += 1
        elif source == "eSOL_FGNNSol":
            value = min(max(float(row["target_value"]), 0), 1)
            bin_index = min(int(value * 10), 9)
            features[component][f"esol_bin:{bin_index}"] += 1

    totals = Counter()
    for component_features in features.values():
        totals.update(component_features)
    targets = {split: {name: total * fractions[split] for name, total in totals.items()} for split in SPLITS}
    assigned_totals = {split: Counter() for split in SPLITS}
    assignment = {giant_component: "train"}
    assigned_totals["train"].update(features[giant_component])

    # Large and feature-rich components go first. With the giant fixed to train,
    # normalized squared target error provides deterministic multi-objective balance.
    remaining = [component for component in features if component != giant_component]
    remaining.sort(key=lambda component: (-features[component]["entities"],
                                          -features[component]["observations"], component))
    feature_weights = {name: (4.0 if name.startswith("esol_bin:") else
                              2.0 if name.startswith("binary:") else
                              1.5 if name.startswith("source:") else 1.0)
                       for name in totals}

    def incremental_cost(component: str, split: str) -> float:
        cost = 0.0
        for name, value in features[component].items():
            target = max(targets[split][name], 1.0)
            before = assigned_totals[split][name] - target
            after = before + value
            cost += feature_weights[name] * (after * after - before * before) / target
        # A small global entity-load term prevents feature-sparse components from
        # accumulating in the same split.
        entity_target = max(targets[split]["entities"], 1.0)
        before = assigned_totals[split]["entities"] - entity_target
        after = before + features[component]["entities"]
        cost += 3.0 * (after * after - before * before) / entity_target
        return cost

    for component in remaining:
        chosen = min(SPLITS, key=lambda split: (incremental_cost(component, split),
                                                stable_tiebreak(args.seed, component, split)))
        assignment[component] = chosen
        assigned_totals[chosen].update(features[component])

    args.entity_output.parent.mkdir(parents=True, exist_ok=True)
    with args.entity_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_sha256", "component_root_sha256", "split"])
        writer.writeheader()
        for row in component_rows:
            writer.writerow({**row, "split": assignment[row["component_root_sha256"]]})
    with args.observation_output.open("w", newline="", encoding="utf-8") as handle:
        fields = ["observation_id", "sequence_sha256", "component_root_sha256", "split",
                  "source_dataset", "target_kind", "target_value"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in observations:
            component = component_by_entity[row["sequence_sha256"]]
            writer.writerow({"observation_id": row["observation_id"], "sequence_sha256": row["sequence_sha256"],
                             "component_root_sha256": component, "split": assignment[component],
                             "source_dataset": row["source_dataset"], "target_kind": row["target_kind"],
                             "target_value": row["target_value"]})

    report = {
        "schema_version": 1, "seed": args.seed, "fractions": fractions,
        "giant_component": giant_component, "giant_component_size": component_sizes[giant_component],
        "giant_component_split": assignment[giant_component], "component_count": len(component_sizes),
        "counts": {split: dict(sorted(assigned_totals[split].items())) for split in SPLITS},
        "component_counts": dict(Counter(assignment.values())),
        "entity_split_sha256": file_sha256(args.entity_output),
        "observation_split_sha256": file_sha256(args.observation_output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
