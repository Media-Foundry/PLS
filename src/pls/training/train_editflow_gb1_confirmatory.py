"""Paired-protocol GB1 acquisition benchmark over frozen confirmatory anchors."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from pls.editflow.hamming import (hamming_distance, queried_nodes_sha256,
                                  variants_from_tokens)
from pls.editflow.optimization import (bound_aware_frontier_acquisition,
                                       hybrid_query_budget,
                                       path_aware_frontier_acquisition)
from pls.training.train_editflow_gb1 import (connected_query_nodes,
                                             evaluation_edges)
from pls.training.train_editflow_gb1_active import (evaluate, fit_ensemble,
                                                    frontier_policy_acquisition,
                                                    uncertainty_acquisition)


def load_anchor_protocol(
    path: Path,
    raw_tokens: np.ndarray,
    measured: np.ndarray,
) -> tuple[list[dict], dict]:
    """Validate a value-blind, immutable anchor protocol against the landscape."""
    protocol = json.loads(path.read_text())
    if protocol.get("schema") != "PLS_EditFlow_GB1_anchor_protocol_v1":
        raise ValueError("unsupported GB1 anchor protocol schema")
    selection = protocol.get("selection", {})
    if selection.get("fitness_accessed") is not False:
        raise ValueError("anchor selection must be explicitly fitness blind")
    if protocol.get("test_evaluated") is not False:
        raise ValueError("anchor protocol must preserve the permanent test freeze")
    anchors = protocol.get("anchors", [])
    if len(anchors) != int(selection.get("count", -1)):
        raise ValueError("anchor protocol count mismatch")
    ranks = [int(row["rank"]) for row in anchors]
    if ranks != list(range(len(anchors))):
        raise ValueError("anchor ranks must be consecutive and ordered")
    nodes = np.asarray([row["node_index"] for row in anchors], dtype=np.int64)
    if len(np.unique(nodes)) != len(nodes):
        raise ValueError("anchor protocol contains duplicate nodes")
    if nodes.size and (nodes.min() < 0 or nodes.max() >= len(raw_tokens)):
        raise ValueError("anchor protocol references a missing node")
    if not np.all(measured[nodes]):
        raise ValueError("confirmatory anchors must be experimentally measured")
    if queried_nodes_sha256(nodes) != protocol.get("anchors_sha256"):
        raise ValueError("anchor protocol SHA-256 mismatch")
    variants = variants_from_tokens(raw_tokens[nodes], "ACDEFGHIKLMNPQRSTVWY")
    if variants != [str(row["variant"]) for row in anchors]:
        raise ValueError("anchor variant identity mismatch")
    return anchors, protocol


def acquire_next_round(
    ensemble: np.ndarray,
    queried: set[int],
    measured: np.ndarray,
    anchor: int,
    increment: int,
    data_config: dict,
    rng: np.random.Generator,
) -> tuple[list[int], dict]:
    mode = data_config["acquisition"]
    if mode == "path_aware":
        acquired = path_aware_frontier_acquisition(
            ensemble,
            queried,
            measured,
            anchor,
            increment,
            alphabet_size=20,
            length=4,
            steps=int(data_config["beam_steps"]),
            beam_width=int(data_config["beam_width"]),
            conservative_beta=float(data_config.get("conservative_beta", 0.0)),
        )
        selected = acquired.batch.node_indices.tolist()
        details = {
            "mode": mode,
            "path_count": len(acquired.paths),
            "path_edges": int(acquired.path_edges.shape[1]),
            "path_selected": len(selected),
        }
    elif mode == "hybrid_path":
        targeted_budget = hybrid_query_budget(
            increment, float(data_config["path_fraction"])
        )
        acquired = path_aware_frontier_acquisition(
            ensemble,
            queried,
            measured,
            anchor,
            targeted_budget,
            alphabet_size=20,
            length=4,
            steps=int(data_config["beam_steps"]),
            beam_width=int(data_config["beam_width"]),
            conservative_beta=float(data_config.get("conservative_beta", 0.0)),
        )
        selected = acquired.batch.node_indices.tolist()
        details = {
            "mode": mode,
            "path_fraction": float(data_config["path_fraction"]),
            "path_budget": targeted_budget,
            "exploration_budget": increment - targeted_budget,
            "path_count": len(acquired.paths),
            "path_edges": int(acquired.path_edges.shape[1]),
            "path_selected": len(selected),
        }
    elif mode == "occupancy_only":
        acquired = path_aware_frontier_acquisition(
            ensemble,
            queried,
            measured,
            anchor,
            increment,
            alphabet_size=20,
            length=4,
            steps=int(data_config["beam_steps"]),
            beam_width=int(data_config["beam_width"]),
            conservative_beta=float(data_config.get("conservative_beta", 0.0)),
            score_mode="occupancy_only",
        )
        selected = acquired.batch.node_indices.tolist()
        details = {
            "mode": mode,
            "path_count": len(acquired.paths),
            "path_edges": int(acquired.path_edges.shape[1]),
            "occupancy_selected": len(selected),
        }
    elif mode == "bound_aware":
        acquired = bound_aware_frontier_acquisition(
            ensemble,
            queried,
            measured,
            anchor,
            increment,
            alphabet_size=20,
            length=4,
            steps=int(data_config["beam_steps"]),
            beam_width=int(data_config["beam_width"]),
            conservative_beta=float(data_config.get("conservative_beta", 0.0)),
        )
        selected = acquired.batch.node_indices.tolist()
        details = {
            "mode": mode,
            "candidate_endpoints": len(acquired.candidate_endpoints),
            "bound_paths": len(acquired.selected_paths),
            "path_edges": int(acquired.path_edges.shape[1]),
            "bound_selected": len(selected),
            "mean_estimated_path_bound": (
                float(acquired.estimated_path_bounds.mean())
                if len(acquired.estimated_path_bounds)
                else 0.0
            ),
        }
    elif mode == "uncertainty":
        acquired, edges = uncertainty_acquisition(
            ensemble, queried, measured, increment
        )
        selected = acquired.node_indices.tolist()
        details = {
            "mode": mode,
            "frontier_edges": int(edges.shape[1]),
            "uncertainty_selected": len(selected),
        }
    elif mode in {"random", "greedy", "ucb", "thompson"}:
        acquired, edges = frontier_policy_acquisition(
            ensemble,
            queried,
            measured,
            increment,
            mode,
            rng,
            beta=float(data_config.get("acquisition_beta", 1.0)),
        )
        selected = acquired.node_indices.tolist()
        details = {
            "mode": mode,
            "frontier_edges": int(edges.shape[1]),
            "policy_selected": len(selected),
        }
    else:
        raise ValueError(
            "unsupported acquisition policy"
        )

    if len(selected) < increment:
        if mode == "occupancy_only":
            fill, edges = frontier_policy_acquisition(
                ensemble,
                queried,
                measured,
                increment - len(selected),
                "random",
                rng,
                excluded_targets=selected,
            )
            details["random_fill"] = len(fill.node_indices)
        elif mode in {"random", "greedy", "ucb", "thompson"}:
            fill, edges = frontier_policy_acquisition(
                ensemble,
                queried,
                measured,
                increment - len(selected),
                mode,
                rng,
                beta=float(data_config.get("acquisition_beta", 1.0)),
                excluded_targets=selected,
            )
            details["policy_fill"] = len(fill.node_indices)
        else:
            fill, edges = uncertainty_acquisition(
                ensemble,
                queried,
                measured,
                increment - len(selected),
                excluded_targets=selected,
            )
            details["uncertainty_fill"] = len(fill.node_indices)
        selected.extend(fill.node_indices.tolist())
        details["fill_frontier_edges"] = int(edges.shape[1])
    if len(selected) != increment or set(selected) & queried:
        raise RuntimeError("acquisition did not purchase the exact new-node budget")
    return list(map(int, selected)), details


def summarize(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "count": 0,
        }
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "count": int(len(values)),
    }


def aggregate_anchor_results(
    anchor_results: list[dict],
    budgets: list[int],
    radii: list[int],
) -> list[dict]:
    """Aggregate independently trained anchors without pooling their queries."""
    aggregate = []
    for stage_index, budget in enumerate(budgets):
        stages = [result["stages"][stage_index] for result in anchor_results]
        row = {
            "budget": int(budget),
            "anchors": len(stages),
            "value_r2": summarize(stage["value"]["r2"] for stage in stages),
            "edge_spearman": summarize(
                stage["edge"]["edge_spearman"] for stage in stages
            ),
            "anchor_macro_kendall_tau": summarize(
                stage["edge"]["anchor_macro_kendall_tau"] for stage in stages
            ),
            "closed_edges": summarize(stage["closed_edges"] for stage in stages),
            "regret": {},
        }
        for radius in radii:
            row["regret"][str(radius)] = {}
            for metric in ("acquired", "novel_design", "campaign"):
                values = [
                    stage["regret"][str(radius)][metric]["regret"]
                    for stage in stages
                    if stage["regret"][str(radius)][metric]["regret"] is not None
                ]
                row["regret"][str(radius)][metric] = {
                    **summarize(values),
                    "zero_regret_fraction": float(
                        np.mean(np.asarray(values) <= 1e-12)
                    ) if values else None,
                    "unavailable_fraction": float(1.0 - len(values) / len(stages)),
                }
        aggregate.append(row)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    data_config = config["data"]
    training = config["training"]
    if config.get("evaluate_test", False):
        parser.error("test evaluation is permanently disabled")
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(training["hip_device"]):
        parser.error("HIP device mismatch")

    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    landscape = np.load(data_config["landscape"])
    raw_tokens = landscape["tokens"].astype(np.int64)
    tokens = torch.from_numpy(raw_tokens + 1)
    fitness = landscape["fitness"].astype(np.float64)
    measured = landscape["is_measured"].astype(bool)
    anchors, anchor_protocol = load_anchor_protocol(
        Path(data_config["anchor_protocol"]), raw_tokens, measured
    )
    field_edges, edge_groups = evaluation_edges(
        raw_tokens,
        measured,
        int(data_config["evaluation_anchors"]),
        data_config["evaluation_salt"],
    )
    budgets = list(map(int, data_config["query_budgets"]))
    radii = list(map(int, data_config["edit_radii"]))
    if budgets != sorted(set(budgets)) or budgets[0] < 1:
        raise ValueError("query budgets must be strictly increasing and positive")
    anchor_stride = int(training["anchor_seed_stride"])
    writer = SummaryWriter(arguments.run_dir / "tensorboard")
    anchor_results = []
    query_manifests = []
    all_rollouts = []

    for anchor_row in anchors:
        rank = int(anchor_row["rank"])
        anchor = int(anchor_row["node_index"])
        variant = str(anchor_row["variant"])
        anchor_training = dict(training)
        anchor_training["ensemble_seeds"] = [
            int(member_seed) + rank * anchor_stride
            for member_seed in training["ensemble_seeds"]
        ]
        initial_query_seed = int(data_config["initial_query_seed"]) + rank * anchor_stride
        queried = set(
            map(
                int,
                connected_query_nodes(
                    measured, anchor, budgets[0], initial_query_seed
                ),
            )
        )
        distances = hamming_distance(raw_tokens, raw_tokens[anchor])
        stages = []
        rollouts = []

        for stage_index, budget in enumerate(budgets):
            if len(queried) != budget:
                raise RuntimeError("query budget mismatch before training round")
            ensemble, states, training_summary, closed_edges = fit_ensemble(
                tokens,
                fitness,
                queried,
                config["model"],
                anchor_training,
                device,
            )
            value_metrics, edge_metrics, regret_metrics = evaluate(
                ensemble,
                fitness,
                measured,
                distances,
                field_edges,
                edge_groups,
                radii,
                int(data_config.get("top_k", 10)),
                queried,
            )
            stage = {
                "round": stage_index,
                "budget": budget,
                "queried_nodes_sha256": queried_nodes_sha256(queried),
                "closed_edges": closed_edges,
                "training": training_summary,
                "value": value_metrics,
                "edge": edge_metrics,
                "regret": regret_metrics,
            }
            stages.append(stage)
            print(
                json.dumps(
                    {
                        "anchor_rank": rank,
                        "anchor": variant,
                        "budget": budget,
                        "r2": value_metrics["r2"],
                        "regret": {
                            radius: {
                                metric: values[metric]["regret"]
                                for metric in ("acquired", "novel_design", "campaign")
                            }
                            for radius, values in regret_metrics.items()
                        },
                    }
                ),
                flush=True,
            )
            writer.add_scalar(f"anchors/{rank:02d}/r2", value_metrics["r2"], budget)
            for radius in radii:
                for metric in ("acquired", "novel_design", "campaign"):
                    metric_value = regret_metrics[str(radius)][metric]["regret"]
                    if metric_value is not None:
                        writer.add_scalar(
                            f"anchors/{rank:02d}/{metric}_regret_radius_{radius}",
                            metric_value,
                            budget,
                        )
            if stage_index == len(budgets) - 1:
                torch.save(
                    {
                        "members": states,
                        "anchor": anchor_row,
                        "queried_nodes": sorted(queried),
                        "config": config,
                    },
                    arguments.run_dir / "checkpoints" / f"anchor_{rank:02d}.pt",
                )
                break
            increment = budgets[stage_index + 1] - budget
            acquisition_seed = initial_query_seed + 104_729 * (stage_index + 1)
            selected, details = acquire_next_round(
                ensemble,
                queried,
                measured,
                anchor,
                increment,
                data_config,
                np.random.default_rng(acquisition_seed),
            )
            queried.update(selected)
            details.update(
                {
                    "anchor_rank": rank,
                    "anchor": variant,
                    "from_budget": budget,
                    "to_budget": len(queried),
                    "selected_nodes": selected,
                    "acquisition_seed": acquisition_seed,
                }
            )
            rollouts.append(details)

        identity = queried_nodes_sha256(queried)
        query_manifests.append(
            {
                "anchor_rank": rank,
                "anchor_node": anchor,
                "anchor_variant": variant,
                "node_indices": sorted(queried),
                "sha256": identity,
                "oracle_values_included": False,
            }
        )
        anchor_results.append(
            {
                "anchor": anchor_row,
                "initial_query_seed": initial_query_seed,
                "ensemble_seeds": anchor_training["ensemble_seeds"],
                "stages": stages,
            }
        )
        all_rollouts.extend(rollouts)

    aggregate = aggregate_anchor_results(anchor_results, budgets, radii)
    for row in aggregate:
        budget = row["budget"]
        writer.add_scalar("aggregate/r2_mean", row["value_r2"]["mean"], budget)
        for radius in radii:
            for metric in ("acquired", "novel_design", "campaign"):
                metric_mean = row["regret"][str(radius)][metric]["mean"]
                if metric_mean is not None:
                    writer.add_scalar(
                        f"aggregate/{metric}_regret_radius_{radius}_mean",
                        metric_mean,
                        budget,
                    )
    writer.close()

    query_budget = {
        "acquisition": data_config["acquisition"],
        "anchors": len(anchors),
        "budget_curve": budgets,
        "final_unique_queries_per_anchor": budgets[-1],
        "total_oracle_queries_across_independent_anchors": len(anchors) * budgets[-1],
        "teacher_query_cost_unit": "unique measured node per independent anchor",
        "same_protocol_required": True,
        "anchor_protocol_sha256": anchor_protocol["anchors_sha256"],
        "test_evaluated": False,
    }
    artifacts = {
        "history.json": anchor_results,
        "aggregate_metrics.json": aggregate,
        "optimization_rollouts.json": all_rollouts,
        "queried_nodes.json": {
            "schema": "PLS_EditFlow_multi_anchor_queried_nodes_v1",
            "manifests": query_manifests,
            "oracle_values_included": False,
        },
        "query_budget.json": query_budget,
    }
    for name, value in artifacts.items():
        (arguments.run_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "query_budget": query_budget,
                "test_evaluated": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
