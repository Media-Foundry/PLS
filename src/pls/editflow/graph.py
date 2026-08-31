"""Edit-graph operators and path-dependent optimization guarantees."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence

import numpy as np
import torch


def _edge_index_tensor(edge_index, *, device=None) -> torch.Tensor:
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if edges.numel() and torch.any(edges < 0):
        raise ValueError("edge_index cannot contain negative node indices")
    return edges


def edge_differences(values: torch.Tensor, edge_index) -> torch.Tensor:
    """Return the oriented scalar difference `value[target]-value[source]`."""
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional node potential")
    edges = _edge_index_tensor(edge_index, device=values.device)
    if edges.numel() and int(edges.max()) >= len(values):
        raise ValueError("edge_index references a missing node")
    return values[edges[1]] - values[edges[0]]


def graph_sobolev_loss(
    student_values: torch.Tensor,
    teacher_values: torch.Tensor,
    edge_index,
    edge_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Squared weighted discrepancy between student and teacher edit fields."""
    if student_values.shape != teacher_values.shape:
        raise ValueError("student and teacher node potentials must have equal shape")
    residual = edge_differences(student_values - teacher_values, edge_index)
    losses = residual.square()
    if edge_weight is not None:
        weights = torch.as_tensor(edge_weight, dtype=losses.dtype, device=losses.device)
        if weights.shape != losses.shape or torch.any(weights < 0):
            raise ValueError("edge_weight must be nonnegative with one value per edge")
        losses = losses * weights
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean() if losses.numel() else losses.sum()
    raise ValueError("reduction must be one of: none, sum, mean")


def shortest_path_discrepancies(
    student_values,
    teacher_values,
    edge_index,
    anchor: int,
    *,
    directed: bool = False,
) -> np.ndarray:
    """Compute minimum accumulated absolute edge-field error from an anchor."""
    student = np.asarray(student_values, dtype=np.float64)
    teacher = np.asarray(teacher_values, dtype=np.float64)
    if student.ndim != 1 or student.shape != teacher.shape:
        raise ValueError("student_values and teacher_values must be equal 1D arrays")
    if not 0 <= anchor < len(student):
        raise ValueError("anchor is outside the node range")
    edges = _edge_index_tensor(edge_index).cpu().numpy()
    if edges.size and int(edges.max()) >= len(student):
        raise ValueError("edge_index references a missing node")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(student))]
    delta = student - teacher
    for source, target in edges.T:
        weight = float(abs((delta[target] - delta[source])))
        adjacency[int(source)].append((int(target), weight))
        if not directed:
            adjacency[int(target)].append((int(source), weight))
    distance = np.full(len(student), np.inf, dtype=np.float64)
    distance[anchor] = 0.0
    queue: list[tuple[float, int]] = [(0.0, anchor)]
    while queue:
        current, node = heapq.heappop(queue)
        if current != distance[node]:
            continue
        for neighbor, weight in adjacency[node]:
            proposed = current + weight
            if proposed < distance[neighbor]:
                distance[neighbor] = proposed
                heapq.heappush(queue, (proposed, neighbor))
    return distance


def path_regret_bound(
    student_values,
    teacher_values,
    edge_index,
    anchor: int,
    teacher_optimum: int,
    student_choice: int,
    *,
    student_optimization_error: float = 0.0,
    directed: bool = False,
) -> float:
    """Return `D(x_T*) + D(x_hat) + eta` from the path-dependent theorem."""
    if student_optimization_error < 0:
        raise ValueError("student_optimization_error must be nonnegative")
    distances = shortest_path_discrepancies(
        student_values, teacher_values, edge_index, anchor, directed=directed
    )
    for node in (teacher_optimum, student_choice):
        if not 0 <= node < len(distances):
            raise ValueError("selected node is outside the node range")
    return float(distances[teacher_optimum] + distances[student_choice] + student_optimization_error)


def exact_optimization_regret(
    teacher_values,
    student_values,
    candidates: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Evaluate the student's selected design using the complete teacher table."""
    teacher = np.asarray(teacher_values, dtype=np.float64)
    student = np.asarray(student_values, dtype=np.float64)
    if teacher.ndim != 1 or teacher.shape != student.shape:
        raise ValueError("teacher_values and student_values must be equal 1D arrays")
    selected = np.arange(len(teacher), dtype=np.int64) if candidates is None else np.asarray(candidates, dtype=np.int64)
    if selected.ndim != 1 or not len(selected):
        raise ValueError("candidates must contain at least one node")
    if np.any(selected < 0) or np.any(selected >= len(teacher)):
        raise ValueError("candidate index is outside the node range")
    teacher_optimum = int(selected[np.argmax(teacher[selected])])
    student_choice = int(selected[np.argmax(student[selected])])
    return {
        "teacher_optimum": teacher_optimum,
        "student_choice": student_choice,
        "teacher_optimum_value": float(teacher[teacher_optimum]),
        "student_choice_teacher_value": float(teacher[student_choice]),
        "regret": float(teacher[teacher_optimum] - teacher[student_choice]),
    }


def exact_design_regrets(
    teacher_values,
    student_values,
    candidates: Sequence[int] | np.ndarray,
    queried_nodes: Iterable[int],
) -> dict:
    """Separate acquisition, novel-design, and end-to-end campaign regret.

    ``candidates`` defines one feasible design set (for example, measured GB1
    variants inside a fixed Hamming radius).  Oracle-purchased nodes are
    intersected with that set.  The novel student design is selected only from
    feasible nodes whose teacher value was not purchased.

    Acquisition and campaign regret use the optimum over the complete feasible
    set.  Novel-design regret instead uses the best *unqueried* feasible node as
    its reference, so it measures surrogate generalization without charging the
    student for designs that were no longer eligible.
    """
    teacher = np.asarray(teacher_values, dtype=np.float64)
    student = np.asarray(student_values, dtype=np.float64)
    if teacher.ndim != 1 or teacher.shape != student.shape:
        raise ValueError("teacher_values and student_values must be equal 1D arrays")
    feasible = np.asarray(candidates, dtype=np.int64)
    if feasible.ndim != 1 or not len(feasible):
        raise ValueError("candidates must contain at least one node")
    if np.any(feasible < 0) or np.any(feasible >= len(teacher)):
        raise ValueError("candidate index is outside the node range")
    if len(np.unique(feasible)) != len(feasible):
        raise ValueError("candidates must not contain duplicate nodes")

    queried = np.fromiter(
        sorted({int(node) for node in queried_nodes}), dtype=np.int64
    )
    if queried.size and (queried.min() < 0 or queried.max() >= len(teacher)):
        raise ValueError("queried node is outside the node range")
    is_acquired = np.isin(feasible, queried, assume_unique=False)
    acquired = feasible[is_acquired]
    novel = feasible[~is_acquired]
    if not len(acquired):
        raise ValueError("no queried node lies in the feasible candidate set")
    optimum = int(feasible[np.argmax(teacher[feasible])])
    acquired_best = int(acquired[np.argmax(teacher[acquired])])
    optimum_value = float(teacher[optimum])
    if len(novel):
        novel_optimum = int(novel[np.argmax(teacher[novel])])
        novel_choice = int(novel[np.argmax(student[novel])])
        novel_report = {
            "available": True,
            "teacher_optimum": novel_optimum,
            "teacher_optimum_value": float(teacher[novel_optimum]),
            "student_choice": novel_choice,
            "student_choice_teacher_value": float(teacher[novel_choice]),
            "regret": float(teacher[novel_optimum] - teacher[novel_choice]),
        }
        campaign_choice = (
            acquired_best
            if teacher[acquired_best] >= teacher[novel_choice]
            else novel_choice
        )
    else:
        novel_report = {
            "available": False,
            "reason": "all feasible candidates were queried",
            "regret": None,
        }
        campaign_choice = acquired_best

    return {
        "schema": "PLS_EditFlow_design_regrets_v1",
        "feasible_nodes": int(len(feasible)),
        "acquired_feasible_nodes": int(len(acquired)),
        "novel_feasible_nodes": int(len(novel)),
        "teacher_optimum": optimum,
        "teacher_optimum_value": optimum_value,
        "acquired": {
            "choice": acquired_best,
            "choice_teacher_value": float(teacher[acquired_best]),
            "regret": float(optimum_value - teacher[acquired_best]),
        },
        "novel_design": novel_report,
        "campaign": {
            "choice": int(campaign_choice),
            "choice_source": (
                "acquired" if campaign_choice == acquired_best else "novel_design"
            ),
            "choice_teacher_value": float(teacher[campaign_choice]),
            "regret": float(optimum_value - teacher[campaign_choice]),
        },
    }


def assert_same_queried_nodes(*node_sets: Iterable[int]) -> frozenset[int]:
    """Enforce identical oracle information, not merely equal query counts."""
    normalized = [frozenset(int(node) for node in nodes) for nodes in node_sets]
    if not normalized:
        raise ValueError("at least one queried-node set is required")
    if any(nodes != normalized[0] for nodes in normalized[1:]):
        raise ValueError("methods do not share the identical queried-node set")
    return normalized[0]
