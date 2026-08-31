"""Matched-budget objectives for value and edit-field distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .graph import _edge_index_tensor, graph_sobolev_loss


@dataclass(frozen=True)
class DistillationLoss:
    total: torch.Tensor
    value: torch.Tensor
    edge: torch.Tensor
    queried_nodes: int
    closed_edges: int


def editflow_distillation_loss(
    student_values: torch.Tensor,
    teacher_values: torch.Tensor,
    edge_index,
    queried_mask: torch.Tensor,
    *,
    value_weight: float = 1.0,
    edge_weight: float = 1.0,
    loss: str = "mse",
) -> DistillationLoss:
    """Use only edges whose two teacher-valued endpoints were both queried."""
    if student_values.ndim != 1 or student_values.shape != teacher_values.shape:
        raise ValueError("student_values and teacher_values must be equal 1D tensors")
    queried = torch.as_tensor(queried_mask, dtype=torch.bool, device=student_values.device)
    if queried.shape != student_values.shape or not queried.any():
        raise ValueError("queried_mask must select at least one node")
    if value_weight < 0 or edge_weight < 0 or value_weight + edge_weight <= 0:
        raise ValueError("loss weights must be nonnegative and not both zero")
    if not torch.isfinite(teacher_values[queried]).all():
        raise ValueError("queried teacher values must be finite")
    if loss == "mse":
        value_loss = F.mse_loss(student_values[queried], teacher_values[queried])
    elif loss == "huber":
        value_loss = F.huber_loss(student_values[queried], teacher_values[queried])
    else:
        raise ValueError("loss must be mse or huber")
    edges = _edge_index_tensor(edge_index, device=student_values.device)
    if edges.numel() and int(edges.max()) >= len(student_values):
        raise ValueError("edge_index references a missing node")
    closed = queried[edges[0]] & queried[edges[1]]
    closed_edges = edges[:, closed]
    edge_loss = graph_sobolev_loss(
        student_values, teacher_values, closed_edges, reduction="mean"
    )
    total = value_weight * value_loss + edge_weight * edge_loss
    return DistillationLoss(
        total=total, value=value_loss, edge=edge_loss,
        queried_nodes=int(queried.sum()), closed_edges=int(closed.sum()),
    )

