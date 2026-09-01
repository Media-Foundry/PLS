"""Sequence-only students built on frozen protein-language-model features."""

from __future__ import annotations

import torch
from torch import nn

from .mutations import AMINO_ACIDS


class PLMPotentialHead(nn.Module):
    """Scalar potential over exact-sequence frozen PLM representations."""

    def __init__(self, global_dimension: int, residue_dimension: int, dimension=256, dropout=.1):
        super().__init__()
        self.global_projection = nn.Sequential(
            nn.LayerNorm(global_dimension), nn.Linear(global_dimension, dimension), nn.GELU()
        )
        self.residue_projection = nn.Sequential(
            nn.LayerNorm(residue_dimension), nn.Linear(residue_dimension, dimension), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dimension * 4), nn.Linear(dimension * 4, dimension),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension, 1),
        )

    def forward(self, global_embedding: torch.Tensor, pooled_residue: torch.Tensor) -> torch.Tensor:
        global_hidden = self.global_projection(global_embedding)
        residue_hidden = self.residue_projection(pooled_residue)
        features = torch.cat((
            global_hidden,
            residue_hidden,
            global_hidden * residue_hidden,
            (global_hidden - residue_hidden).abs(),
        ), dim=-1)
        return self.head(features).squeeze(-1)


class PLMMutationDeltaHead(nn.Module):
    """Predict one edit effect from the parent PLM state and explicit edit token."""

    def __init__(self, global_dimension: int, residue_dimension: int, dimension=256, dropout=.1):
        super().__init__()
        amino_acids = len(AMINO_ACIDS)
        self.amino_acid = nn.Embedding(amino_acids, 32)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(global_dimension), nn.Linear(global_dimension, dimension), nn.GELU()
        )
        self.pooled_projection = nn.Sequential(
            nn.LayerNorm(residue_dimension), nn.Linear(residue_dimension, dimension), nn.GELU()
        )
        self.local_projection = nn.Sequential(
            nn.LayerNorm(residue_dimension), nn.Linear(residue_dimension, dimension), nn.GELU()
        )
        input_dimension = dimension * 3 + 32 * 3 + 2
        self.head = nn.Sequential(
            nn.LayerNorm(input_dimension), nn.Linear(input_dimension, dimension),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension, 1),
        )

    def forward(
        self,
        global_embedding: torch.Tensor,
        pooled_residue: torch.Tensor,
        local_residue: torch.Tensor,
        source_residue: torch.Tensor,
        target_residue: torch.Tensor,
        normalized_position: torch.Tensor,
        normalized_log_length: torch.Tensor,
    ) -> torch.Tensor:
        source = self.amino_acid(source_residue)
        target = self.amino_acid(target_residue)
        features = torch.cat((
            self.global_projection(global_embedding),
            self.pooled_projection(pooled_residue),
            self.local_projection(local_residue),
            source,
            target,
            target - source,
            normalized_position[:, None],
            normalized_log_length[:, None],
        ), dim=-1)
        return self.head(features).squeeze(-1)


def commuting_cycle_residual(
    anchor_to_i: torch.Tensor,
    after_i_to_j: torch.Tensor,
    anchor_to_j: torch.Tensor,
    after_j_to_i: torch.Tensor,
) -> torch.Tensor:
    """Residual around a commuting two-substitution square."""
    shapes = {tuple(value.shape) for value in (
        anchor_to_i, after_i_to_j, anchor_to_j, after_j_to_i
    )}
    if len(shapes) != 1:
        raise ValueError("cycle-effect tensors must have equal shapes")
    return anchor_to_i + after_i_to_j - anchor_to_j - after_j_to_i
