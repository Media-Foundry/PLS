"""Confidence-gated invariant baseline for residue-level V4 structure features."""

from __future__ import annotations

import torch
from torch import nn

from pls.features.structure_v4_schema import (PHYSCHEM_DIMENSION,
                                              PLS_SPATIAL_SCALAR_DIMENSION,
                                              SPATIAL_VECTOR_CHANNELS)


class StructureDescriptorEncoder(nn.Module):
    def __init__(self, hidden_dimension: int = 256, output_dimension: int = 128,
                 dropout: float = .15):
        super().__init__()
        vector_invariants = SPATIAL_VECTOR_CHANNELS + SPATIAL_VECTOR_CHANNELS * (SPATIAL_VECTOR_CHANNELS + 1) // 2
        self.residue_encoder = nn.Sequential(
            nn.LayerNorm(PHYSCHEM_DIMENSION + PLS_SPATIAL_SCALAR_DIMENSION + vector_invariants),
            nn.Linear(PHYSCHEM_DIMENSION + PLS_SPATIAL_SCALAR_DIMENSION + vector_invariants, hidden_dimension),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dimension, output_dimension), nn.GELU())
        self.attention = nn.Linear(output_dimension + 1, 1)
        self.confidence_gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())

    @staticmethod
    def vector_invariants(vectors: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(vectors, dim=-1)
        gram = torch.einsum("...ic,...jc->...ij", vectors, vectors)
        indices = torch.triu_indices(gram.shape[-2], gram.shape[-1], device=gram.device)
        return torch.cat([norms, gram[..., indices[0], indices[1]]], dim=-1)

    def forward(self, physchem, spatial_scalars, spatial_vectors, mask=None):
        plddt = spatial_scalars[..., -1]
        residue = self.residue_encoder(torch.cat([
            physchem, spatial_scalars, self.vector_invariants(spatial_vectors)], dim=-1))
        attention_logits = self.attention(torch.cat([residue, plddt[..., None]], dim=-1)).squeeze(-1)
        if mask is not None: attention_logits = attention_logits.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(attention_logits, dim=-1)
        pooled = torch.sum(weights[..., None] * residue, dim=-2)
        if mask is None:
            mean_confidence, low_fraction = plddt.mean(-1), (plddt < .7).float().mean(-1)
        else:
            count = mask.sum(-1).clamp_min(1)
            mean_confidence = (plddt * mask).sum(-1) / count
            low_fraction = ((plddt < .7) & mask).sum(-1) / count
        lengths = mask.sum(-1).float() if mask is not None else torch.full_like(mean_confidence, plddt.shape[-1])
        quality = torch.stack([mean_confidence, low_fraction, torch.log1p(lengths) / 10], -1)
        gate = self.confidence_gate(quality)
        return gate * pooled, {"attention": weights, "gate": gate, "quality": quality}
