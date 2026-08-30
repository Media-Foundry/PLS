"""Monotonic observation heads over a shared intrinsic-solubility latent score."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LatentEndpointModel(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 512, dropout: float = .2, endpoints: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension // 2, 1),
        )
        self.raw_slopes = nn.Parameter(torch.zeros(endpoints))
        self.intercepts = nn.Parameter(torch.zeros(endpoints))

    @property
    def slopes(self) -> torch.Tensor:
        return F.softplus(self.raw_slopes) + 1e-4

    def latent(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features).squeeze(-1)

    def observation_logits(self, features: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
        score = self.latent(features)
        return self.slopes[endpoint] * score + self.intercepts[endpoint]

    def forward(self, features: torch.Tensor, endpoint: torch.Tensor, continuous_endpoint: int = 2) -> torch.Tensor:
        logits = self.observation_logits(features, endpoint)
        return torch.where(endpoint == continuous_endpoint, torch.sigmoid(logits), logits)
