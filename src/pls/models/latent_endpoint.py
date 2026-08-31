"""Monotonic observation heads over a shared, scale-unidentified latent score.

When assay residuals are enabled this is a shared-latent plus endpoint-specific
residual model; it is not a strict monotonic observation of one identifiable
intrinsic-solubility variable.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LatentEndpointModel(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 512, dropout: float = .2, endpoints: int = 3, label_noise_max: float = 0., assay_residual_dimension: int = 0, assay_residual_scale: float = 0.):
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
        self.assay_residual_dimension = int(assay_residual_dimension)
        self.assay_residual_scale = float(assay_residual_scale)
        if self.assay_residual_dimension < 0 or self.assay_residual_scale < 0:
            raise ValueError("assay residual dimension and scale must be non-negative")
        if bool(self.assay_residual_dimension) != bool(self.assay_residual_scale):
            raise ValueError("assay residual dimension and scale must both be enabled")
        if self.assay_residual_dimension:
            self.assay_residual_encoder = nn.Sequential(
                nn.LayerNorm(input_dimension),
                nn.Linear(input_dimension, self.assay_residual_dimension),
                nn.Tanh(),
            )
            self.assay_residual_weights = nn.Parameter(torch.zeros(endpoints, self.assay_residual_dimension))
        self.label_noise_max = float(label_noise_max)
        if not 0 <= self.label_noise_max < .5:
            raise ValueError("label noise maximum must be in [0, 0.5)")
        self.raw_false_positive = nn.Parameter(torch.full((endpoints,), -3.))
        self.raw_false_negative = nn.Parameter(torch.full((endpoints,), -3.))

    @property
    def slopes(self) -> torch.Tensor:
        return F.softplus(self.raw_slopes) + 1e-4

    def latent(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features).squeeze(-1)

    def observation_logits(self, features: torch.Tensor, endpoint: torch.Tensor) -> torch.Tensor:
        score = self.latent(features)
        logits = self.slopes[endpoint] * score + self.intercepts[endpoint]
        if self.assay_residual_dimension:
            residual_features = self.assay_residual_encoder(features)
            residual = (residual_features * self.assay_residual_weights[endpoint]).sum(-1)
            logits = logits + self.assay_residual_scale * torch.tanh(residual)
        return logits

    def assay_residual_penalty(self) -> torch.Tensor:
        if not self.assay_residual_dimension:
            return self.raw_slopes.new_zeros(())
        return self.assay_residual_weights.square().mean()

    @property
    def false_positive(self) -> torch.Tensor:
        return self.label_noise_max * torch.sigmoid(self.raw_false_positive)

    @property
    def false_negative(self) -> torch.Tensor:
        return self.label_noise_max * torch.sigmoid(self.raw_false_negative)

    def forward(self, features: torch.Tensor, endpoint: torch.Tensor, continuous_endpoint: int = 2) -> torch.Tensor:
        logits = self.observation_logits(features, endpoint)
        if self.label_noise_max:
            probability = torch.sigmoid(logits); observed = self.false_positive[endpoint] + (1 - self.false_positive[endpoint] - self.false_negative[endpoint]) * probability; noisy_logits = torch.logit(observed.clamp(1e-6, 1 - 1e-6))
        else:
            noisy_logits = logits
        return torch.where(endpoint == continuous_endpoint, torch.sigmoid(logits), noisy_logits)
