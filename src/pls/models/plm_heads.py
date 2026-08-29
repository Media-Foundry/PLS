"""Shared frozen-PLM projector with endpoint-specific prediction heads."""

from __future__ import annotations

import torch
from torch import nn


TASKS = ("uesolds", "pdbsol", "esol")


class PLMDatasetHeads(nn.Module):
    def __init__(self, input_dimension: int = 1280, hidden_dimension: int = 256,
                 representation_dimension: int = 128, dropout: float = .2,
                 task_adapters: bool = False, latent_endpoint: bool = False):
        super().__init__()
        if latent_endpoint and task_adapters:
            raise ValueError("latent_endpoint requires a task-independent shared representation")
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dimension), nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dimension, representation_dimension),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.task_adapters = task_adapters
        self.latent_endpoint = latent_endpoint
        self.adapters = nn.ModuleDict({task: nn.Sequential(
            nn.LayerNorm(representation_dimension), nn.Linear(representation_dimension, representation_dimension),
            nn.GELU(), nn.Dropout(dropout)) for task in TASKS}) if task_adapters else nn.ModuleDict()
        self.heads = nn.ModuleDict({task: nn.Linear(representation_dimension, 1) for task in TASKS})
        if latent_endpoint:
            self.latent_score = nn.Linear(representation_dimension, 1)
            self.endpoint_log_slopes = nn.ParameterDict({task: nn.Parameter(torch.zeros(())) for task in TASKS})
            self.endpoint_intercepts = nn.ParameterDict({task: nn.Parameter(torch.zeros(())) for task in TASKS})

    def observe_latent(self, latent: torch.Tensor, task: str) -> torch.Tensor:
        slope = nn.functional.softplus(self.endpoint_log_slopes[task]) + 1e-4
        observation = slope * latent + self.endpoint_intercepts[task]
        return torch.sigmoid(observation) if task == "esol" else observation

    def forward(self, embeddings: torch.Tensor, task: str) -> torch.Tensor:
        representation = self.encoder(embeddings)
        if self.task_adapters:
            representation = representation + self.adapters[task](representation)
        if self.latent_endpoint:
            return self.observe_latent(self.latent_score(representation).squeeze(-1), task)
        return self.heads[task](representation).squeeze(-1)
