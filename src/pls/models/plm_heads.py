"""Shared frozen-PLM projector with endpoint-specific prediction heads."""

from __future__ import annotations

import torch
from torch import nn


TASKS = ("uesolds", "pdbsol", "esol")


class PLMDatasetHeads(nn.Module):
    def __init__(self, input_dimension: int = 1280, hidden_dimension: int = 256,
                 representation_dimension: int = 128, dropout: float = .2,
                 task_adapters: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dimension), nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dimension, representation_dimension),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.task_adapters = task_adapters
        self.adapters = nn.ModuleDict({task: nn.Sequential(
            nn.LayerNorm(representation_dimension), nn.Linear(representation_dimension, representation_dimension),
            nn.GELU(), nn.Dropout(dropout)) for task in TASKS}) if task_adapters else nn.ModuleDict()
        self.heads = nn.ModuleDict({task: nn.Linear(representation_dimension, 1) for task in TASKS})

    def forward(self, embeddings: torch.Tensor, task: str) -> torch.Tensor:
        representation = self.encoder(embeddings)
        if self.task_adapters:
            representation = representation + self.adapters[task](representation)
        return self.heads[task](representation).squeeze(-1)
