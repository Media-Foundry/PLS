"""A compact sequence-native scalar-potential student for edit search."""

from __future__ import annotations

import torch
from torch import nn

from .mutations import AMINO_ACIDS


TOKEN_BY_AMINO_ACID = {amino_acid: index + 1 for index, amino_acid in enumerate(AMINO_ACIDS)}


def encode_sequences(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    maximum = max(map(len, sequences))
    if maximum == 0:
        raise ValueError("sequences cannot be empty")
    tokens = torch.zeros(len(sequences), maximum, dtype=torch.long)
    mask = torch.zeros(len(sequences), maximum, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        try:
            encoded = [TOKEN_BY_AMINO_ACID[residue] for residue in sequence]
        except KeyError as error:
            raise ValueError(f"noncanonical amino acid: {error.args[0]}") from error
        tokens[row, :len(encoded)] = torch.tensor(encoded)
        mask[row, :len(encoded)] = True
    return tokens, mask


class EditPotentialStudent(nn.Module):
    """Transformer potential whose scalar differences define all edit effects."""

    def __init__(self, dimension=128, layers=3, heads=4, dropout=.1, max_length=2048):
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.max_length = int(max_length)
        self.amino_acid = nn.Embedding(len(AMINO_ACIDS) + 1, dimension, padding_idx=0)
        self.position = nn.Embedding(max_length, dimension)
        block = nn.TransformerEncoderLayer(
            dimension, heads, dimension * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dimension, 1),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.max_length:
            raise ValueError("tokens must be [batch, length] within max_length")
        mask = tokens.ne(0) if mask is None else mask.bool()
        if mask.shape != tokens.shape or torch.any(mask.sum(1) == 0):
            raise ValueError("mask must select at least one residue per sequence")
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
        hidden = (self.amino_acid(tokens) + self.position(positions)) * mask[..., None]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = hidden * mask[..., None]
        pooled = hidden.sum(1) / mask.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1)

