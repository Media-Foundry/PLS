"""Auditable protein substitution edits and mutation-neighborhood generation."""

from __future__ import annotations

from dataclasses import dataclass


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True, order=True)
class Substitution:
    position: int
    source: str
    target: str

    def __post_init__(self):
        if self.position < 0:
            raise ValueError("position must be zero-based and nonnegative")
        if self.source not in AMINO_ACIDS or self.target not in AMINO_ACIDS:
            raise ValueError("source and target must be canonical amino acids")
        if self.source == self.target:
            raise ValueError("a substitution must change the residue")


def apply_substitution(sequence: str, edit: Substitution) -> str:
    if edit.position >= len(sequence):
        raise ValueError("substitution position lies outside the sequence")
    if sequence[edit.position] != edit.source:
        raise ValueError("substitution source does not match the parent sequence")
    return sequence[:edit.position] + edit.target + sequence[edit.position + 1:]


def enumerate_single_substitutions(sequence: str):
    if not sequence or any(residue not in AMINO_ACIDS for residue in sequence):
        raise ValueError("sequence must contain only canonical amino acids")
    for position, source in enumerate(sequence):
        for target in AMINO_ACIDS:
            if target != source:
                edit = Substitution(position, source, target)
                yield edit, apply_substitution(sequence, edit)

