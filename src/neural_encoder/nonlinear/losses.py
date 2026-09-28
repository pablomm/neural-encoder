"""PyTorch objectives for learning agreement across paired views."""

from collections.abc import Sequence
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["symmetric_info_nce", "cosine_pull", "MultiViewContrastiveLoss"]


def symmetric_info_nce(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.09,
) -> torch.Tensor:
    """Symmetric InfoNCE with paired rows and in-batch negatives.

    Each row must represent a distinct sample within a view. Both inputs
    have shape (batch, features), with at least two paired samples.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    logits = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).T / temperature
    targets = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def cosine_pull(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Mean one-minus-cosine similarity between corresponding rows."""
    return (1 - (F.normalize(z1, dim=-1) * F.normalize(z2, dim=-1)).sum(dim=-1)).mean()


class MultiViewContrastiveLoss(nn.Module):
    """Average symmetric InfoNCE and cosine pull over all view pairs.

    forward(outputs, inputs=None) returns a scalar tensor. outputs is a
    sequence of aligned (batch, features) tensors. inputs is accepted for
    compatibility with objectives using original representations, but is
    unused here. nce_weight and pull_weight must be nonnegative, with at
    least one positive weight.
    """

    def __init__(
        self, *, temperature: float = 0.09, nce_weight: float = 1.0,
        pull_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if nce_weight < 0 or pull_weight < 0 or nce_weight + pull_weight <= 0:
            raise ValueError("Loss weights must be nonnegative with at least one positive.")
        self.temperature = temperature
        self.nce_weight = nce_weight
        self.pull_weight = pull_weight

    def forward(
        self, outputs: Sequence[torch.Tensor], *, inputs: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(outputs) < 2:
            raise ValueError("At least two views are required.")
        if outputs[0].ndim != 2 or any(z.shape != outputs[0].shape for z in outputs):
            raise ValueError("View tensors must be equally shaped (batch, features) matrices.")
        if self.nce_weight and len(outputs[0]) < 2:
            raise ValueError("InfoNCE requires at least two samples.")
        losses = []
        for first, second in combinations(outputs, 2):
            loss = first.new_zeros(())
            if self.nce_weight:
                loss = loss + self.nce_weight * symmetric_info_nce(first, second, self.temperature)
            if self.pull_weight:
                loss = loss + self.pull_weight * cosine_pull(first, second)
            losses.append(loss)
        return torch.stack(losses).mean()
