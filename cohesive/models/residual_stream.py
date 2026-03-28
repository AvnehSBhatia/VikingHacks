"""Four-block gated residual stream: x ← x + σ(gate(x)) ⊙ proj(x) + bias."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        return x + g * self.proj(x) + self.bias


class ResidualCorrector4(nn.Module):
    """
    Four residual blocks, then L2-normalise.

    ``x`` is [B, D] or [D] (batch added).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(GatedResidualBlock(dim) for _ in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for b in self.blocks:
            x = b(x)
        return F.normalize(x, dim=-1)
