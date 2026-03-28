"""
Learnable deformation S and residual corrector: z_hat = ResNet(S @ z_sum).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_stream import ResidualCorrector4


class HalluCorrectorModule(nn.Module):
    """
    z_def = normalize(z @ S.T)   (batch: [B,D] @ [D,D].T)
    z_hat = ResidualCorrector4(z_def)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.S = nn.Parameter(torch.eye(dim))
        self.residual = ResidualCorrector4(dim)

    def deform(self, z: torch.Tensor) -> torch.Tensor:
        """z_def = normalize(z @ S) per row."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z_def = torch.matmul(z, self.S)
        return F.normalize(z_def, dim=-1)

    def forward(self, z_summary: torch.Tensor) -> torch.Tensor:
        z_def = self.deform(z_summary)
        return self.residual(z_def)
