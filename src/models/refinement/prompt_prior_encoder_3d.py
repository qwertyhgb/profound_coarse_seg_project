"""3D prompt-prior utilities for Stage-2 refinement.

The module follows the promptable segmentation idea used by SAM/MedSAM but uses
volumetric dense priors that are simple and trainable in this project: a box mask
and a center-point Gaussian map.
"""
from __future__ import annotations
import torch
from torch import nn

from src.models.decoders.blocks_3d import ConvBlock3D


class PromptPriorEncoder3D(nn.Module):
    """Encode dense 3D prompt priors into feature channels."""

    def __init__(self, in_channels: int = 3, out_channels: int = 16, norm: str = "instance") -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, out_channels, norm=norm),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, coarse_prob: torch.Tensor, box_prior: torch.Tensor, point_prior: torch.Tensor) -> torch.Tensor:
        """Encode [coarse probability, box mask, point Gaussian] maps."""
        return self.encoder(torch.cat([coarse_prob, box_prior, point_prior], dim=1))
