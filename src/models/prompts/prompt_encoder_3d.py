"""3D prompt encoding utilities for Stage-2 refinement.

The project uses dense volumetric prompt maps rather than 2D SAM tokens in the
current Stage 2: a coarse probability map, a 3D box prior, and a center-point
Gaussian. This keeps the implementation trainable on PI-CAI patches while
matching the spatial-prompt idea used by MedSAM/PCaSAM/SAM-Med3D style methods.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.models.decoders.blocks_3d import ConvBlock3D


class DensePromptEncoder3D(nn.Module):
    """Encode dense 3D prompt priors into feature channels."""

    def __init__(self, in_channels: int = 3, out_channels: int = 16, norm: str = "instance") -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, out_channels, norm=norm),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, coarse_prob: torch.Tensor, box_prior: torch.Tensor, point_prior: torch.Tensor) -> torch.Tensor:
        """Encode ``[coarse probability, box mask, point Gaussian]`` maps."""
        return self.encoder(torch.cat([coarse_prob, box_prior, point_prior], dim=1))


# Backward-compatible research name used in configs/docs.
PromptEncoder3D = DensePromptEncoder3D


def build_box_prior(bbox_zyxzyx: Sequence[int], shape_zyx: Sequence[int]) -> np.ndarray:
    """Create a binary 3D box prior with shape ``[1,D,H,W]``."""
    d, h, w = [int(v) for v in shape_zyx]
    z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox_zyxzyx]
    prior = np.zeros((1, d, h, w), dtype=np.float32)
    prior[:, max(0, z0):min(d, z1), max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1.0
    return prior


def build_point_prior(center_zyx: Sequence[float], shape_zyx: Sequence[int], sigma: float = 3.0) -> np.ndarray:
    """Create a center-point Gaussian prior with shape ``[1,D,H,W]``."""
    d, h, w = [int(v) for v in shape_zyx]
    zc, yc, xc = [float(v) for v in center_zyx]
    z = np.arange(d, dtype=np.float32)[:, None, None]
    y = np.arange(h, dtype=np.float32)[None, :, None]
    x = np.arange(w, dtype=np.float32)[None, None, :]
    sigma2 = max(float(sigma) ** 2, 1e-6)
    prior = np.exp(-((z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2) / (2.0 * sigma2))
    return prior[None].astype(np.float32)


def build_dense_prompt_priors(
    coarse_prob: np.ndarray,
    bbox_zyxzyx: Sequence[int],
    center_zyx: Sequence[float],
    point_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coarse, box, and point prompt priors as ``[1,D,H,W]`` arrays."""
    coarse = np.asarray(coarse_prob, dtype=np.float32)
    if coarse.ndim == 3:
        coarse = coarse[None]
    if coarse.ndim != 4 or coarse.shape[0] != 1:
        raise ValueError(f"Expected coarse probability [D,H,W] or [1,D,H,W], got {coarse_prob.shape}")
    shape = coarse.shape[1:]
    return coarse, build_box_prior(bbox_zyxzyx, shape), build_point_prior(center_zyx, shape, sigma=point_sigma)
