"""High-resolution refinement head for PCaSAM-3D-ProFound.

SAM-Med3D reasons in a low-resolution token space. This module injects
ProFound stage1/stage2 spatial detail after the SAM mask decoder so small
lesions are not represented only by trilinearly upsampled 8^3 logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighResRefinementHead3D(nn.Module):
    """Refine SAM logits using high-resolution ProFound features.

    The head predicts a residual logit map at stage1 resolution and upsamples
    it to the input grid. A small learnable residual scale keeps initialization
    conservative while allowing the model to recover small-object detail.
    """

    def __init__(
        self,
        encoder_channels: list[int] | None = None,
        hidden_dim: int = 32,
        residual_init: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_channels = encoder_channels or [96, 192, 384, 768]
        self.stage1_proj = _proj_block(encoder_channels[0], hidden_dim)
        self.stage2_proj = _proj_block(encoder_channels[1], hidden_dim)
        self.logit_proj = nn.Sequential(
            nn.Conv3d(2, hidden_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(hidden_dim * 3, hidden_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_dim, 1, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        sam_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        features: dict[str, torch.Tensor],
        input_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = features["stage1"].shape[2:]
        f1 = self.stage1_proj(features["stage1"])
        f2 = self.stage2_proj(features["stage2"])
        f2 = F.interpolate(f2, size=target, mode="trilinear", align_corners=False)

        sam_s = F.interpolate(sam_logits, size=target, mode="trilinear", align_corners=False)
        coarse_s = F.interpolate(coarse_logits, size=target, mode="trilinear", align_corners=False)
        logit_feat = self.logit_proj(torch.cat([sam_s, coarse_s], dim=1))

        residual = self.fusion(torch.cat([f1, f2, logit_feat], dim=1))
        if residual.shape[2:] != input_shape:
            residual = F.interpolate(residual, size=input_shape, mode="trilinear", align_corners=False)
        refined = sam_logits + self.residual_scale * residual
        return refined, residual


def _proj_block(in_channels: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
        nn.InstanceNorm3d(hidden_dim, affine=True),
        nn.GELU(),
    )
