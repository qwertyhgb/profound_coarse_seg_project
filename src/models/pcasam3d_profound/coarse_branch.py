"""Lightweight top-down coarse proposal decoder for PCaSAM-3D-ProFound.

The coarse branch is a prompt-source decoder, not the final segmentation model.
It follows the FPN/UNet idea of top-down semantic propagation with shallow
spatial detail, and exposes auxiliary logits for deep supervision during
coarse pretraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseBranch(nn.Module):
    """FPN-style coarse segmentation head from ProFound multi-scale features.

    Produces a full-resolution coarse logit map, optional auxiliary logits, and
    a high-resolution proposal feature used by the objectness head.
    """

    def __init__(
        self,
        encoder_channels: list[int] | None = None,
        hidden_dim: int = 64,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        encoder_channels = encoder_channels or [96, 192, 384, 768]
        self.hidden_dim = int(hidden_dim)

        self.lateral1 = _lateral_block(encoder_channels[0], hidden_dim)
        self.lateral2 = _lateral_block(encoder_channels[1], hidden_dim)
        self.lateral3 = _lateral_block(encoder_channels[2], hidden_dim)
        self.lateral4 = _lateral_block(encoder_channels[3], hidden_dim)

        self.smooth3 = _smooth_block(hidden_dim)
        self.smooth2 = _smooth_block(hidden_dim)
        self.smooth1 = _smooth_block(hidden_dim)

        self.context = nn.Sequential(
            nn.Conv3d(hidden_dim * 4, hidden_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.GELU(),
        )
        self.head = nn.Conv3d(hidden_dim, out_channels, 1)
        self.aux_head2 = nn.Conv3d(hidden_dim, out_channels, 1)
        self.aux_head3 = nn.Conv3d(hidden_dim, out_channels, 1)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        input_shape: tuple[int, int, int],
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | list[torch.Tensor]]:
        """Produce coarse logits at input resolution.

        Args:
            features: ProFound multi-scale feature dict with stage1..stage4.
            input_shape: target full-resolution shape (D, H, W).
            return_features: when true, return logits, aux logits, and proposal
                features for objectness sharing.
        """
        c1 = self.lateral1(features["stage1"])
        c2 = self.lateral2(features["stage2"])
        c3 = self.lateral3(features["stage3"])
        p4 = self.lateral4(features["stage4"])

        p3 = self.smooth3(c3 + F.interpolate(p4, size=c3.shape[2:], mode="trilinear", align_corners=False))
        p2 = self.smooth2(c2 + F.interpolate(p3, size=c2.shape[2:], mode="trilinear", align_corners=False))
        p1 = self.smooth1(c1 + F.interpolate(p2, size=c1.shape[2:], mode="trilinear", align_corners=False))

        target = p1.shape[2:]
        fused = torch.cat([
            p1,
            F.interpolate(p2, size=target, mode="trilinear", align_corners=False),
            F.interpolate(p3, size=target, mode="trilinear", align_corners=False),
            F.interpolate(p4, size=target, mode="trilinear", align_corners=False),
        ], dim=1)
        proposal_feature = self.context(fused)

        logits = self.head(proposal_feature)
        if logits.shape[2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode="trilinear", align_corners=False)

        if not return_features:
            return logits

        aux_logits = [self.aux_head2(p2), self.aux_head3(p3)]
        aux_logits = [
            F.interpolate(aux, size=input_shape, mode="trilinear", align_corners=False)
            if aux.shape[2:] != input_shape else aux
            for aux in aux_logits
        ]
        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "proposal_feature": proposal_feature,
        }


def _lateral_block(in_channels: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
        nn.InstanceNorm3d(hidden_dim, affine=True),
        nn.GELU(),
    )


def _smooth_block(hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
        nn.InstanceNorm3d(hidden_dim, affine=True),
        nn.GELU(),
    )
