"""Feature bridge: project ProFound-Conv multi-scale features to SAM-Med3D embedding space.

ProFound-Conv (ConvNeXtV2-Tiny) produces features at 4 stages:
  stage1: [B, 96,  D/4,  H/4,  W/4]
  stage2: [B, 192, D/8,  H/8,  W/8]
  stage3: [B, 384, D/16, H/16, W/16]
  stage4: [B, 768, D/32, H/32, W/32]

SAM-Med3D's mask decoder expects a single image embedding:
  [B, 384, D/16, H/16, W/16]  (for 128^3 input with patch_size=16 → 8^3 tokens)

The bridge fuses multi-scale ProFound features into this target shape using
a lightweight FPN-style aggregation followed by a projection to 384 dims.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProFoundToSAMBridge(nn.Module):
    """Project ProFound-Conv multi-scale features to SAM-Med3D embedding space.

    Strategy:
    - Upsample stage4 (768ch) to stage3 spatial resolution
    - Concatenate with stage3 (384ch)
    - Apply 1x1 conv to reduce to embed_dim (384)
    - Optionally incorporate stage2 via lateral connection

    This produces a dense feature map compatible with SAM-Med3D's mask decoder.
    """

    def __init__(
        self,
        encoder_channels: list[int] | None = None,
        embed_dim: int = 384,
        target_spatial: tuple[int, int, int] = (8, 8, 8),
        use_stage2: bool = True,
        norm: str = "layer",
    ) -> None:
        super().__init__()
        encoder_channels = encoder_channels or [96, 192, 384, 768]
        self.embed_dim = embed_dim
        self.target_spatial = target_spatial
        self.use_stage2 = use_stage2

        # Lateral projections to embed_dim
        self.lateral4 = nn.Sequential(
            nn.Conv3d(encoder_channels[3], embed_dim, 1, bias=False),
            _norm3d(embed_dim, norm),
            nn.GELU(),
        )
        self.lateral3 = nn.Sequential(
            nn.Conv3d(encoder_channels[2], embed_dim, 1, bias=False),
            _norm3d(embed_dim, norm),
            nn.GELU(),
        )
        if use_stage2:
            self.lateral2 = nn.Sequential(
                nn.Conv3d(encoder_channels[1], embed_dim, 1, bias=False),
                _norm3d(embed_dim, norm),
                nn.GELU(),
            )
        else:
            self.lateral2 = None

        # Fusion conv after concatenation
        fusion_in = embed_dim * (3 if use_stage2 else 2)
        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_in, embed_dim, 3, padding=1, bias=False),
            _norm3d(embed_dim, norm),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, 1, bias=False),
            _norm3d(embed_dim, norm),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert ProFound multi-scale features to SAM-compatible embedding.

        Args:
            features: dict with keys stage1..stage4, each [B, C_i, D_i, H_i, W_i]

        Returns:
            [B, embed_dim, *target_spatial] feature map for SAM mask decoder
        """
        f4 = self.lateral4(features["stage4"])
        f3 = self.lateral3(features["stage3"])

        # Resize all to target spatial
        target = self.target_spatial
        f4 = F.interpolate(f4, size=target, mode="trilinear", align_corners=False)
        f3 = F.interpolate(f3, size=target, mode="trilinear", align_corners=False)

        if self.use_stage2 and self.lateral2 is not None:
            f2 = self.lateral2(features["stage2"])
            f2 = F.interpolate(f2, size=target, mode="trilinear", align_corners=False)
            fused = torch.cat([f4, f3, f2], dim=1)
        else:
            fused = torch.cat([f4, f3], dim=1)

        return self.fusion(fused)


def _norm3d(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "layer":
        return nn.GroupNorm(1, channels)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    elif norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "group":
        return nn.GroupNorm(min(32, channels), channels)
    else:
        return nn.GroupNorm(1, channels)
