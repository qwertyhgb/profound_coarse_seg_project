"""Reusable 3D decoder blocks."""
from __future__ import annotations
import torch
from torch import nn


def norm3d(channels: int, kind: str = "instance") -> nn.Module:
    """Build a robust 3D normalization layer."""
    if kind == "group":
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    return nn.InstanceNorm3d(channels, affine=True)


class ConvBlock3D(nn.Module):
    """Two Conv3d-Norm-GELU layers."""
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            norm3d(out_channels, norm),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            norm3d(out_channels, norm),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
