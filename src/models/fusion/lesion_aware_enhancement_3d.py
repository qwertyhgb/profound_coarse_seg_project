"""Lightweight lesion-aware 3D feature enhancement."""
from __future__ import annotations
import torch
from torch import nn
from src.models.decoders.blocks_3d import norm3d


class LesionAwareEnhancement3D(nn.Module):
    """Residual local-context enhancement with channel and spatial gating.

    This module is intentionally lightweight and ablation-friendly. It strengthens
    local lesion-sensitive responses in the deepest ProFound feature without adding
    prompt logic or a transformer refiner.
    """
    def __init__(self, channels: int, hidden_channels: int | None = None, reduction: int = 8, norm: str = "instance") -> None:
        super().__init__()
        hidden = hidden_channels or channels
        se_hidden = max(channels // reduction, 4)
        self.proj = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            norm3d(hidden, norm),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 3, padding=1, bias=False),
            norm3d(channels, norm),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, se_hidden, 1),
            nn.GELU(),
            nn.Conv3d(se_hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv3d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        feat = feat * self.channel_gate(feat) * self.spatial_gate(feat)
        return x + feat
