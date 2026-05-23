"""UNetR3D-style coarse segmentation decoder."""
from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from .blocks_3d import ConvBlock3D


class UNetR3DStyleCoarseDecoder(nn.Module):
    """UNet-like 3D decoder for coarse lesion logits.

    The decoder accepts either multi-scale feature dictionaries with stage1..stage4
    or a single bottleneck feature under stage4. Final logits are always resized to
    input_shape, preventing downstream shape mismatch in the loss.
    """
    def __init__(self, encoder_channels: list[int], decoder_channels: list[int], out_channels: int = 1, norm: str = "instance") -> None:
        super().__init__()
        if len(encoder_channels) < 1:
            raise ValueError("encoder_channels must contain at least one stage channel count")
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.deep_proj = ConvBlock3D(encoder_channels[-1], decoder_channels[0], norm=norm)
        blocks = []
        in_ch = decoder_channels[0]
        skip_channels = list(reversed(encoder_channels[:-1]))
        for i, out_ch in enumerate(decoder_channels[1:]):
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            blocks.append(ConvBlock3D(in_ch + skip_ch, out_ch, norm=norm))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.single_scale_blocks = nn.ModuleList([ConvBlock3D(decoder_channels[-1], decoder_channels[-1], norm=norm) for _ in range(2)])
        self.head = nn.Conv3d(decoder_channels[-1], out_channels, 1)

    def forward(self, features: dict[str, torch.Tensor] | torch.Tensor, input_shape: tuple[int, int, int]) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            features = {"stage4": features}
        ordered = self._ordered_features(features)
        x = self.deep_proj(ordered[-1])
        skips = list(reversed(ordered[:-1]))
        if skips:
            for i, block in enumerate(self.blocks):
                skip = skips[i] if i < len(skips) else None
                if skip is not None:
                    x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
                    x = torch.cat([x, skip], dim=1)
                else:
                    x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
                x = block(x)
        else:
            for block in self.single_scale_blocks:
                x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
                x = block(x)
        logits = self.head(x)
        if logits.shape[2:] != tuple(input_shape):
            logits = F.interpolate(logits, size=input_shape, mode="trilinear", align_corners=False)
        return logits

    @staticmethod
    def _ordered_features(features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        keys = [k for k in ("stage1", "stage2", "stage3", "stage4") if k in features]
        if not keys:
            raise KeyError("Encoder features must contain at least 'stage4' or stage1..stage4")
        return [features[k] for k in keys]
