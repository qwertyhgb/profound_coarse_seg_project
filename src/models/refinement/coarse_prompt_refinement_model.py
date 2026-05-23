"""Prompt-conditioned 3D coarse-to-fine refinement model."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from src.models.decoders.blocks_3d import ConvBlock3D
from .prompt_prior_encoder_3d import PromptPriorEncoder3D


class DownBlock3D(nn.Module):
    """Strided downsampling followed by a ConvBlock3D."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance") -> None:
        super().__init__()
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = ConvBlock3D(out_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock3D(nn.Module):
    """Upsample, concatenate skip feature, and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm: str = "instance") -> None:
        super().__init__()
        self.block = ConvBlock3D(in_channels + skip_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class CoarsePromptRefinementModel(nn.Module):
    """3D prompt-conditioned mask refinement with optional proposal objectness.

    This is still lightweight, but it follows the SAM/MedSAM separation between
    prompt-conditioned mask prediction and candidate quality estimation: the
    mask head refines the proposed lesion, while the objectness head can later
    reject false-positive coarse components.
    """

    def __init__(
        self,
        image_channels: int = 3,
        prompt_channels: int = 16,
        base_channels: int = 32,
        channel_multipliers: tuple[int, int, int] = (1, 2, 4),
        out_channels: int = 1,
        norm: str = "instance",
        residual_with_coarse: bool = True,
        use_objectness_head: bool = True,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        self.residual_with_coarse = bool(residual_with_coarse)
        self.use_objectness_head = bool(use_objectness_head)
        self.return_dict = bool(return_dict)
        self.prompt_encoder = PromptPriorEncoder3D(in_channels=3, out_channels=prompt_channels, norm=norm)
        c1 = base_channels * channel_multipliers[0]
        c2 = base_channels * channel_multipliers[1]
        c3 = base_channels * channel_multipliers[2]
        self.stem = ConvBlock3D(image_channels + prompt_channels, c1, norm=norm)
        self.down1 = DownBlock3D(c1, c2, norm=norm)
        self.down2 = DownBlock3D(c2, c3, norm=norm)
        self.bridge = ConvBlock3D(c3, c3, norm=norm)
        self.up1 = UpBlock3D(c3, c2, c2, norm=norm)
        self.up2 = UpBlock3D(c2, c1, c1, norm=norm)
        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)
        self.objectness_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(c3, max(c3 // 2, 8)),
            nn.GELU(),
            nn.Linear(max(c3 // 2, 8), 1),
        ) if self.use_objectness_head else None

    def forward(
        self,
        image: torch.Tensor,
        coarse_prob: torch.Tensor,
        box_prior: torch.Tensor,
        point_prior: torch.Tensor,
    ):
        prompt_feat = self.prompt_encoder(coarse_prob, box_prior, point_prior)
        x0 = self.stem(torch.cat([image, prompt_feat], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        bridge = self.bridge(x2)
        x = self.up1(bridge, x1)
        x = self.up2(x, x0)
        logits = self.head(x)
        if self.residual_with_coarse:
            coarse_logits = torch.logit(coarse_prob.clamp(1e-4, 1.0 - 1e-4))
            logits = logits + coarse_logits
        if not self.return_dict:
            return logits
        out = {"logits": logits}
        if self.objectness_head is not None:
            out["objectness_logits"] = self.objectness_head(bridge).squeeze(1)
        return out
