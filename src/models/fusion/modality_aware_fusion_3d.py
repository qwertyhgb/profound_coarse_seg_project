"""Lightweight modality-aware fusion for T2W/ADC/HBV mpMRI.

The module keeps the output as three channels, so the ProFound-Conv encoder can
still use its pretrained first convolution. It is identity-preserving at
initialization and learns small gated residual corrections instead of replacing
or rescaling the foundation encoder input space.
"""
from __future__ import annotations
import torch
from torch import nn


class ModalityAwareFusion3D(nn.Module):
    """Residual 3D fusion stem for three-channel prostate mpMRI.

    Input/output shape: [B, 3, D, H, W]. The residual design is intentionally
    conservative: the pretrained ProFound encoder still receives a T2W/ADC/HBV
    shaped tensor, while the network can recalibrate modalities case-by-case.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 16,
        residual_scale: float = 0.1,
        modality_dropout_prob: float = 0.0,
        modality_dropout_keep_at_least: int = 2,
        modality_dropout_probs: list[float] | tuple[float, float, float] | None = None,
        modality_gate_prior: list[float] | tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("ModalityAwareFusion3D is designed for [T2W, ADC, HBV] three-channel input.")
        self.residual_scale = float(residual_scale)
        self.modality_dropout_prob = float(modality_dropout_prob)
        self.modality_dropout_keep_at_least = int(modality_dropout_keep_at_least)
        if modality_dropout_probs is not None and len(modality_dropout_probs) != in_channels:
            raise ValueError("modality_dropout_probs must have one value per modality channel.")
        drop_probs = modality_dropout_probs or [self.modality_dropout_prob] * in_channels
        self.register_buffer("modality_dropout_probs", torch.tensor(drop_probs, dtype=torch.float32))
        if modality_gate_prior is not None and len(modality_gate_prior) != in_channels:
            raise ValueError("modality_gate_prior must have one value per modality channel.")
        self.modality_gate_prior = modality_gate_prior
        self.local_fusion = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1),
        )
        # Start as an exact identity adapter. This protects the frozen ProFound
        # input distribution and lets training opt into modality recalibration.
        nn.init.zeros_(self.local_fusion[-1].weight)
        nn.init.zeros_(self.local_fusion[-1].bias)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, max(hidden_channels // 2, 4), kernel_size=1),
            nn.GELU(),
            nn.Conv3d(max(hidden_channels // 2, 4), in_channels, kernel_size=1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1),
        )
        nn.init.zeros_(self.global_gate[-1].weight)
        nn.init.zeros_(self.global_gate[-1].bias)
        if modality_gate_prior is not None:
            prior = torch.tensor([min(max(float(p), 1e-4), 1.0 - 1e-4) for p in modality_gate_prior])
            with torch.no_grad():
                self.global_gate[-1].bias.copy_(torch.logit(prior))
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)
        self.last_modality_weights: torch.Tensor | None = None

    def _apply_modality_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask modalities during training for robustness."""
        if not self.training:
            return x
        B, C = x.shape[:2]
        drop_probs = self.modality_dropout_probs.to(device=x.device, dtype=x.dtype).view(1, C)
        if float(drop_probs.max().item()) <= 0:
            return x
        drop = torch.rand((B, C), device=x.device) < drop_probs
        min_keep = max(1, min(self.modality_dropout_keep_at_least, C))
        keep = torch.ones((B, C), device=x.device, dtype=x.dtype)
        for b in range(B):
            if int((~drop[b]).sum().item()) < min_keep:
                scores = torch.rand(C, device=x.device)
                keep_idx = torch.topk(scores, k=min_keep).indices
                drop[b].fill_(True)
                drop[b, keep_idx] = False
        keep = keep.masked_fill(drop, 0.0)
        keep = keep * (C / keep.sum(dim=1, keepdim=True).clamp_min(1.0))
        return x * keep.view(B, C, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_modality_dropout(x)
        gate = torch.sigmoid(self.global_gate(x) + self.spatial_gate(x))
        correction = self.local_fusion(x)
        self.last_modality_weights = gate.detach().mean(dim=(2, 3, 4))
        return x + self.residual_scale * gate * correction
