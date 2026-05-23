"""Modality-Aware Cross-Attention Fusion for 3D multi-modal mpMRI.

Inspired by:
- PCaSAM (Nature Digital Medicine 2025): per-modality encoding + CrossAttentionFusion
  at feature level for prostate cancer segmentation
- AFF / iAFF (Dai et al., WACV 2021): iterative attentional feature fusion with
  multi-scale channel attention (MS-CAM)
- TMA-TransBTS (2025): 3D multi-scale cross-attention for brain tumor segmentation

Design rationale for our 3D adaptation:
- PCaSAM encodes each modality separately through a full ViT (4x cost). This is
  infeasible for 3D volumes. Instead, we apply modality-aware attention AFTER the
  shared encoder, operating on the compressed feature maps (8³ spatial).
- We decompose the 3-channel input into modality-specific feature streams using
  learned channel projections, then apply cross-attention between modalities.
- The MS-CAM (multi-scale channel attention) from AFF captures both local and
  global channel dependencies, which is important because T2W/ADC/HBV have
  fundamentally different intensity semantics.
- The iterative refinement from iAFF (two-pass attention) improves fusion quality
  without adding much computation at the compressed spatial resolution.

Module placement in PCaSAM-3D-ProFound:
  ProFound features (stage3/4) → Feature Bridge → ModalityCrossAttention3D → SAM Decoder
  This is AFTER the shared encoder but BEFORE the mask decoder, operating at 8³.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_CAM_3D(nn.Module):
    """Multi-Scale Channel Attention Module (3D version).

    From AFF (Dai et al., WACV 2021): combines local (per-voxel) and global
    (volume-level) channel attention to capture both fine-grained and holistic
    modality relationships.

    Local branch: 1×1×1 Conv → ReLU → 1×1×1 Conv (per-voxel channel recalibration)
    Global branch: GAP → 1×1×1 Conv → ReLU → 1×1×1 Conv (SE-style global attention)
    Output: local + global (before sigmoid, used as attention logits)
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        inter_channels = max(channels // reduction, 16)

        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention logits (not yet sigmoid)."""
        return self.local_att(x) + self.global_att(x)


class ModalityCrossAttention3D(nn.Module):
    """Modality-aware cross-attention fusion for 3D feature maps.

    This module operates on the image embedding AFTER the Feature Bridge
    (shape: [B, C, D, H, W] where C=384, spatial=8³).

    Strategy (inspired by PCaSAM + iAFF):
    1. Decompose the fused feature into 3 modality-aware streams using learned
       projections (simulating per-modality encoding without 3x encoder cost)
    2. Apply cross-attention: each stream attends to the other two
    3. Use iterative attentional fusion (iAFF-style) to merge streams back
    4. Residual connection to preserve original information

    The "modality decomposition" is learned — the network discovers which
    feature dimensions correspond to which modality's contribution. This is
    reasonable because ProFound-Conv's stem processes 3 input channels, and
    different feature channels naturally encode different modality information.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_modalities: int = 3,
        num_heads: int = 6,
        reduction: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads

        # ─── Modality Decomposition ───
        # Learn to project the fused feature into modality-specific streams
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim, 1, bias=False),
                nn.GroupNorm(1, embed_dim),
                nn.GELU(),
            )
            for _ in range(num_modalities)
        ])

        # ─── Cross-Attention between modalities ───
        # Each modality attends to the concatenation of other modalities
        # Using efficient linear attention to keep computation manageable at 8³
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention3D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_modalities)
        ])

        # ─── Iterative Attentional Feature Fusion (iAFF-style) ───
        # First pass: weighted sum with MS-CAM attention
        self.ms_cam_1 = MS_CAM_3D(embed_dim, reduction)
        self.fusion_proj_1 = nn.Conv3d(embed_dim, num_modalities * embed_dim, 1, bias=False)

        # Second pass: refine with another MS-CAM
        self.ms_cam_2 = MS_CAM_3D(embed_dim, reduction)
        self.fusion_proj_2 = nn.Conv3d(embed_dim, num_modalities * embed_dim, 1, bias=False)

        # ─── Output projection with residual ───
        self.output_norm = nn.GroupNorm(1, embed_dim)
        self.output_proj = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, 1, bias=False),
        )

        # Learnable residual scale (initialized small for stable training)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply modality-aware cross-attention fusion.

        Args:
            x: [B, C, D, H, W] image embedding from Feature Bridge

        Returns:
            [B, C, D, H, W] enhanced embedding with cross-modal interactions
        """
        identity = x

        # ─── Step 1: Modality Decomposition ───
        modality_features = [proj(x) for proj in self.modality_projections]
        # Each: [B, C, D, H, W]

        # ─── Step 2: Cross-Attention ───
        # Each modality attends to the other two
        enhanced_features = []
        for i in range(self.num_modalities):
            # Context = average of other modalities
            others = [modality_features[j] for j in range(self.num_modalities) if j != i]
            context = sum(others) / len(others)
            enhanced = self.cross_attn_layers[i](modality_features[i], context)
            enhanced_features.append(enhanced)

        # ─── Step 3: iAFF-style Iterative Fusion ───
        # First pass
        feat_sum = sum(enhanced_features)
        attn_logits_1 = self.ms_cam_1(feat_sum)
        weights_1 = self.fusion_proj_1(attn_logits_1)
        weights_1 = torch.chunk(weights_1, self.num_modalities, dim=1)

        fused_1 = sum(
            enhanced_features[i] * torch.sigmoid(weights_1[i])
            for i in range(self.num_modalities)
        )

        # Second pass (iterative refinement)
        attn_logits_2 = self.ms_cam_2(fused_1)
        weights_2 = self.fusion_proj_2(attn_logits_2)
        weights_2 = torch.chunk(weights_2, self.num_modalities, dim=1)

        fused_2 = sum(
            enhanced_features[i] * torch.sigmoid(weights_2[i])
            for i in range(self.num_modalities)
        )

        # ─── Step 4: Output with residual ───
        out = self.output_norm(fused_2)
        out = self.output_proj(out)

        return identity + self.residual_scale * out


class CrossModalAttention3D(nn.Module):
    """Efficient cross-modal attention for 3D feature maps.

    Uses multi-head attention where query comes from one modality and
    key/value come from another modality (context).

    At 8³ = 512 tokens, standard attention is computationally feasible.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention: query attends to context.

        Args:
            query: [B, C, D, H, W] - the modality being enhanced
            context: [B, C, D, H, W] - the other modality(ies) providing context

        Returns:
            [B, C, D, H, W] - enhanced query
        """
        B, C, D, H, W = query.shape
        N = D * H * W  # number of spatial tokens

        # Reshape to sequence: [B, N, C]
        q = query.flatten(2).permute(0, 2, 1)   # [B, N, C]
        kv = context.flatten(2).permute(0, 2, 1)  # [B, N, C]

        # Pre-norm
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)

        # Project
        q_proj = self.q_proj(q_normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(kv_normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(kv_normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Each: [B, num_heads, N, head_dim]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q_proj @ k_proj.transpose(-2, -1)) * scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = (attn @ v_proj).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.out_proj(out)

        # Residual + FFN
        q = q + out
        q = q + self.ffn(q)

        # Reshape back to volume
        return q.permute(0, 2, 1).view(B, C, D, H, W)
