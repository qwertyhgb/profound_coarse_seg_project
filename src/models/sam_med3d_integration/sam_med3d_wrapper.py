"""Phase 1: SAM-Med3D multi-channel wrapper for PI-CAI Stage-2 refinement.

This module wraps the official SAM-Med3D model to:
- Accept multi-modal mpMRI input [B, 3, D, H, W] (T2W/ADC/HBV)
- Bypass the ImageNet preprocess (which is meaningless for medical images)
- Provide a clean forward interface for prompt-driven fine-tuning

Strategy for multi-channel input:
- Modify PatchEmbed3D in_chans from 1 to 3
- Initialize new patch_embed weight by replicating the 1ch pretrained weight
  along the channel dim (divided by 3 to preserve activation magnitude)
- All other SAM-Med3D weights load directly from pretrained checkpoint

This preserves the bulk of pretraining knowledge while enabling multi-modal input.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_sam_med3d_on_path() -> None:
    """Add SAM-Med3D repo to Python path if not already there."""
    sam_repo = Path(__file__).resolve().parents[4] / "SAM-Med3D"
    if sam_repo.is_dir() and str(sam_repo) not in sys.path:
        sys.path.insert(0, str(sam_repo))


def _expand_patch_embed_to_3ch(
    patch_embed_module: nn.Module,
    pretrained_weight_1ch: torch.Tensor,
    pretrained_bias: Optional[torch.Tensor],
    in_chans: int = 3,
) -> None:
    """Replace 1-channel PatchEmbed3D weights with channel-replicated 3ch weights.

    pretrained_weight_1ch: [embed_dim, 1, k, k, k]
    Output weight:         [embed_dim, in_chans, k, k, k]
    """
    if pretrained_weight_1ch.shape[1] != 1:
        raise ValueError(
            f"Expected 1-channel pretrained weight, got shape {pretrained_weight_1ch.shape}"
        )
    new_weight = pretrained_weight_1ch.repeat(1, in_chans, 1, 1, 1) / float(in_chans)
    with torch.no_grad():
        patch_embed_module.proj.weight.copy_(new_weight)
        if pretrained_bias is not None and patch_embed_module.proj.bias is not None:
            patch_embed_module.proj.bias.copy_(pretrained_bias)


class SAMMed3DStage2(nn.Module):
    """SAM-Med3D wrapper for prompt-conditioned 3D refinement.

    Forward inputs:
        image:        [B, in_chans, D, H, W] expected to be normalized externally
                      (typically zero-mean unit-variance per channel)
        point_coords: [B, N, 3] in (z, y, x) order, in input volume coordinates
        point_labels: [B, N] with 1 for positive, 0 for negative, -1 for padding
        boxes:        not used (PromptEncoder3D._embed_boxes is 2D-style; we use points)
        prev_mask:    [B, 1, D/8, H/8, W/8] optional low-res mask from previous iteration

    Forward outputs:
        masks_logits: [B, 1, D, H, W] (upsampled from stride-4 to input size)
        iou_pred:     [B, 1] mask quality estimate
    """

    def __init__(
        self,
        sam_checkpoint_path: Optional[str] = None,
        in_chans: int = 3,
        roi_size: int = 128,
        freeze_image_encoder: bool = False,
    ) -> None:
        super().__init__()
        _ensure_sam_med3d_on_path()
        from segment_anything.build_sam3D import build_sam3D_vit_b_ori
        from segment_anything.modeling.image_encoder3D import PatchEmbed3D

        self.in_chans = int(in_chans)
        self.roi_size = int(roi_size)

        # Build base SAM-Med3D with single-channel patch_embed first, load weights
        sam = build_sam3D_vit_b_ori(checkpoint=sam_checkpoint_path)

        # Save original 1ch weights for channel expansion
        orig_pe_weight = sam.image_encoder.patch_embed.proj.weight.detach().clone()
        orig_pe_bias = (
            sam.image_encoder.patch_embed.proj.bias.detach().clone()
            if sam.image_encoder.patch_embed.proj.bias is not None
            else None
        )

        # Replace patch_embed with multi-channel version
        new_patch_embed = PatchEmbed3D(
            kernel_size=(16, 16, 16),
            stride=(16, 16, 16),
            in_chans=self.in_chans,
            embed_dim=sam.image_encoder.patch_embed.proj.out_channels,
        )
        _expand_patch_embed_to_3ch(
            new_patch_embed,
            pretrained_weight_1ch=orig_pe_weight,
            pretrained_bias=orig_pe_bias,
            in_chans=self.in_chans,
        )
        sam.image_encoder.patch_embed = new_patch_embed

        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Run image encoder. image: [B, in_chans, D, H, W] -> [B, 384, D/16, H/16, W/16]."""
        return self.image_encoder(image)

    def forward(
        self,
        image: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        prev_mask: Optional[torch.Tensor] = None,
        return_low_res: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward.

        Returns:
            masks_logits: [B, 1, D, H, W] upsampled to input size
            iou_pred:     [B, 1] mask quality estimate
        """
        if image.shape[1] != self.in_chans:
            raise ValueError(
                f"image has {image.shape[1]} channels, expected {self.in_chans}"
            )
        input_size = image.shape[-3:]

        image_embedding = self.image_encoder(image)

        points = None
        if point_coords is not None and point_labels is not None:
            points = (point_coords, point_labels)

        sparse_emb, dense_emb = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=prev_mask,
        )

        masks_logits, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        # Upsample from stride-4 to input size
        masks_full = F.interpolate(
            masks_logits, size=input_size, mode="trilinear", align_corners=False
        )

        if return_low_res:
            return masks_full, iou_pred, masks_logits
        return masks_full, iou_pred


def build_sam_med3d_stage2(cfg: dict) -> SAMMed3DStage2:
    """Build SAMMed3DStage2 from a config dict."""
    return SAMMed3DStage2(
        sam_checkpoint_path=cfg.get("sam_checkpoint_path"),
        in_chans=cfg.get("in_chans", 3),
        roi_size=cfg.get("roi_size", 128),
        freeze_image_encoder=cfg.get("freeze_image_encoder", False),
    )
