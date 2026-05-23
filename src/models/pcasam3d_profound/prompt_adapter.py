"""Prompt adapter: convert normalized point coords to SAM-Med3D prompt encoder format.

SAM-Med3D's PromptEncoder3D expects point_coords in absolute voxel coordinates
within the input image space (e.g., [0, 128) for a 128^3 input). This adapter
handles the conversion from our normalized [0, 1] representation.

It also provides the interface to feed the coarse mask prior as a dense prompt.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PromptAdapter(nn.Module):
    """Adapt auto-generated prompts for SAM-Med3D's PromptEncoder3D.

    Converts normalized [0,1] point coordinates to absolute voxel coordinates
    and prepares the mask prior for dense prompt input.

    NOTE: SAM-Med3D's PromptEncoder3D.mask_downscaling applies 4x spatial
    downsampling (two stride-2 Conv3d layers). So the mask input must be at
    image_embedding_size (e.g., 8x8x8), and after downscaling it becomes
    image_embedding_size/4 (e.g., 2x2x2). The dense_prompt_embeddings output
    is then at that reduced size.

    However, the mask_decoder does `src = src + dense_prompt_embeddings` where
    src is at image_embedding_size. This means we need the mask input to be at
    image_embedding_size * 4 so that after 4x downscaling it matches
    image_embedding_size.

    mask_input_size = image_embedding_size * 4 = (32, 32, 32) for 8x8x8 embeddings
    """

    def __init__(
        self,
        input_image_size: tuple[int, int, int] = (128, 128, 128),
        image_embedding_size: tuple[int, int, int] = (8, 8, 8),
    ) -> None:
        super().__init__()
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        # mask_downscaling does 4x spatial reduction, so input must be 4x embedding size
        self.mask_input_size = tuple(s * 4 for s in image_embedding_size)

    def forward(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_prior: Optional[torch.Tensor] = None,
        box_coords: Optional[torch.Tensor] = None,
        box_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Convert prompts to SAM-Med3D format.

        Args:
            point_coords: [B, N, 3] normalized [0,1] coords (z, y, x)
            point_labels: [B, N] labels (1=positive, 0=negative, -1=padding)
            box_coords: [B, 2, 3] normalized [0,1] box corners, optional
            box_valid: [B] bool tensor. Boxes are used only when all batch
                items have a valid target because SAM boxes do not have a
                padding label.
            mask_prior: [B, 1, d, h, w] low-res coarse probability

        Returns:
            points: tuple of (coords_abs, labels) or None
            boxes: [B, 2, 3] absolute voxel coords or None
            masks: [B, 1, d, h, w] mask prior or None
        """
        # Convert normalized coords to absolute voxel coords
        D, H, W = self.input_image_size
        scale = torch.tensor(
            [D, H, W], dtype=point_coords.dtype, device=point_coords.device
        ).view(1, 1, 3)
        coords_abs = point_coords * scale  # [B, N, 3]

        points = (coords_abs, point_labels)

        boxes = None
        if box_coords is not None:
            use_boxes = True
            if box_valid is not None:
                use_boxes = bool(torch.as_tensor(box_valid, device=box_coords.device).all().item())
            if use_boxes:
                boxes = box_coords * scale

        # Mask prior: resize to mask_input_size (4x image_embedding_size)
        # so that after PromptEncoder3D's 4x downscaling it matches image_embedding_size
        masks = None
        if mask_prior is not None:
            target = self.mask_input_size
            if mask_prior.shape[2:] != target:
                masks = torch.nn.functional.interpolate(
                    mask_prior, size=target, mode="trilinear", align_corners=False
                )
            else:
                masks = mask_prior

        return points, boxes, masks
