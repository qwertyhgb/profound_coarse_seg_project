"""PCaSAM-3D-ProFound: Unified end-to-end model.

Architecture:
    Input [B, 3, D, H, W] (T2W/ADC/HBV mpMRI)
        │
        ▼
    ┌─ ProFound-Conv Encoder (pretrained, optionally frozen) ─┐
    │  stage1: [B, 96,  D/4,  H/4,  W/4]                     │
    │  stage2: [B, 192, D/8,  H/8,  W/8]                     │
    │  stage3: [B, 384, D/16, H/16, W/16]                    │
    │  stage4: [B, 768, D/32, H/32, W/32]                    │
    └──────────────────────────────────────────────────────────┘
        │                           │
        ▼                           ▼
    ┌─ Coarse Branch ─┐     ┌─ Feature Bridge ─────────────┐
    │  FPN → logits   │     │  FPN → [B, 384, 8, 8, 8]    │
    │  [B,1,D,H,W]   │     │  (SAM embedding space)       │
    └─────────────────┘     └──────────────────────────────┘
        │                           │
        ▼                           │
    ┌─ Auto Prompt 3D ─┐           │
    │  point_coords     │           │
    │  mask_prior       │           │
    └───────────────────┘           │
        │                           │
        ▼                           ▼
    ┌─ SAM-Med3D Prompt Encoder ─┐  │
    │  sparse_emb, dense_emb     │  │
    └────────────────────────────┘  │
        │                           │
        ▼                           ▼
    ┌─ SAM-Med3D Mask Decoder ─────────────────────────────┐
    │  TwoWayTransformer3D + hypernetwork MLP              │
    │  → refined_logits [B, 1, D, H, W]                   │
    │  → iou_pred [B, 1]                                   │
    └──────────────────────────────────────────────────────┘

Training losses:
    1. Coarse branch: Dice + BCE (auxiliary, weighted lower)
    2. Refined mask: Dice + Focal-Tversky + BCE (primary)
    3. IoU prediction: MSE against actual Dice (optional)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..profound.profound_conv_wrapper import ProFoundConvEncoderWrapper
from ..fusion.lesion_aware_enhancement_3d import LesionAwareEnhancement3D
from ..fusion.modality_aware_fusion_3d import ModalityAwareFusion3D
from .feature_bridge import ProFoundToSAMBridge
from .coarse_branch import CoarseBranch
from .auto_prompt_3d import AutoPrompt3DFromCoarse
from .prompt_adapter import PromptAdapter
from .modality_cross_attention_3d import ModalityCrossAttention3D
from .high_res_refinement import HighResRefinementHead3D


def _ensure_sam_med3d_on_path() -> None:
    """Add SAM-Med3D repo to Python path if not already there."""
    sam_repo = Path(__file__).resolve().parents[4] / "SAM-Med3D"
    if sam_repo.is_dir() and str(sam_repo) not in sys.path:
        sys.path.insert(0, str(sam_repo))


class SelfGatedMultiScaleFusion3D(nn.Module):
    """Inject stage2/stage3 context into the 8^3 SAM embedding with a gate."""

    def __init__(
        self,
        encoder_channels: list[int],
        embed_dim: int = 384,
        residual_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.stage2_proj = nn.Sequential(
            nn.Conv3d(encoder_channels[1], embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
        )
        self.stage3_proj = nn.Sequential(
            nn.Conv3d(encoder_channels[2], embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv3d(embed_dim * 3, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, image_embedding: torch.Tensor, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if "stage2" not in features or "stage3" not in features:
            return image_embedding
        target = image_embedding.shape[2:]
        stage2 = F.interpolate(
            self.stage2_proj(features["stage2"]), size=target, mode="trilinear", align_corners=False
        )
        stage3 = F.interpolate(
            self.stage3_proj(features["stage3"]), size=target, mode="trilinear", align_corners=False
        )
        residual = self.context(torch.cat([image_embedding, stage2, stage3], dim=1))
        gate = self.gate(torch.cat([image_embedding, residual], dim=1))
        return image_embedding + self.residual_scale * gate * residual


class ImageEmbeddingAlignment3D(nn.Module):
    """Residual alignment before the SAM-Med3D mask decoder.

    ProFound embeddings have the right shape for SAM-Med3D, but not
    necessarily the same feature distribution as SAM-Med3D's own encoder.
    This zero-initialized residual path lets Stage-2 learn that mapping while
    preserving the pretrained decoder behavior at initialization.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int | None = None,
        residual_init: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or embed_dim)
        self.norm = nn.GroupNorm(1, embed_dim)
        self.proj = nn.Sequential(
            nn.Conv3d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, embed_dim, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.proj(self.norm(x))


class SequenceAdapter(nn.Module):
    """Small bottleneck adapter for SAM decoder token/image sequences."""

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 64,
        residual_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * self.up(self.act(self.down(self.norm(x))))


class PCaSAM3DProFoundModel(nn.Module):
    """PCaSAM-3D-ProFound: ProFound encoder + SAM-Med3D decoder with auto prompts.

    This model combines:
    - ProFound-Conv as a domain-specific image encoder for prostate mpMRI
    - A lightweight coarse branch for automatic prompt generation (PCaSAM-style)
    - SAM-Med3D's prompt encoder + mask decoder for prompt-conditioned refinement
    - A feature bridge that projects ProFound features to SAM's embedding space
    """

    def __init__(
        self,
        # ProFound encoder config
        profound_checkpoint_path: str | None = None,
        profound_repo_path: str | None = None,
        profound_model_import_path: str | None = None,
        profound_model_kwargs: dict | None = None,
        profound_checkpoint_format: str = "auto",
        freeze_encoder: bool = True,
        unfreeze_last_n_stages: int | None = None,
        encoder_channels: list[int] | None = None,
        # Enhancement
        use_lesion_aware_enhancement: bool = True,
        # Modality fusion
        use_modality_aware_fusion: bool = True,
        modality_fusion_hidden_channels: int = 16,
        modality_fusion_residual_scale: float = 0.1,
        modality_dropout_prob: float = 0.0,
        modality_dropout_keep_at_least: int = 2,
        modality_dropout_probs: list[float] | None = None,
        modality_gate_prior: list[float] | None = None,
        # SAM-Med3D decoder config
        sam_checkpoint_path: str | None = None,
        sam_embed_dim: int = 384,
        sam_image_size: int = 128,
        freeze_sam_decoder: bool = False,
        # Feature bridge config
        bridge_use_stage2: bool = True,
        bridge_norm: str = "layer",
        # Modality cross-attention (PCaSAM-style, after Feature Bridge)
        use_modality_cross_attention: bool = True,
        cross_attn_num_heads: int = 6,
        cross_attn_reduction: int = 4,
        # Decoder alignment / adapter config
        use_decoder_alignment: bool = False,
        decoder_alignment_hidden_dim: int | None = None,
        decoder_alignment_scale: float = 0.1,
        use_mask_decoder_adapters: bool = False,
        mask_decoder_adapter_dim: int = 64,
        mask_decoder_adapter_scale: float = 1e-3,
        use_self_gated_multiscale: bool = False,
        self_gated_multiscale_scale: float = 0.05,
        # Coarse branch config
        coarse_hidden_dim: int = 64,
        use_high_res_refinement: bool = True,
        high_res_refinement_hidden_dim: int = 32,
        high_res_refinement_scale: float = 0.5,
        # Auto prompt config
        coarse_threshold: float = 0.3,
        max_proposals: int = 5,
        min_component_voxels: int = 20,
        point_type: str = "centroid",
        no_prompt_if_empty: bool = True,
        no_prompt_threshold: float = 0.05,
        use_box_prompts: bool = True,
        box_margin_voxels: int = 4,
        soft_box_std_scale: float = 2.0,
        training_point_mode: str = "topk_peaks",
        training_nms_kernel: int = 9,
        train_hard_prompt_prob: float = 0.20,
        training_use_soft_box: bool = False,
        use_objectness_gate: bool = True,
        objectness_threshold: float = 0.20,
        # Prompt dropout / robustness
        prompt_dropout_enabled: bool = False,
        prompt_drop_point_prob: float = 0.0,
        prompt_drop_box_prob: float = 0.0,
        prompt_drop_mask_prob: float = 0.0,
        # Optional zoom-in refinement for inference/evaluation
        zoom_margin_voxels: int = 8,
        zoom_min_size_voxels: int = 24,
        # Training config
        use_mask_prior: bool = True,
        return_dict: bool = True,
    ) -> None:
        super().__init__()
        self.return_dict = return_dict
        self.use_mask_prior = use_mask_prior
        self.use_box_prompts = bool(use_box_prompts)
        self.use_objectness_gate = bool(use_objectness_gate)
        self.objectness_threshold = float(objectness_threshold)
        self.prompt_dropout_enabled = bool(prompt_dropout_enabled)
        self.prompt_drop_point_prob = float(prompt_drop_point_prob)
        self.prompt_drop_box_prob = float(prompt_drop_box_prob)
        self.prompt_drop_mask_prob = float(prompt_drop_mask_prob)
        self.zoom_margin_voxels = int(zoom_margin_voxels)
        self.zoom_min_size_voxels = int(zoom_min_size_voxels)
        self.use_high_res_refinement = bool(use_high_res_refinement)
        self.use_mask_decoder_adapters = bool(use_mask_decoder_adapters)
        encoder_channels = encoder_channels or [96, 192, 384, 768]
        self.encoder_channels = encoder_channels

        # Image embedding spatial size (for 128^3 input with patch_size=16)
        image_embedding_size = (
            sam_image_size // 16,
            sam_image_size // 16,
            sam_image_size // 16,
        )
        self.image_embedding_size = image_embedding_size
        self.sam_image_size = sam_image_size

        # ─── ProFound-Conv Encoder ───
        self.modality_fusion = (
            ModalityAwareFusion3D(
                in_channels=3,
                hidden_channels=modality_fusion_hidden_channels,
                residual_scale=modality_fusion_residual_scale,
                modality_dropout_prob=modality_dropout_prob,
                modality_dropout_keep_at_least=modality_dropout_keep_at_least,
                modality_dropout_probs=modality_dropout_probs,
                modality_gate_prior=modality_gate_prior,
            )
            if use_modality_aware_fusion
            else nn.Identity()
        )

        self.encoder = ProFoundConvEncoderWrapper(
            checkpoint_path=profound_checkpoint_path,
            profound_repo_path=profound_repo_path,
            profound_model_import_path=profound_model_import_path,
            profound_model_kwargs=profound_model_kwargs,
            profound_checkpoint_format=profound_checkpoint_format,
            freeze_encoder=freeze_encoder,
            strict_load=False,
            return_multi_scale_features=True,
        )

        # Optionally unfreeze last N stages
        if unfreeze_last_n_stages is not None and unfreeze_last_n_stages > 0:
            self._unfreeze_last_stages(unfreeze_last_n_stages)

        # ─── Lesion-Aware Enhancement ───
        self.enhancement = (
            LesionAwareEnhancement3D(encoder_channels[-1])
            if use_lesion_aware_enhancement
            else nn.Identity()
        )

        # ─── Feature Bridge: ProFound → SAM embedding space ───
        self.feature_bridge = ProFoundToSAMBridge(
            encoder_channels=encoder_channels,
            embed_dim=sam_embed_dim,
            target_spatial=image_embedding_size,
            use_stage2=bridge_use_stage2,
            norm=bridge_norm,
        )

        # ─── Modality Cross-Attention (PCaSAM-style feature-level fusion) ───
        self.modality_cross_attention = (
            ModalityCrossAttention3D(
                embed_dim=sam_embed_dim,
                num_modalities=3,
                num_heads=cross_attn_num_heads,
                reduction=cross_attn_reduction,
            )
            if use_modality_cross_attention
            else nn.Identity()
        )

        self.self_gated_multiscale = (
            SelfGatedMultiScaleFusion3D(
                encoder_channels=encoder_channels,
                embed_dim=sam_embed_dim,
                residual_init=self_gated_multiscale_scale,
            )
            if use_self_gated_multiscale
            else None
        )

        self.decoder_alignment = (
            ImageEmbeddingAlignment3D(
                embed_dim=sam_embed_dim,
                hidden_dim=decoder_alignment_hidden_dim,
                residual_init=decoder_alignment_scale,
            )
            if use_decoder_alignment
            else nn.Identity()
        )

        # ─── Coarse Branch (for auto prompt generation) ───
        self.coarse_branch = CoarseBranch(
            encoder_channels=encoder_channels,
            hidden_dim=coarse_hidden_dim,
            out_channels=1,
        )
        self.objectness_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(coarse_hidden_dim, coarse_hidden_dim),
            nn.GELU(),
            nn.Linear(coarse_hidden_dim, 1),
        )
        self.high_res_refinement = (
            HighResRefinementHead3D(
                encoder_channels=encoder_channels,
                hidden_dim=high_res_refinement_hidden_dim,
                residual_init=high_res_refinement_scale,
            )
            if use_high_res_refinement
            else None
        )

        # ─── Auto Prompt Generator ───
        self.auto_prompt = AutoPrompt3DFromCoarse(
            coarse_threshold=coarse_threshold,
            max_proposals=max_proposals,
            min_component_voxels=min_component_voxels,
            point_type=point_type,
            embed_dim=sam_embed_dim,
            image_embedding_size=image_embedding_size,
            no_prompt_if_empty=no_prompt_if_empty,
            no_prompt_threshold=no_prompt_threshold,
            box_margin_voxels=box_margin_voxels,
            soft_box_std_scale=soft_box_std_scale,
            training_point_mode=training_point_mode,
            training_nms_kernel=training_nms_kernel,
            train_hard_prompt_prob=train_hard_prompt_prob,
            training_use_soft_box=training_use_soft_box,
        )

        # ─── Prompt Adapter ───
        self.prompt_adapter = PromptAdapter(
            input_image_size=(sam_image_size, sam_image_size, sam_image_size),
            image_embedding_size=image_embedding_size,
        )

        # ─── SAM-Med3D Prompt Encoder + Mask Decoder ───
        self._build_sam_decoder(sam_checkpoint_path, sam_embed_dim, image_embedding_size, sam_image_size)
        self._patch_prompt_encoder_3d_boxes()
        if self.use_mask_decoder_adapters:
            self._install_mask_decoder_adapters(
                bottleneck_dim=mask_decoder_adapter_dim,
                residual_init=mask_decoder_adapter_scale,
            )

        if freeze_sam_decoder:
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False
            for p in self.mask_decoder.parameters():
                p.requires_grad = False

    def _install_mask_decoder_adapters(
        self,
        bottleneck_dim: int = 64,
        residual_init: float = 1e-3,
    ) -> None:
        """Attach zero-initialized adapters to each SAM TwoWay block.

        This keeps the vendored SAM-Med3D code unchanged. The original decoder
        path is exactly preserved at initialization, then only the small adapter
        residuals need to learn the ProFound-to-SAM domain shift.
        """
        transformer = getattr(self.mask_decoder, "transformer", None)
        layers = getattr(transformer, "layers", None)
        if layers is None:
            raise RuntimeError("Mask decoder transformer layers not found; cannot install adapters.")

        for block in layers:
            embed_dim = int(block.norm1.normalized_shape[0])
            block.query_self_adapter = SequenceAdapter(embed_dim, bottleneck_dim, residual_init)
            block.query_cross_adapter = SequenceAdapter(embed_dim, bottleneck_dim, residual_init)
            block.query_mlp_adapter = SequenceAdapter(embed_dim, bottleneck_dim, residual_init)
            block.key_cross_adapter = SequenceAdapter(embed_dim, bottleneck_dim, residual_init)

            def _forward_with_adapters(block_self, queries, keys, query_pe, key_pe):
                if block_self.skip_first_layer_pe:
                    queries = block_self.self_attn(q=queries, k=queries, v=queries)
                else:
                    q = queries + query_pe
                    attn_out = block_self.self_attn(q=q, k=q, v=queries)
                    queries = queries + attn_out
                queries = block_self.norm1(queries)
                queries = queries + block_self.query_self_adapter(queries)

                q = queries + query_pe
                k = keys + key_pe
                attn_out = block_self.cross_attn_token_to_image(q=q, k=k, v=keys)
                queries = queries + attn_out
                queries = block_self.norm2(queries)
                queries = queries + block_self.query_cross_adapter(queries)

                mlp_out = block_self.mlp(queries)
                queries = queries + mlp_out
                queries = block_self.norm3(queries)
                queries = queries + block_self.query_mlp_adapter(queries)

                q = queries + query_pe
                k = keys + key_pe
                attn_out = block_self.cross_attn_image_to_token(q=k, k=q, v=queries)
                keys = keys + attn_out
                keys = block_self.norm4(keys)
                keys = keys + block_self.key_cross_adapter(keys)
                return queries, keys

            block.forward = types.MethodType(_forward_with_adapters, block)

    def _build_sam_decoder(
        self,
        sam_checkpoint_path: str | None,
        embed_dim: int,
        image_embedding_size: tuple[int, int, int],
        image_size: int,
    ) -> None:
        """Build SAM-Med3D's prompt encoder and mask decoder from pretrained weights."""
        _ensure_sam_med3d_on_path()
        from segment_anything.modeling.prompt_encoder3D import PromptEncoder3D
        from segment_anything.modeling.mask_decoder3D import MaskDecoder3D

        self.prompt_encoder = PromptEncoder3D(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # Load pretrained SAM-Med3D decoder weights if available
        if sam_checkpoint_path is not None:
            self._load_sam_decoder_weights(sam_checkpoint_path)

    def _patch_prompt_encoder_3d_boxes(self) -> None:
        """Make SAM-Med3D prompt encoder accept one 3D box per batch item.

        The vendored PromptEncoder3D has a 2D-style box embed path in some
        checkpoints/repos. Keep the external dependency untouched and patch the
        instance locally to embed two 3D corners: [z0, y0, x0] and [z1, y1, x1].
        """
        while len(self.prompt_encoder.point_embeddings) < 4:
            self.prompt_encoder.point_embeddings.append(nn.Embedding(1, self.prompt_encoder.embed_dim))
        self.prompt_encoder.num_point_embeddings = max(self.prompt_encoder.num_point_embeddings, 4)

        def _embed_boxes_3d(prompt_encoder, boxes: torch.Tensor) -> torch.Tensor:
            boxes = boxes + 0.5
            if boxes.ndim == 3 and boxes.shape[1:] == (2, 3):
                coords = boxes
            elif boxes.ndim == 2 and boxes.shape[1] == 6:
                coords = boxes.reshape(-1, 2, 3)
            else:
                raise ValueError(f"Expected boxes as [B,2,3] or [B,6], got {tuple(boxes.shape)}")
            corner_embedding = prompt_encoder.pe_layer.forward_with_coords(coords, prompt_encoder.input_image_size)
            corner_embedding[:, 0, :] += prompt_encoder.point_embeddings[2].weight
            corner_embedding[:, 1, :] += prompt_encoder.point_embeddings[3].weight
            return corner_embedding

        self.prompt_encoder._embed_boxes = types.MethodType(_embed_boxes_3d, self.prompt_encoder)

    def _load_sam_decoder_weights(self, checkpoint_path: str) -> None:
        """Load prompt_encoder and mask_decoder weights from SAM-Med3D checkpoint."""
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state

        # Extract prompt_encoder and mask_decoder weights
        pe_state = {}
        md_state = {}
        for k, v in state_dict.items():
            if k.startswith("prompt_encoder."):
                pe_state[k.replace("prompt_encoder.", "")] = v
            elif k.startswith("mask_decoder."):
                md_state[k.replace("mask_decoder.", "")] = v

        if pe_state:
            missing, unexpected = self.prompt_encoder.load_state_dict(pe_state, strict=False)
            if missing:
                print(f"[PCaSAM3D] prompt_encoder missing keys: {missing[:5]}...")
        if md_state:
            missing, unexpected = self.mask_decoder.load_state_dict(md_state, strict=False)
            if missing:
                print(f"[PCaSAM3D] mask_decoder missing keys: {missing[:5]}...")

        n_loaded = len(pe_state) + len(md_state)
        print(f"[PCaSAM3D] Loaded {n_loaded} SAM-Med3D decoder tensors from {checkpoint_path}")

    def _unfreeze_last_stages(self, n: int) -> None:
        """Unfreeze the last N stages of the ProFound encoder."""
        # ConvNeXtV2 stages are typically named stages.0, stages.1, stages.2, stages.3
        model = self.encoder.model
        stage_names = []
        for name, _ in model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "stages":
                stage_idx = int(parts[1])
                if stage_idx not in stage_names:
                    stage_names.append(stage_idx)

        stages_to_unfreeze = sorted(stage_names)[-n:]
        for name, param in model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] == "stages":
                if int(parts[1]) in stages_to_unfreeze:
                    param.requires_grad = True


    def _apply_prompt_dropout(
        self,
        point_labels: torch.Tensor,
        box_valid: Optional[torch.Tensor],
        mask_prior: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Randomly drop prompt types during training for prompt robustness."""
        if not (self.training and self.prompt_dropout_enabled):
            return point_labels, box_valid, mask_prior

        device = point_labels.device
        dropped_point = False
        dropped_box = False
        dropped_mask = False
        if self.prompt_drop_point_prob > 0 and torch.rand((), device=device).item() < self.prompt_drop_point_prob:
            point_labels = -torch.ones_like(point_labels)
            dropped_point = True
        if box_valid is not None and self.prompt_drop_box_prob > 0 and torch.rand((), device=device).item() < self.prompt_drop_box_prob:
            box_valid = torch.zeros_like(box_valid)
            dropped_box = True
        if mask_prior is not None and self.prompt_drop_mask_prob > 0 and torch.rand((), device=device).item() < self.prompt_drop_mask_prob:
            mask_prior = None
            dropped_mask = True

        if dropped_point and (box_valid is None or dropped_box) and dropped_mask:
            point_labels = torch.ones_like(point_labels)
        return point_labels, box_valid, mask_prior


    def forward_coarse(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward only the PCaSAM-style prompt-free coarse branch.

        This is used for stage-1 coarse/prompt-source pretraining. It avoids
        routing through the SAM decoder so the proposal generator can be trained
        and selected on lesion recall before prompt-guided refinement is learned.
        """
        input_shape = image.shape[2:]
        fused_image = self.modality_fusion(image)
        modality_weights = getattr(self.modality_fusion, "last_modality_weights", None)
        features = self.encoder(fused_image)
        deep_key = "stage4" if "stage4" in features else sorted(features.keys())[-1]
        features[deep_key] = self.enhancement(features[deep_key])
        coarse_out = self.coarse_branch(features, input_shape, return_features=True)
        coarse_logits = coarse_out["logits"]
        objectness_logit = self.objectness_head(coarse_out["proposal_feature"])
        return {
            "coarse_logits": coarse_logits,
            "coarse_aux_logits": coarse_out["aux_logits"],
            "coarse_prob": torch.sigmoid(coarse_logits),
            "objectness_logit": objectness_logit,
            "objectness_prob": torch.sigmoid(objectness_logit),
            "modality_weights": modality_weights,
        }


    def _box_to_slices(
        self,
        box: torch.Tensor,
        input_shape: tuple[int, int, int],
    ) -> tuple[slice, slice, slice]:
        """Convert a normalized [2,3] z/y/x box to padded crop slices."""
        D, H, W = input_shape
        scale = torch.tensor([D - 1, H - 1, W - 1], device=box.device, dtype=box.dtype).clamp_min(1)
        lo = torch.minimum(box[0], box[1]) * scale
        hi = torch.maximum(box[0], box[1]) * scale
        margin = float(self.zoom_margin_voxels)
        lo = torch.floor(lo - margin)
        hi = torch.ceil(hi + margin)
        min_size = torch.tensor(
            [self.zoom_min_size_voxels, self.zoom_min_size_voxels, self.zoom_min_size_voxels],
            device=box.device,
            dtype=box.dtype,
        )
        center = (lo + hi) * 0.5
        size = torch.maximum(hi - lo + 1.0, min_size)
        lo = torch.floor(center - size * 0.5)
        hi = torch.ceil(center + size * 0.5)
        max_idx = torch.tensor([D - 1, H - 1, W - 1], device=box.device, dtype=box.dtype)
        lo = torch.maximum(lo, torch.zeros_like(lo))
        hi = torch.minimum(hi, max_idx)
        z0, y0, x0 = [int(v) for v in lo.detach().cpu().tolist()]
        z1, y1, x1 = [int(v) + 1 for v in hi.detach().cpu().tolist()]
        return slice(z0, max(z1, z0 + 1)), slice(y0, max(y1, y0 + 1)), slice(x0, max(x1, x0 + 1))

    def forward_zoom_in(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run global auto-prompting, refine the top box crop, and paste it back.

        This is an inference/evaluation helper. It preserves the normal full-volume
        output when no valid box survives objectness/prompt filtering.
        """
        base_out = self.forward(image)
        box_coords = base_out.get("box_coords")
        box_valid = base_out.get("box_valid")
        if box_coords is None or box_valid is None or not bool(box_valid.any().item()):
            base_out["zoom_refined_logits"] = base_out["refined_logits"]
            base_out["zoom_used"] = torch.zeros(image.shape[0], dtype=torch.bool, device=image.device)
            return base_out

        B, _, D, H, W = image.shape
        merged_prob = torch.sigmoid(base_out["refined_logits"]).clone()
        zoom_used = torch.zeros(B, dtype=torch.bool, device=image.device)

        for b in range(B):
            if not bool(box_valid[b].item()):
                continue
            zs, ys, xs = self._box_to_slices(box_coords[b], (D, H, W))
            crop = image[b:b + 1, :, zs, ys, xs]
            crop = F.interpolate(crop, size=(D, H, W), mode="trilinear", align_corners=False)
            crop_out = self.forward(crop)
            crop_prob = torch.sigmoid(crop_out["refined_logits"])
            crop_prob = F.interpolate(
                crop_prob,
                size=(zs.stop - zs.start, ys.stop - ys.start, xs.stop - xs.start),
                mode="trilinear",
                align_corners=False,
            )
            region = merged_prob[b:b + 1, :, zs, ys, xs]
            merged_prob[b:b + 1, :, zs, ys, xs] = torch.maximum(region, crop_prob)
            zoom_used[b] = True

        eps = torch.finfo(merged_prob.dtype).eps
        merged_prob = merged_prob.clamp(eps, 1.0 - eps)
        base_out["zoom_refined_logits"] = torch.logit(merged_prob)
        base_out["zoom_used"] = zoom_used
        return base_out

    def forward(
        self,
        image: torch.Tensor,
        external_point_coords: Optional[torch.Tensor] = None,
        external_point_labels: Optional[torch.Tensor] = None,
        external_box_coords: Optional[torch.Tensor] = None,
        external_box_valid: Optional[torch.Tensor] = None,
        external_mask_prior: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """End-to-end forward pass.

        Args:
            image: [B, 3, D, H, W] normalized mpMRI input
            external_point_coords: [B, N, 3] optional external prompts (override auto)
            external_point_labels: [B, N] optional external prompt labels
            external_box_coords: [B, 2, 3] optional normalized box corners
            external_box_valid: [B] optional bool mask for valid boxes
            external_mask_prior: [B, 1, D, H, W] optional dense prompt prior

        Returns:
            dict with:
                refined_logits: [B, 1, D, H, W] final refined segmentation logits
                coarse_logits:  [B, 1, D, H, W] coarse branch logits (for aux loss)
                iou_pred:       [B, 1] mask quality prediction
                coarse_prob:    [B, 1, D, H, W] coarse probability map
                point_coords:   [B, N, 3] generated point prompts
                point_labels:   [B, N] prompt labels
                box_coords:     [B, 2, 3] generated box prompt corners
                box_valid:      [B] valid box mask
        """
        input_shape = image.shape[2:]  # (D, H, W)

        # ─── 1. Modality-Aware Fusion ───
        fused_image = self.modality_fusion(image)
        modality_weights = getattr(self.modality_fusion, "last_modality_weights", None)

        # ─── 2. ProFound-Conv Encoder ───
        features = self.encoder(fused_image)

        # Apply lesion-aware enhancement to deepest features
        deep_key = "stage4" if "stage4" in features else sorted(features.keys())[-1]
        features[deep_key] = self.enhancement(features[deep_key])

        # ─── 2. Coarse Branch → coarse logits ───
        coarse_out = self.coarse_branch(features, input_shape, return_features=True)
        coarse_logits = coarse_out["logits"]
        objectness_logit = self.objectness_head(coarse_out["proposal_feature"])
        objectness_prob = torch.sigmoid(objectness_logit)

        # ─── 3. Auto Prompt Generation ───
        if external_point_coords is not None and external_point_labels is not None:
            # Use externally provided prompts (e.g., GT-jitter curriculum training)
            coarse_prob = torch.sigmoid(coarse_logits)
            prior_source = external_mask_prior.float() if external_mask_prior is not None else coarse_prob
            mask_prior = F.interpolate(
                prior_source, size=self.image_embedding_size,
                mode="trilinear", align_corners=False,
            )
            point_coords = external_point_coords
            point_labels = external_point_labels
            box_coords = external_box_coords
            box_valid = external_box_valid
            prompt_mode = "external"
        else:
            prompt_out = self.auto_prompt(coarse_logits, input_shape)
            point_coords = prompt_out["point_coords"]
            point_labels = prompt_out["point_labels"]
            box_coords = prompt_out.get("box_coords")
            box_valid = prompt_out.get("box_valid")
            mask_prior = prompt_out["mask_prior"]
            coarse_prob = prompt_out["coarse_prob"]
            prompt_mode = prompt_out.get("prompt_mode", "auto")
            if self.use_objectness_gate and not self.training:
                keep_prompt = (objectness_prob.view(-1) >= self.objectness_threshold)
                point_labels = torch.where(
                    keep_prompt.view(-1, 1),
                    point_labels,
                    -torch.ones_like(point_labels),
                )
                if box_valid is not None:
                    box_valid = box_valid & keep_prompt

        point_labels, box_valid, mask_prior = self._apply_prompt_dropout(
            point_labels, box_valid, mask_prior
        )

        # ─── 4. Feature Bridge → SAM embedding ───
        image_embedding = self.feature_bridge(features)  # [B, 384, 8, 8, 8]

        # ─── 4.5. Modality Cross-Attention (PCaSAM-style feature fusion) ───
        image_embedding = self.modality_cross_attention(image_embedding)

        # ─── 4.6. Self-gated multi-scale context for small lesions ───
        if self.self_gated_multiscale is not None:
            image_embedding = self.self_gated_multiscale(image_embedding, features)

        # ─── 4.7. ProFound-to-SAM embedding alignment ───
        image_embedding = self.decoder_alignment(image_embedding)

        # ─── 5. Prompt Adapter → SAM format ───
        points, boxes, masks = self.prompt_adapter(
            point_coords,
            point_labels,
            mask_prior=mask_prior if self.use_mask_prior else None,
            box_coords=box_coords if self.use_box_prompts else None,
            box_valid=box_valid,
        )

        # ─── 6. SAM Prompt Encoder ───
        sparse_emb, dense_emb = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        if dense_emb.shape[2:] != image_embedding.shape[2:]:
            raise RuntimeError(
                f"Dense prompt shape {tuple(dense_emb.shape)} does not match image embedding "
                f"shape {tuple(image_embedding.shape)}. Check PromptAdapter mask_input_size."
            )

        # ─── 7. SAM Mask Decoder ───
        masks_logits, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        # ─── 8. Upsample to input resolution, then recover high-res detail ───
        sam_upsampled_logits = F.interpolate(
            masks_logits, size=input_shape, mode="trilinear", align_corners=False
        )
        high_res_residual = None
        if self.high_res_refinement is not None:
            refined_logits, high_res_residual = self.high_res_refinement(
                sam_upsampled_logits, coarse_logits, features, input_shape
            )
        else:
            refined_logits = sam_upsampled_logits

        if self.return_dict:
            return {
                "refined_logits": refined_logits,
                "sam_upsampled_logits": sam_upsampled_logits,
                "high_res_residual_logits": high_res_residual,
                "coarse_logits": coarse_logits,
                "coarse_aux_logits": coarse_out["aux_logits"],
                "iou_pred": iou_pred,
                "coarse_prob": coarse_prob,
                "objectness_logit": objectness_logit,
                "objectness_prob": objectness_prob,
                "modality_weights": modality_weights,
                "point_coords": point_coords,
                "point_labels": point_labels,
                "box_coords": box_coords,
                "box_valid": box_valid,
                "prompt_mode": prompt_mode,
            }
        return refined_logits


def build_pcasam3d_profound(cfg: dict) -> PCaSAM3DProFoundModel:
    """Build PCaSAM3DProFoundModel from a config dict."""
    model_cfg = cfg.get("model", cfg)
    return PCaSAM3DProFoundModel(
        # ProFound encoder
        profound_checkpoint_path=model_cfg.get("profound_checkpoint_path"),
        profound_repo_path=model_cfg.get("profound_repo_path"),
        profound_model_import_path=model_cfg.get("profound_model_import_path"),
        profound_model_kwargs=model_cfg.get("profound_model_kwargs"),
        profound_checkpoint_format=model_cfg.get("profound_checkpoint_format", "auto"),
        freeze_encoder=model_cfg.get("freeze_encoder", True),
        unfreeze_last_n_stages=model_cfg.get("unfreeze_last_n_stages"),
        encoder_channels=model_cfg.get("encoder_channels", [96, 192, 384, 768]),
        # Enhancement
        use_lesion_aware_enhancement=model_cfg.get("use_lesion_aware_enhancement", True),
        # Modality fusion
        use_modality_aware_fusion=model_cfg.get("use_modality_aware_fusion", True),
        modality_fusion_hidden_channels=model_cfg.get("modality_fusion_hidden_channels", 16),
        modality_fusion_residual_scale=model_cfg.get("modality_fusion_residual_scale", 0.1),
        modality_dropout_prob=model_cfg.get("modality_dropout", {}).get("prob", 0.0),
        modality_dropout_keep_at_least=model_cfg.get("modality_dropout", {}).get("keep_at_least", 2),
        modality_dropout_probs=model_cfg.get("modality_dropout", {}).get("probs"),
        modality_gate_prior=model_cfg.get("modality_gate_prior"),
        # SAM decoder
        sam_checkpoint_path=model_cfg.get("sam_checkpoint_path"),
        sam_embed_dim=model_cfg.get("sam_embed_dim", 384),
        sam_image_size=model_cfg.get("sam_image_size", 128),
        freeze_sam_decoder=model_cfg.get("freeze_sam_decoder", False),
        # Feature bridge
        bridge_use_stage2=model_cfg.get("bridge_use_stage2", True),
        bridge_norm=model_cfg.get("bridge_norm", "layer"),
        # Modality cross-attention
        use_modality_cross_attention=model_cfg.get("use_modality_cross_attention", True),
        cross_attn_num_heads=model_cfg.get("cross_attn_num_heads", 6),
        cross_attn_reduction=model_cfg.get("cross_attn_reduction", 4),
        use_decoder_alignment=model_cfg.get("use_decoder_alignment", False),
        decoder_alignment_hidden_dim=model_cfg.get("decoder_alignment_hidden_dim"),
        decoder_alignment_scale=model_cfg.get("decoder_alignment_scale", 0.1),
        use_mask_decoder_adapters=model_cfg.get("use_mask_decoder_adapters", False),
        mask_decoder_adapter_dim=model_cfg.get("mask_decoder_adapter_dim", 64),
        mask_decoder_adapter_scale=model_cfg.get("mask_decoder_adapter_scale", 1e-3),
        use_self_gated_multiscale=model_cfg.get("use_self_gated_multiscale", False),
        self_gated_multiscale_scale=model_cfg.get("self_gated_multiscale_scale", 0.05),
        # Coarse branch
        coarse_hidden_dim=model_cfg.get("coarse_hidden_dim", 64),
        use_high_res_refinement=model_cfg.get("use_high_res_refinement", True),
        high_res_refinement_hidden_dim=model_cfg.get("high_res_refinement_hidden_dim", 32),
        high_res_refinement_scale=model_cfg.get("high_res_refinement_scale", 0.5),
        # Auto prompt
        coarse_threshold=model_cfg.get("coarse_threshold", 0.3),
        max_proposals=model_cfg.get("max_proposals", 5),
        min_component_voxels=model_cfg.get("min_component_voxels", 20),
        point_type=model_cfg.get("point_type", "centroid"),
        no_prompt_if_empty=model_cfg.get("no_prompt_if_empty", True),
        no_prompt_threshold=model_cfg.get("no_prompt_threshold", 0.05),
        use_box_prompts=model_cfg.get("use_box_prompts", True),
        box_margin_voxels=model_cfg.get("box_margin_voxels", 4),
        soft_box_std_scale=model_cfg.get("soft_box_std_scale", 2.0),
        training_point_mode=model_cfg.get("training_point_mode", "topk_peaks"),
        training_nms_kernel=model_cfg.get("training_nms_kernel", 9),
        train_hard_prompt_prob=model_cfg.get("train_hard_prompt_prob", 0.20),
        training_use_soft_box=model_cfg.get("training_use_soft_box", False),
        use_objectness_gate=model_cfg.get("use_objectness_gate", True),
        objectness_threshold=model_cfg.get("objectness_threshold", 0.20),
        prompt_dropout_enabled=model_cfg.get("prompt_dropout", {}).get("enabled", False),
        prompt_drop_point_prob=model_cfg.get("prompt_dropout", {}).get("drop_point_prob", 0.0),
        prompt_drop_box_prob=model_cfg.get("prompt_dropout", {}).get("drop_box_prob", 0.0),
        prompt_drop_mask_prob=model_cfg.get("prompt_dropout", {}).get("drop_mask_prob", 0.0),
        zoom_margin_voxels=model_cfg.get("zoom_margin_voxels", 8),
        zoom_min_size_voxels=model_cfg.get("zoom_min_size_voxels", 24),
        # Training
        use_mask_prior=model_cfg.get("use_mask_prior", True),
        return_dict=model_cfg.get("return_dict", True),
    )
