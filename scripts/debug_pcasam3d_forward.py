#!/usr/bin/env python
"""Debug forward pass for PCaSAM-3D-ProFound model.

Tests model construction and forward pass with synthetic data.
Does NOT require ProFound checkpoint (uses a mock encoder for shape testing).

Usage:
    /root/anaconda3/envs/lm/bin/python scripts/debug_pcasam3d_forward.py --mode shapes
    /root/anaconda3/envs/lm/bin/python scripts/debug_pcasam3d_forward.py --mode full --config configs/train_pcasam3d_profound.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn


class MockProFoundEncoder(nn.Module):
    """Mock encoder that produces multi-scale features matching ProFound-Conv."""

    def __init__(self, in_chans=3, channels=(96, 192, 384, 768)):
        super().__init__()
        self.channels = channels
        self.stem = nn.Conv3d(in_chans, channels[0], 4, stride=4, padding=0)
        self.stage2_down = nn.Conv3d(channels[0], channels[1], 2, stride=2)
        self.stage3_down = nn.Conv3d(channels[1], channels[2], 2, stride=2)
        self.stage4_down = nn.Conv3d(channels[2], channels[3], 2, stride=2)

    def forward(self, x, ret_hids=False):
        s1 = self.stem(x)
        s2 = self.stage2_down(s1)
        s3 = self.stage3_down(s2)
        s4 = self.stage4_down(s3)
        if ret_hids:
            return x, [s1, s2, s3, s4]
        return s4


def test_shapes():
    """Test model component shapes with mock encoder."""
    print("=" * 80)
    print("PCaSAM-3D-ProFound Shape Test (Mock Encoder)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, D, H, W = 2, 3, 128, 128, 128

    # Build components individually
    from src.models.pcasam3d_profound.feature_bridge import ProFoundToSAMBridge
    from src.models.pcasam3d_profound.coarse_branch import CoarseBranch
    from src.models.pcasam3d_profound.auto_prompt_3d import AutoPrompt3DFromCoarse
    from src.models.pcasam3d_profound.prompt_adapter import PromptAdapter

    encoder = MockProFoundEncoder().to(device)
    bridge = ProFoundToSAMBridge(encoder_channels=[96, 192, 384, 768], embed_dim=384, target_spatial=(8, 8, 8)).to(device)
    coarse = CoarseBranch(encoder_channels=[96, 192, 384, 768], hidden_dim=64).to(device)
    auto_prompt = AutoPrompt3DFromCoarse(embed_dim=384, image_embedding_size=(8, 8, 8)).to(device)
    prompt_adapter = PromptAdapter(input_image_size=(128, 128, 128), image_embedding_size=(8, 8, 8)).to(device)

    # Forward
    x = torch.randn(B, C, D, H, W, device=device)
    print(f"\nInput: {x.shape}")

    _, features_list = encoder(x, ret_hids=True)
    features = {f"stage{i+1}": f for i, f in enumerate(features_list)}
    for k, v in features.items():
        print(f"  {k}: {v.shape}")

    # Feature bridge
    image_embedding = bridge(features)
    print(f"\nFeature Bridge output: {image_embedding.shape}")
    assert image_embedding.shape == (B, 384, 8, 8, 8), f"Expected (B, 384, 8, 8, 8), got {image_embedding.shape}"

    # Coarse branch
    coarse_logits = coarse(features, input_shape=(D, H, W))
    print(f"Coarse logits: {coarse_logits.shape}")
    assert coarse_logits.shape == (B, 1, D, H, W), f"Expected (B, 1, D, H, W), got {coarse_logits.shape}"

    # Auto prompt
    prompt_out = auto_prompt(coarse_logits, input_shape=(D, H, W))
    print(f"Point coords: {prompt_out['point_coords'].shape}")
    print(f"Point labels: {prompt_out['point_labels'].shape}")
    print(f"Mask prior: {prompt_out['mask_prior'].shape}")
    print(f"Box coords: {prompt_out['box_coords'].shape}")
    print(f"Box valid: {prompt_out['box_valid'].shape}")

    # Prompt adapter
    points, boxes, masks = prompt_adapter(
        prompt_out["point_coords"],
        prompt_out["point_labels"],
        mask_prior=prompt_out["mask_prior"],
        box_coords=prompt_out["box_coords"],
        box_valid=prompt_out["box_valid"],
    )
    print(f"\nPrompt Adapter:")
    print(f"  Points coords (abs): {points[0].shape}")
    print(f"  Points labels: {points[1].shape}")
    if boxes is not None:
        print(f"  Boxes: {boxes.shape}")
    if masks is not None:
        print(f"  Mask prior: {masks.shape}")
    # SAM decoder (if available)
    try:
        sam_repo = Path(__file__).resolve().parents[2] / "SAM-Med3D"
        if sam_repo.is_dir() and str(sam_repo) not in sys.path:
            sys.path.insert(0, str(sam_repo))
        from segment_anything.modeling.prompt_encoder3D import PromptEncoder3D
        from segment_anything.modeling.mask_decoder3D import MaskDecoder3D

        prompt_encoder = PromptEncoder3D(
            embed_dim=384, image_embedding_size=(8, 8, 8),
            input_image_size=(128, 128, 128), mask_in_chans=16,
        ).to(device)
        mask_decoder = MaskDecoder3D(
            num_multimask_outputs=3, transformer_dim=384,
            iou_head_depth=3, iou_head_hidden_dim=256,
        ).to(device)

        sparse_emb, dense_emb = prompt_encoder(points=points, boxes=None, masks=masks)
        print(f"\nSAM Prompt Encoder:")
        print(f"  Sparse embeddings: {sparse_emb.shape}")
        print(f"  Dense embeddings: {dense_emb.shape}")

        masks_logits, iou_pred = mask_decoder(
            image_embeddings=image_embedding,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        print(f"\nSAM Mask Decoder:")
        print(f"  Masks logits: {masks_logits.shape}")
        print(f"  IoU pred: {iou_pred.shape}")

        # Upsample
        refined = torch.nn.functional.interpolate(masks_logits, size=(D, H, W), mode="trilinear", align_corners=False)
        print(f"  Refined (upsampled): {refined.shape}")

    except ImportError as e:
        print(f"\n[SKIP] SAM-Med3D not available: {e}")

    print("\n" + "=" * 80)
    print("All shape tests PASSED!")
    print("=" * 80)


def test_full_model(config_path: str):
    """Test full model with real ProFound encoder (requires checkpoint)."""
    print("=" * 80)
    print("PCaSAM-3D-ProFound Full Model Test")
    print("=" * 80)

    from src.models.pcasam3d_profound import build_pcasam3d_profound
    from src.models.pcasam3d_profound.pcasam3d_loss import build_pcasam3d_loss
    from src.utils.config_utils import load_config

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_pcasam3d_profound(cfg).to(device)
    loss_fn = build_pcasam3d_loss(cfg)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: {n_total:.1f}M total, {n_train:.1f}M trainable")

    # Forward pass
    B = 1
    patch_size = tuple(cfg["data"].get("patch_size", [128, 128, 128]))
    x = torch.randn(B, 3, *patch_size, device=device)
    label = (torch.rand(B, 1, *patch_size, device=device) > 0.9).float()

    print(f"\nInput: {x.shape}")
    print(f"Label: {label.shape}")

    model.train()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(x)
        losses = loss_fn(output, label)

    print(f"\nOutputs:")
    for k, v in output.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")

    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Backward
    losses["total_loss"].backward()
    print("\nBackward pass: OK")

    # Check gradients
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name.split(".")[0]] = grad_norms.get(name.split(".")[0], 0) + p.grad.norm().item()
    print("\nGradient norms by module:")
    for module, norm in sorted(grad_norms.items()):
        print(f"  {module}: {norm:.4f}")

    print("\n" + "=" * 80)
    print("Full model test PASSED!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser("PCaSAM-3D-ProFound Debug")
    parser.add_argument("--mode", choices=["shapes", "full"], default="shapes")
    parser.add_argument("--config", default="configs/train_pcasam3d_profound.yaml")
    args = parser.parse_args()

    if args.mode == "shapes":
        test_shapes()
    elif args.mode == "full":
        test_full_model(args.config)


if __name__ == "__main__":
    main()
