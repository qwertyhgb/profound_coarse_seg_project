#!/usr/bin/env python
"""
Phase 0: SAM-Med3D Zero-Shot Evaluation on PI-CAI fold-0 val set.

Uses GT bounding boxes as prompts (upper-bound for prompt quality).
This is the Go/No-Go check before investing in encoder replacement.

Decision criteria:
  - GT-prompt zero-shot Dice >= 0.40 → proceed to Phase 1
  - GT-prompt zero-shot Dice  < 0.20 → SAM-Med3D not suitable, pivot

Usage:
    /root/anaconda3/envs/lm/bin/python scripts/phase0_sam_med3d_zeroshot.py \
        --sam-checkpoint /path/to/sam_med3d_turbo.pth \
        --val-split data/splits/5fold/fold_0/val.txt \
        --processed-root ../picai_preprocessing_project/data/processed/picai_profound_prompt_v2 \
        --output-dir outputs/phase0_zeroshot \
        --roi-size 128 128 128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM_ROOT = ROOT.parent / "SAM-Med3D"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_case(npz_path: Path):
    """Load image [3,D,H,W] and label [1,D,H,W] from npz."""
    with np.load(npz_path, allow_pickle=False) as d:
        image = d["image"].astype(np.float32)   # [3, D, H, W]
        label = d["label"].astype(np.float32)   # [1, D, H, W]
        case_id = str(d["case_id"])
    return image, label, case_id


def normalize_channel(vol: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance normalization on non-zero voxels."""
    mask = vol != 0
    if mask.sum() == 0:
        return vol
    mean = vol[mask].mean()
    std = vol[mask].std()
    if std < 1e-6:
        return vol - mean
    return (vol - mean) / std


def prepare_single_channel(image: np.ndarray, channel: int = 0) -> np.ndarray:
    """Extract one channel and normalize. Returns [1, D, H, W]."""
    ch = image[channel]
    ch = normalize_channel(ch)
    return ch[None]  # [1, D, H, W]


def prepare_three_channel_avg(image: np.ndarray) -> np.ndarray:
    """Average three modalities into one channel. Returns [1, D, H, W]."""
    channels = [normalize_channel(image[i]) for i in range(image.shape[0])]
    avg = np.mean(channels, axis=0)
    return avg[None]  # [1, D, H, W]


def resize_volume(vol: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize [C, D, H, W] to target_size using trilinear interpolation."""
    t = torch.from_numpy(vol).float().unsqueeze(0)  # [1, C, D, H, W]
    t = F.interpolate(t, size=target_size, mode="trilinear", align_corners=False)
    return t.squeeze(0).numpy()  # [C, D, H, W]


def get_gt_bbox_3d(label: np.ndarray, margin: tuple = (2, 4, 4)) -> list[int] | None:
    """
    Get 3D bounding box [z0, z1, y0, y1, x0, x1] from binary label.
    Returns None if label is empty.
    """
    mask = label[0] > 0.5
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    D, H, W = label.shape[1:]
    mz, my, mx = margin
    return [
        max(0, int(z0) - mz), min(D, int(z1) + mz),
        max(0, int(y0) - my), min(H, int(y1) + my),
        max(0, int(x0) - mx), min(W, int(x1) + mx),
    ]


def get_gt_lesion_bboxes(label: np.ndarray, margin: tuple = (2, 4, 4)) -> list[list[int]]:
    """Get per-connected-component bounding boxes from label."""
    mask = label[0] > 0.5
    if not mask.any():
        return []
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    D, H, W = label.shape[1:]
    mz, my, mx = margin
    bboxes = []
    for comp_id in range(1, n + 1):
        coords = np.argwhere(labeled == comp_id)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        bboxes.append([
            max(0, int(z0) - mz), min(D, int(z1) + mz),
            max(0, int(y0) - my), min(H, int(y1) + my),
            max(0, int(x0) - mx), min(W, int(x1) + mx),
        ])
    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# SAM-Med3D inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_to_sam_box(bbox_zyxzyx: list[int], roi_size: tuple) -> torch.Tensor:
    """
    Convert [z0, z1, y0, y1, x0, x1] bbox to SAM-Med3D box format.
    SAM-Med3D PromptEncoder3D._embed_boxes expects shape [B, 1, 6] with
    coords normalized to input_image_size.
    Actually it expects [B, N, 2, 3] corner format... let's check.

    Looking at prompt_encoder3D.py:
        def _embed_boxes(self, boxes):
            boxes = boxes + 0.5
            coords = boxes.reshape(-1, 2, 2)   # ← 2D format!

    SAM-Med3D's PromptEncoder3D._embed_boxes still uses 2D box format (4 coords).
    For 3D we need to pass as points instead, or use the 3D version.

    Actually looking at the code more carefully:
    The 3D prompt encoder uses point_embeddings[2] and [3] for box corners,
    but _embed_boxes reshapes to (-1, 2, 2) which is 2D.

    We'll use point prompts (center point) as the primary prompt type,
    which is what SAM-Med3D training uses.
    """
    z0, z1, y0, y1, x0, x1 = bbox_zyxzyx
    # Center point of the bbox
    cz = (z0 + z1) / 2.0
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    # SAM-Med3D expects coords in [D, H, W] order
    point = torch.tensor([[[cz, cy, cx]]], dtype=torch.float32)  # [1, 1, 3]
    label = torch.tensor([[1]], dtype=torch.long)                 # [1, 1] positive
    return point, label


def run_sam_inference(
    model,
    image_1ch: np.ndarray,   # [1, D, H, W] single channel, normalized
    label: np.ndarray,        # [1, D, H, W] GT label (for prompt extraction)
    roi_size: tuple,
    device: torch.device,
    prompt_mode: str = "center_point",  # "center_point" or "per_lesion"
) -> dict:
    """
    Run SAM-Med3D inference on one case.
    Returns dict with pred_mask, dice, and per-lesion metrics.
    """
    D, H, W = image_1ch.shape[1:]

    # Resize to roi_size
    img_resized = resize_volume(image_1ch, roi_size)   # [1, roi, roi, roi]
    lbl_resized = resize_volume(label.astype(np.float32), roi_size)  # [1, roi, roi, roi]
    lbl_resized = (lbl_resized > 0.5).astype(np.float32)

    img_t = torch.from_numpy(img_resized).float().unsqueeze(0).to(device)  # [1, 1, roi, roi, roi]
    lbl_t = torch.from_numpy(lbl_resized).float()

    has_lesion = lbl_t.sum() > 0

    if not has_lesion:
        # Negative case: run with no prompt, expect empty prediction
        with torch.no_grad():
            img_emb = model.image_encoder(img_t)
            sparse, dense = model.prompt_encoder(points=None, boxes=None, masks=None)
            masks_logits, iou_pred = model.mask_decoder(
                image_embeddings=img_emb,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            masks_logits = F.interpolate(
                masks_logits, size=roi_size, mode="trilinear", align_corners=False
            )
        pred = (torch.sigmoid(masks_logits[0, 0]) > 0.5).cpu().numpy()
        return {
            "has_lesion": False,
            "pred_voxels": int(pred.sum()),
            "dice": 1.0 if pred.sum() == 0 else 0.0,  # TN is good
            "is_true_negative": pred.sum() == 0,
        }

    # Positive case: use GT bbox center as point prompt
    # Get bbox in resized space
    scale = np.array(roi_size) / np.array([D, H, W])
    gt_bbox = get_gt_bbox_3d(label)
    if gt_bbox is None:
        return {"has_lesion": True, "dice": 0.0, "error": "no_bbox"}

    # Scale bbox to roi_size
    z0, z1, y0, y1, x0, x1 = gt_bbox
    cz = ((z0 + z1) / 2.0) * scale[0]
    cy = ((y0 + y1) / 2.0) * scale[1]
    cx = ((x0 + x1) / 2.0) * scale[2]

    point_coords = torch.tensor([[[cz, cy, cx]]], dtype=torch.float32).to(device)  # [1, 1, 3]
    point_labels = torch.tensor([[1]], dtype=torch.long).to(device)                # [1, 1]

    with torch.no_grad():
        img_emb = model.image_encoder(img_t)
        sparse, dense = model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        # Call mask_decoder directly (not sam3D.forward) to avoid
        # postprocess_masks hardcoded img_size=128 issue.
        # mask_decoder outputs [B, 1, roi/4, roi/4, roi/4]
        masks_logits, iou_pred = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        # Upsample from stride-4 back to roi_size
        masks_logits = F.interpolate(
            masks_logits, size=roi_size, mode="trilinear", align_corners=False
        )

    pred_prob = torch.sigmoid(masks_logits[0, 0]).cpu()  # [roi, roi, roi]
    pred_mask = (pred_prob > 0.5).float()

    # Compute Dice in resized space
    gt = lbl_t[0]
    intersection = (pred_mask * gt).sum()
    denom = pred_mask.sum() + gt.sum()
    dice = float((2 * intersection + 1e-6) / (denom + 1e-6))

    # Lesion recall: does prediction overlap with GT?
    gt_labeled, n_gt = ndimage.label(gt.numpy() > 0.5, structure=np.ones((3, 3, 3)))
    hit = 0
    for comp_id in range(1, n_gt + 1):
        comp_mask = torch.from_numpy((gt_labeled == comp_id).astype(np.float32))
        if (pred_mask * comp_mask).sum() > 0:
            hit += 1

    gt_voxels = int(gt.sum())
    bucket = voxel_bucket(gt_voxels)

    return {
        "has_lesion": True,
        "dice": dice,
        "iou_pred": float(iou_pred[0, 0]),
        "pred_voxels": int(pred_mask.sum()),
        "gt_voxels": gt_voxels,
        "gt_lesions": n_gt,
        "hit_lesions": hit,
        "lesion_recall": hit / max(n_gt, 1),
        "voxel_bucket": bucket,
    }


def voxel_bucket(n: int) -> str:
    if n <= 100:   return "le_100"
    if n <= 500:   return "100_500"
    if n <= 1000:  return "500_1k"
    if n <= 5000:  return "1k_5k"
    return "gt_5k"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: SAM-Med3D zero-shot eval on PI-CAI")
    parser.add_argument("--sam-checkpoint", type=str, default=None,
                        help="Path to sam_med3d_turbo.pth or sam_med3d.pth. "
                             "If not provided, uses random weights (sanity check only).")
    parser.add_argument("--val-split", type=str,
                        default="data/splits/5fold/fold_0/val.txt")
    parser.add_argument("--processed-root", type=str,
                        default="../picai_preprocessing_project/data/processed/picai_profound_prompt_v2")
    parser.add_argument("--output-dir", type=str, default="outputs/phase0_zeroshot")
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128],
                        metavar=("D", "H", "W"))
    parser.add_argument("--channel", type=str, default="avg",
                        choices=["t2w", "adc", "hbv", "avg"],
                        help="Which channel to feed SAM-Med3D (single-channel model). "
                             "'avg' averages all three modalities.")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases for quick smoke test.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    roi_size = tuple(args.roi_size)

    # ── Build SAM-Med3D ──────────────────────────────────────────────────────
    from segment_anything.build_sam3D import build_sam3D_vit_b_ori
    print(f"Building SAM-Med3D vit_b_ori (img_size=128)...")
    model = build_sam3D_vit_b_ori(checkpoint=args.sam_checkpoint)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")
    if args.sam_checkpoint:
        print(f"  Loaded weights from: {args.sam_checkpoint}")
    else:
        print("  WARNING: No checkpoint provided — using random weights (sanity check only)")

    # ── Load val split ───────────────────────────────────────────────────────
    val_split = Path(args.val_split)
    if not val_split.is_absolute():
        val_split = ROOT / val_split
    case_ids = [l.strip() for l in val_split.read_text().splitlines() if l.strip()]
    if args.max_cases:
        case_ids = case_ids[:args.max_cases]
    print(f"Val cases: {len(case_ids)}")

    processed_root = Path(args.processed_root)
    if not processed_root.is_absolute():
        processed_root = ROOT / processed_root

    # ── Run evaluation ───────────────────────────────────────────────────────
    channel_map = {"t2w": 0, "adc": 1, "hbv": 2}
    results = []
    errors = []

    for case_id in tqdm(case_ids, desc="Zero-shot eval"):
        # Find npz
        npz_candidates = list(processed_root.rglob(f"{case_id}.npz"))
        if not npz_candidates:
            errors.append({"case_id": case_id, "error": "npz_not_found"})
            continue
        npz_path = npz_candidates[0]

        try:
            image, label, _ = load_case(npz_path)
        except Exception as e:
            errors.append({"case_id": case_id, "error": str(e)})
            continue

        # Prepare single-channel input
        if args.channel == "avg":
            image_1ch = prepare_three_channel_avg(image)
        else:
            image_1ch = prepare_single_channel(image, channel_map[args.channel])

        try:
            result = run_sam_inference(
                model=model,
                image_1ch=image_1ch,
                label=label,
                roi_size=roi_size,
                device=device,
            )
            result["case_id"] = case_id
            results.append(result)
        except Exception as e:
            errors.append({"case_id": case_id, "error": str(e)})

    # ── Aggregate metrics ────────────────────────────────────────────────────
    positive_results = [r for r in results if r.get("has_lesion", False)]
    negative_results = [r for r in results if not r.get("has_lesion", False)]

    pos_dice = [r["dice"] for r in positive_results if "dice" in r]
    all_dice = [r["dice"] for r in results if "dice" in r]
    lesion_recalls = [r["lesion_recall"] for r in positive_results if "lesion_recall" in r]
    total_gt_lesions = sum(r.get("gt_lesions", 0) for r in positive_results)
    total_hit_lesions = sum(r.get("hit_lesions", 0) for r in positive_results)
    tn_rate = sum(1 for r in negative_results if r.get("is_true_negative", False)) / max(len(negative_results), 1)

    # Per-bucket Dice
    buckets = ["le_100", "100_500", "500_1k", "1k_5k", "gt_5k"]
    bucket_dice = {}
    for b in buckets:
        b_vals = [r["dice"] for r in positive_results if r.get("voxel_bucket") == b]
        if b_vals:
            bucket_dice[b] = {"mean": float(np.mean(b_vals)), "n": len(b_vals)}

    summary = {
        "total_cases": len(results),
        "positive_cases": len(positive_results),
        "negative_cases": len(negative_results),
        "errors": len(errors),
        "channel_mode": args.channel,
        "roi_size": list(roi_size),
        "checkpoint": args.sam_checkpoint or "random_init",
        # Key metrics
        "overall_dice": float(np.mean(all_dice)) if all_dice else 0.0,
        "positive_case_dice": float(np.mean(pos_dice)) if pos_dice else 0.0,
        "lesion_recall": total_hit_lesions / max(total_gt_lesions, 1),
        "total_gt_lesions": total_gt_lesions,
        "total_hit_lesions": total_hit_lesions,
        "true_negative_rate": tn_rate,
        "bucket_dice": bucket_dice,
    }

    # ── Print summary ────────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("PHASE 0: SAM-Med3D Zero-Shot Evaluation Summary")
    print(sep)
    print(f"  Cases evaluated : {summary['total_cases']} ({summary['positive_cases']} pos, {summary['negative_cases']} neg)")
    print(f"  Errors          : {summary['errors']}")
    print(f"  Channel mode    : {summary['channel_mode']}")
    print(f"  ROI size        : {roi_size}")
    print(f"  Checkpoint      : {summary['checkpoint']}")
    print(f"")
    print(f"  ── Key Metrics ──────────────────────────────────────────────")
    print(f"  Overall Dice         : {summary['overall_dice']:.4f}")
    print(f"  Positive-case Dice   : {summary['positive_case_dice']:.4f}  ← main indicator")
    print(f"  Lesion Recall        : {summary['lesion_recall']:.4f}  ({summary['total_hit_lesions']}/{summary['total_gt_lesions']} lesions)")
    print(f"  True Negative Rate   : {summary['true_negative_rate']:.4f}")
    print(f"")
    print(f"  ── Dice by Lesion Size (GT voxels in resized space) ─────────")
    for b in buckets:
        if b in bucket_dice:
            print(f"  {b:>10}: Dice={bucket_dice[b]['mean']:.4f}  (n={bucket_dice[b]['n']})")
    print(f"")

    # ── Go/No-Go decision ────────────────────────────────────────────────────
    pos_dice_val = summary["positive_case_dice"]
    print(f"  ── Go/No-Go Decision ────────────────────────────────────────")
    if pos_dice_val >= 0.40:
        verdict = "✅ GO  — Positive-case Dice >= 0.40. Proceed to Phase 1 (fine-tune SAM-Med3D on PI-CAI)."
    elif pos_dice_val >= 0.20:
        verdict = "⚠️  MARGINAL — Dice in [0.20, 0.40). Fine-tune first, then re-evaluate before Phase 2."
    else:
        verdict = "❌ NO-GO — Positive-case Dice < 0.20. SAM-Med3D may not suit this task. Consider pivoting."
    print(f"  {verdict}")
    print(sep)

    # ── Save results ─────────────────────────────────────────────────────────
    def _json_safe(obj):
        """Recursively convert numpy/bool types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, bool):
            return obj
        return obj

    (output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    (output_dir / "case_results.json").write_text(json.dumps(_json_safe(results), indent=2))
    if errors:
        (output_dir / "errors.json").write_text(json.dumps(_json_safe(errors), indent=2))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
