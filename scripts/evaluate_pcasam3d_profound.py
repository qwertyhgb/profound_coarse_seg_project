#!/usr/bin/env python
"""Evaluate PCaSAM-3D-ProFound at case level.

Loads each validation/test case, resizes to 128^3, runs the model,
and reports per-case and aggregate metrics including:
- Dice (all, positive, negative cases)
- Precision, Recall
- Lesion detection rate
- False positive case rate
- Coarse branch quality metrics

Usage:
    /root/anaconda3/envs/lm/bin/python scripts/evaluate_pcasam3d_profound.py \
        --config configs/train_pcasam3d_profound.yaml \
        --checkpoint outputs/pcasam3d_profound/fold_0/checkpoints/best_by_val_recall_safe_dice.pth \
        --split data/splits/5fold/fold_0/val.txt \
        --output-dir outputs/pcasam3d_profound/fold_0/evaluation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.pcasam3d_dataset import PCaSAM3DDataset, pcasam3d_collate_fn
from src.models.pcasam3d_profound import build_pcasam3d_profound
from src.utils.config_utils import load_config
from src.utils.seed import set_seed


def compute_case_metrics(pred_logits: torch.Tensor, label: torch.Tensor, threshold: float = 0.5):
    """Compute per-case segmentation metrics."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    gt = (label > 0.5).float()

    dims = tuple(range(1, pred.ndim))
    tp = (pred * gt).sum(dim=dims).item()
    fp = (pred * (1 - gt)).sum(dim=dims).item()
    fn = ((1 - pred) * gt).sum(dim=dims).item()

    has_lesion_gt = gt.sum().item() > 0
    has_lesion_pred = pred.sum().item() > 0

    denom = 2 * tp + fp + fn
    dice = (2 * tp) / (denom + 1e-6) if denom > 0 else 1.0
    precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else (1.0 if not has_lesion_gt else 0.0)
    recall = tp / (tp + fn + 1e-6) if has_lesion_gt else (1.0 if not has_lesion_pred else 0.0)

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "has_lesion_gt": bool(has_lesion_gt),
        "has_lesion_pred": bool(has_lesion_pred),
        "n_voxel_gt": int(gt.sum().item()),
        "n_voxel_pred": int(pred.sum().item()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def main():
    parser = argparse.ArgumentParser("PCaSAM-3D-ProFound Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=None, help="Override val split file")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--zoom-refine", action="store_true", help="Use top-box zoom-in refinement and evaluate the pasted result.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load checkpoint
    model = build_pcasam3d_profound(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Dataset
    data_cfg = cfg["data"]
    split_file = args.split or data_cfg.get("val_split")
    patch_size = tuple(data_cfg.get("patch_size", [128, 128, 128]))

    dataset = PCaSAM3DDataset(
        processed_root=data_cfg["processed_root"],
        split_file=split_file,
        mode="val",
        patch_size=patch_size,
        normalize=data_cfg.get("normalize", "channelwise_nonzero"),
        max_cases=args.max_cases,
    )
    print(f"Evaluating {len(dataset)} cases | threshold={args.threshold}")

    # Evaluate
    results = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["label"].unsqueeze(0).to(device)
            case_id = sample["case_id"]

            output = model.forward_zoom_in(image) if args.zoom_refine else model(image)
            refined_logits = output["zoom_refined_logits"] if args.zoom_refine else output["refined_logits"]
            coarse_logits = output["coarse_logits"]

            # Refined metrics
            refined_m = compute_case_metrics(refined_logits, label, args.threshold)
            # Coarse metrics
            coarse_m = compute_case_metrics(coarse_logits, label, float(cfg.get("metrics", {}).get("coarse_threshold", 0.3)))

            results.append({
                "case_id": case_id,
                **{f"refined_{k}": v for k, v in refined_m.items()},
                **{f"coarse_{k}": v for k, v in coarse_m.items()},
                "zoom_used": bool(output.get("zoom_used", torch.zeros(1, dtype=torch.bool, device=device))[0].item()) if args.zoom_refine else False,
            })

    # Aggregate
    all_refined_dice = [r["refined_dice"] for r in results]
    pos_refined_dice = [r["refined_dice"] for r in results if r["refined_has_lesion_gt"]]
    neg_refined_dice = [r["refined_dice"] for r in results if not r["refined_has_lesion_gt"]]

    # Detection metrics
    n_pos = sum(1 for r in results if r["refined_has_lesion_gt"])
    n_neg = sum(1 for r in results if not r["refined_has_lesion_gt"])
    detected = sum(1 for r in results if r["refined_has_lesion_gt"] and r["refined_has_lesion_pred"])
    false_pos = sum(1 for r in results if not r["refined_has_lesion_gt"] and r["refined_has_lesion_pred"])

    summary = {
        "n_total": len(results),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": args.threshold,
        "zoom_refine": bool(args.zoom_refine),
        "zoom_used_rate": sum(1 for r in results if r.get("zoom_used", False)) / max(len(results), 1),
        "refined_mean_dice_all": float(np.mean(all_refined_dice)) if all_refined_dice else 0.0,
        "refined_mean_dice_positive": float(np.mean(pos_refined_dice)) if pos_refined_dice else 0.0,
        "refined_mean_dice_negative": float(np.mean(neg_refined_dice)) if neg_refined_dice else 0.0,
        "refined_detection_rate": detected / max(n_pos, 1),
        "refined_false_positive_rate": false_pos / max(n_neg, 1),
        "refined_mean_precision": float(np.mean([r["refined_precision"] for r in results])),
        "refined_mean_recall": float(np.mean([r["refined_recall"] for r in results if r["refined_has_lesion_gt"]])) if n_pos > 0 else 0.0,
        # Coarse branch quality
        "coarse_mean_dice_positive": float(np.mean([r["coarse_dice"] for r in results if r["coarse_has_lesion_gt"]])) if n_pos > 0 else 0.0,
        "coarse_detection_rate": sum(1 for r in results if r["coarse_has_lesion_gt"] and r["coarse_has_lesion_pred"]) / max(n_pos, 1),
    }

    # Print
    print("\n" + "=" * 70)
    print("PCaSAM-3D-ProFound Evaluation Results")
    print("=" * 70)
    print(f"  Cases: {summary['n_total']} (pos={summary['n_positive']}, neg={summary['n_negative']})")
    print(f"  Threshold: {summary['threshold']}")
    print(f"  Zoom refine: {summary['zoom_refine']} (used={summary['zoom_used_rate']:.4f})")
    print(f"  --- Refined (final output) ---")
    print(f"  Mean Dice (all):      {summary['refined_mean_dice_all']:.4f}")
    print(f"  Mean Dice (positive): {summary['refined_mean_dice_positive']:.4f}")
    print(f"  Mean Dice (negative): {summary['refined_mean_dice_negative']:.4f}")
    print(f"  Detection rate:       {summary['refined_detection_rate']:.4f}")
    print(f"  False positive rate:  {summary['refined_false_positive_rate']:.4f}")
    print(f"  Mean Precision:       {summary['refined_mean_precision']:.4f}")
    print(f"  Mean Recall (pos):    {summary['refined_mean_recall']:.4f}")
    print(f"  --- Coarse branch ---")
    print(f"  Coarse Dice (pos):    {summary['coarse_mean_dice_positive']:.4f}")
    print(f"  Coarse Detection:     {summary['coarse_detection_rate']:.4f}")
    print("=" * 70)

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/pcasam3d_v1/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "case_results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
