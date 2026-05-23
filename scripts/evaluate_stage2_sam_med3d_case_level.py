#!/usr/bin/env python
"""Case-level evaluation for SAM-Med3D Stage-2 refinement.

Merges per-proposal patch predictions back to full case volumes,
then computes Dice / Precision / Recall / lesion_recall / FP-per-case.

Directly comparable to:
  outputs/coarse_score_es_3090/fold_0/stage2_refinement_v3_multithr_gtjitter/
  case_eval_mask060_obj010_min1/summary.md
  (v3 baseline: dice=0.4595, lesion_recall=0.8621, FP/case=1.47)

Usage:
    /root/anaconda3/envs/lm/bin/python scripts/evaluate_stage2_sam_med3d_case_level.py \
        --checkpoint outputs/stage2_sam_med3d_v1/fold_0/checkpoints/best_by_val_recall_safe_dice.pth \
        --prompt-csv outputs/coarse_score_es_3090/fold_0/stage2_data_v3/val/prompts/coarse_prompts_multithreshold.csv \
        --processed-root ../picai_preprocessing_project/data/processed/picai_profound_prompt_v2 \
        --output-dir outputs/stage2_sam_med3d_v1/fold_0/case_eval \
        --mask-threshold 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

from src.datasets.stage2_sam_med3d_dataset import Stage2SAMMed3DDataset, stage2_sam_med3d_collate_fn
from src.models.sam_med3d_integration import build_sam_med3d_stage2
from src.utils.checkpoint import load_checkpoint
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_label(processed_root: Path, case_id: str) -> np.ndarray:
    direct = processed_root / f"{case_id}.npz"
    if not direct.is_file():
        matches = list(processed_root.rglob(f"{case_id}.npz"))
        if not matches:
            raise FileNotFoundError(f"NPZ not found for {case_id}")
        direct = matches[0]
    with np.load(direct, allow_pickle=False) as d:
        label = d["label"].astype(np.float32)
    return label[0] if label.ndim == 4 else label  # [D, H, W]


def _counts(pred: np.ndarray, gt: np.ndarray):
    p = pred > 0; g = gt > 0
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    return tp, fp, fn


def _component_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    struct = np.ones((3, 3, 3), dtype=np.uint8)
    gt_cc, n_gt = ndimage.label(gt > 0, structure=struct)
    pred_cc, n_pred = ndimage.label(pred > 0, structure=struct)
    hit = 0
    tp_comp = fp_comp = 0
    for pid in range(1, n_pred + 1):
        comp = pred_cc == pid
        if np.logical_and(comp, gt > 0).any():
            tp_comp += 1
        else:
            fp_comp += 1
    for gid in range(1, n_gt + 1):
        if np.logical_and(pred > 0, gt_cc == gid).any():
            hit += 1
    return {
        "hit_lesions": float(hit),
        "total_gt_lesions": float(n_gt),
        "pred_components": float(n_pred),
        "tp_pred_components": float(tp_comp),
        "fp_pred_components": float(fp_comp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-csv", required=True)
    parser.add_argument("--processed-root",
                        default="../picai_preprocessing_project/data/processed/picai_profound_prompt_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--crop-margin-ratio", type=float, default=1.5)
    parser.add_argument("--target-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=None)
    # SAM-Med3D model config
    parser.add_argument("--sam-checkpoint",
                        default="/opt/data/private/lm/project-segmentation-for-MIR/SAM-Med3D/ckpt/sam_med3d_turbo.pth")
    parser.add_argument("--in-chans", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = Path(args.processed_root)
    if not processed_root.is_absolute():
        processed_root = ROOT / processed_root
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    target_size = tuple(args.target_size)

    # ── Build model ──────────────────────────────────────────────────────────
    print(f"Loading model from: {args.checkpoint}")
    model = build_sam_med3d_stage2({
        "sam_checkpoint_path": args.sam_checkpoint,
        "in_chans": args.in_chans,
        "roi_size": target_size[0],
        "freeze_image_encoder": False,
    }).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print("Model loaded OK")

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds = Stage2SAMMed3DDataset(
        processed_root=processed_root,
        prompt_csv=args.prompt_csv,
        crop_margin_ratio=args.crop_margin_ratio,
        target_size=target_size,
        normalize="channelwise_nonzero",
        max_prompts=args.max_prompts,
        positive_only=False,
        negative_ratio=None,
        point_jitter_voxels=0,  # no jitter at eval
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=stage2_sam_med3d_collate_fn,
    )
    print(f"Eval prompts: {len(ds)}")

    # ── Forward pass: collect per-proposal predictions ────────────────────────
    # Each proposal: probability in 128³ space, plus crop_box to map back
    proposal_cache: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Forward", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            pt_coords = batch["point_coords"].to(device, non_blocking=True)
            pt_labels = batch["point_label"].to(device, non_blocking=True)

            masks_logits, iou_pred = model(images, pt_coords, pt_labels)
            probs = torch.sigmoid(masks_logits).cpu().numpy()  # [B, 1, 128, 128, 128]

            for i in range(len(batch["case_id"])):
                proposal_cache.append({
                    "case_id": batch["case_id"][i],
                    "proposal_rank": int(batch["proposal_rank"][i]),
                    "prob_128": probs[i, 0],                                    # [128, 128, 128]
                    "crop_box": batch["crop_box_zyxzyx"][i].numpy().tolist(),   # [z0,z1,y0,y1,x0,x1]
                    "original_shape": batch["original_shape_zyx"][i].numpy().tolist(),  # [D,H,W]
                    "iou_pred": float(iou_pred[i, 0]),
                })

    # ── Merge proposals back to case-level volumes ────────────────────────────
    print("Merging proposals to case volumes...")
    merged_prob: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}

    for item in tqdm(proposal_cache, desc="Merge", dynamic_ncols=True):
        case_id = item["case_id"]
        orig_shape = tuple(item["original_shape"])  # (D, H, W)
        crop_box = item["crop_box"]                 # [z0, z1, y0, y1, x0, x1]
        z0, z1, y0, y1, x0, x1 = crop_box
        crop_size = (z1 - z0, y1 - y0, x1 - x0)

        # Resize 128³ prediction back to native crop size
        prob_t = torch.from_numpy(item["prob_128"]).float().unsqueeze(0).unsqueeze(0)  # [1,1,128,128,128]
        prob_native = F.interpolate(
            prob_t, size=crop_size, mode="trilinear", align_corners=False
        )[0, 0].numpy()  # [d, h, w]

        # Initialize case volume if needed
        if case_id not in merged_prob:
            merged_prob[case_id] = np.zeros(orig_shape, dtype=np.float32)
            labels[case_id] = _load_label(processed_root, case_id)

        # Max-merge into case volume
        target = merged_prob[case_id][z0:z1, y0:y1, x0:x1]
        merged_prob[case_id][z0:z1, y0:y1, x0:x1] = np.maximum(target, prob_native)

    # ── Case-level metrics ────────────────────────────────────────────────────
    print("Computing case-level metrics...")
    case_rows = []
    total_tp = total_fp = total_fn = 0.0
    total_hit = total_gt_lesions = total_pred_comp = total_fp_comp = total_tp_comp = 0.0

    for case_id in sorted(merged_prob.keys()):
        prob = merged_prob[case_id]
        gt = labels[case_id]
        pred = (prob >= args.mask_threshold).astype(np.uint8)

        tp, fp, fn = _counts(pred, gt > 0)
        comp = _component_stats(pred, gt > 0)
        dice = 2 * tp / max(2 * tp + fp + fn, 1e-8)
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)

        case_rows.append({
            "case_id": case_id,
            "dice": round(dice, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "gt_voxels": int((gt > 0).sum()),
            "pred_voxels": int(pred.sum()),
            **{k: round(v, 4) for k, v in comp.items()},
        })

        total_tp += tp; total_fp += fp; total_fn += fn
        total_hit += comp["hit_lesions"]
        total_gt_lesions += comp["total_gt_lesions"]
        total_pred_comp += comp["pred_components"]
        total_tp_comp += comp["tp_pred_components"]
        total_fp_comp += comp["fp_pred_components"]

        if args.save_predictions:
            np.savez_compressed(
                output_dir / f"{case_id}_stage2_pred.npz",
                probability=prob.astype(np.float32),
                mask=pred.astype(np.uint8),
                case_id=np.array(case_id),
            )

    n_cases = len(case_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "prompt_csv": str(args.prompt_csv),
        "mask_threshold": args.mask_threshold,
        "cases": n_cases,
        # ── Key metrics ──
        "dice": round(2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1e-8), 6),
        "precision": round(total_tp / max(total_tp + total_fp, 1e-8), 6),
        "recall": round(total_tp / max(total_tp + total_fn, 1e-8), 6),
        "lesion_recall": round(total_hit / max(total_gt_lesions, 1e-8), 6),
        "hit_lesions": total_hit,
        "total_gt_lesions": total_gt_lesions,
        "pred_components_per_case": round(total_pred_comp / max(n_cases, 1), 4),
        "fp_components_per_case": round(total_fp_comp / max(n_cases, 1), 4),
        "component_precision": round(total_tp_comp / max(total_pred_comp, 1e-8), 4),
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    with (output_dir / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader(); writer.writerows(case_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("SAM-Med3D Stage-2 Case-Level Evaluation")
    print(sep)
    print(f"  Checkpoint   : {Path(args.checkpoint).name}")
    print(f"  Cases        : {n_cases}")
    print(f"  Threshold    : {args.mask_threshold}")
    print()
    print(f"  ── Voxel Metrics ─────────────────────────────────────────────")
    print(f"  Dice                  : {summary['dice']:.4f}")
    print(f"  Precision             : {summary['precision']:.4f}")
    print(f"  Recall                : {summary['recall']:.4f}")
    print()
    print(f"  ── Lesion Metrics ────────────────────────────────────────────")
    print(f"  Lesion Recall         : {summary['lesion_recall']:.4f}  "
          f"({int(summary['hit_lesions'])}/{int(summary['total_gt_lesions'])} lesions)")
    print(f"  Pred components/case  : {summary['pred_components_per_case']:.4f}")
    print(f"  FP components/case    : {summary['fp_components_per_case']:.4f}")
    print(f"  Component precision   : {summary['component_precision']:.4f}")
    print()
    print(f"  ── Comparison with v3 baseline ───────────────────────────────")
    print(f"  {'Metric':<25} {'SAM-Med3D v1':>14} {'v3 baseline':>14}")
    print(f"  {'-'*53}")
    v3 = {"dice": 0.4595, "precision": 0.3403, "recall": 0.7071,
          "lesion_recall": 0.8621, "fp_components_per_case": 1.4696}
    for k, v3v in v3.items():
        cur = summary.get(k, float("nan"))
        delta = cur - v3v
        sign = "▲" if delta > 0 else "▼"
        print(f"  {k:<25} {cur:>14.4f} {v3v:>14.4f}  {sign}{abs(delta):.4f}")
    print(sep)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
