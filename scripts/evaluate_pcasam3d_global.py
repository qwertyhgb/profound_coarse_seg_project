#!/usr/bin/env python
"""Evaluate PCaSAM-3D-ProFound with dataset-global metrics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from src.datasets.pcasam3d_dataset import PCaSAM3DDataset
from src.models.pcasam3d_profound import build_pcasam3d_profound
from src.utils.checkpoint import load_checkpoint
from src.utils.config_utils import load_config
from src.utils.seed import set_seed


def parse_thresholds(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def counts(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    pred_b = pred > 0
    gt_b = gt > 0
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    return tp, fp, fn


def case_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    tp, fp, fn = counts(pred, gt)
    has_gt = bool((gt > 0).any())
    has_pred = bool((pred > 0).any())
    denom = 2.0 * tp + fp + fn
    dice = 2.0 * tp / max(denom, 1e-8) if denom > 0 else 1.0
    precision = tp / max(tp + fp, 1e-8) if has_pred else (1.0 if not has_gt else 0.0)
    recall = tp / max(tp + fn, 1e-8) if has_gt else (1.0 if not has_pred else 0.0)
    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "has_gt": has_gt,
        "has_pred": has_pred,
        "gt_voxels": int((gt > 0).sum()),
        "pred_voxels": int((pred > 0).sum()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def component_stats(pred: np.ndarray, gt: np.ndarray, min_component_size: int = 0) -> dict[str, float]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_cc, n_gt = ndimage.label(gt > 0, structure=structure)
    pred_cc, n_pred = ndimage.label(pred > 0, structure=structure)
    hit = 0
    pred_components = 0
    tp_components = 0
    fp_components = 0
    valid_pred = np.zeros_like(pred, dtype=bool)
    for pid in range(1, n_pred + 1):
        comp = pred_cc == pid
        if min_component_size > 0 and int(comp.sum()) < min_component_size:
            continue
        pred_components += 1
        valid_pred |= comp
        if np.logical_and(comp, gt > 0).any():
            tp_components += 1
        else:
            fp_components += 1
    for gid in range(1, n_gt + 1):
        lesion = gt_cc == gid
        if np.logical_and(valid_pred, lesion).any():
            hit += 1
    return {
        "hit_lesions": float(hit),
        "total_gt_lesions": float(n_gt),
        "pred_components": float(pred_components),
        "tp_components": float(tp_components),
        "fp_components": float(fp_components),
    }


def finalize_threshold(rows: list[dict[str, Any]], threshold: float, n_cases: int) -> dict[str, Any]:
    total_tp = sum(float(r["tp"]) for r in rows)
    total_fp = sum(float(r["fp"]) for r in rows)
    total_fn = sum(float(r["fn"]) for r in rows)
    total_hit = sum(float(r["hit_lesions"]) for r in rows)
    total_gt_lesions = sum(float(r["total_gt_lesions"]) for r in rows)
    total_pred_comp = sum(float(r["pred_components"]) for r in rows)
    total_tp_comp = sum(float(r["tp_components"]) for r in rows)
    total_fp_comp = sum(float(r["fp_components"]) for r in rows)
    pos = [r for r in rows if r["has_gt"]]
    neg = [r for r in rows if not r["has_gt"]]
    return {
        "threshold": float(threshold),
        "n_cases": int(n_cases),
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "global_dice": 2.0 * total_tp / max(2.0 * total_tp + total_fp + total_fn, 1e-8),
        "global_precision": total_tp / max(total_tp + total_fp, 1e-8),
        "global_recall": total_tp / max(total_tp + total_fn, 1e-8),
        "macro_dice_all": float(np.mean([r["dice"] for r in rows])) if rows else 0.0,
        "macro_dice_positive": float(np.mean([r["dice"] for r in pos])) if pos else 0.0,
        "macro_precision_positive": float(np.mean([r["precision"] for r in pos])) if pos else 0.0,
        "macro_recall_positive": float(np.mean([r["recall"] for r in pos])) if pos else 0.0,
        "negative_mean_dice": float(np.mean([r["dice"] for r in neg])) if neg else 0.0,
        "case_detection_rate": sum(1 for r in pos if r["has_pred"]) / max(len(pos), 1),
        "false_positive_case_rate": sum(1 for r in neg if r["has_pred"]) / max(len(neg), 1),
        "lesion_recall": total_hit / max(total_gt_lesions, 1e-8),
        "hit_lesions": total_hit,
        "total_gt_lesions": total_gt_lesions,
        "pred_components_per_case": total_pred_comp / max(n_cases, 1),
        "fp_components_per_case": total_fp_comp / max(n_cases, 1),
        "component_precision": total_tp_comp / max(total_pred_comp, 1e-8),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def selection_score(row: dict[str, Any]) -> float:
    recall_gap = max(0.0, 0.85 - float(row["lesion_recall"]))
    fp_gap = max(0.0, float(row["fp_components_per_case"]) - 1.2)
    return (
        float(row["global_dice"])
        + 0.25 * float(row["macro_dice_positive"])
        + 0.15 * float(row["lesion_recall"])
        - recall_gap
        - 0.05 * float(row["fp_components_per_case"])
        - 0.05 * fp_gap
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser("PCaSAM-3D-ProFound global evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--min-component-size", type=int, default=None)
    parser.add_argument("--zoom-refine", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    thresholds = parse_thresholds(args.thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")

    model = build_pcasam3d_profound(cfg).to(device)
    state = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch={state.get('epoch', 'n/a')})")

    data_cfg = cfg["data"]
    dataset = PCaSAM3DDataset(
        processed_root=data_cfg["processed_root"],
        split_file=args.split or data_cfg.get("val_split"),
        mode="val",
        patch_size=tuple(data_cfg.get("patch_size", [128, 128, 128])),
        normalize=data_cfg.get("normalize", "channelwise_nonzero"),
        max_cases=args.max_cases,
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    min_component_size = int(args.min_component_size if args.min_component_size is not None else cfg.get("metrics", {}).get("min_pred_component_size", 0))
    print(f"Evaluating {len(dataset)} cases | thresholds={thresholds} | min_component_size={min_component_size}")

    per_thr_rows: dict[float, list[dict[str, Any]]] = {thr: [] for thr in thresholds}
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            label_t = sample["label"].unsqueeze(0).to(device)
            case_id = str(sample["case_id"])
            output = model.forward_zoom_in(image) if args.zoom_refine else model(image)
            logits = output["zoom_refined_logits"] if args.zoom_refine else output["refined_logits"]
            prob = torch.sigmoid(logits).detach().float().cpu().numpy()[0, 0]
            gt = (label_t.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
            for thr in thresholds:
                pred = (prob >= thr).astype(np.uint8)
                cm = case_metrics(pred, gt)
                comp = component_stats(pred, gt, min_component_size=min_component_size)
                per_thr_rows[thr].append({"case_id": case_id, "threshold": float(thr), **cm, **comp})

    summary_rows = []
    for thr, rows in per_thr_rows.items():
        summary = finalize_threshold(rows, thr, len(dataset))
        summary["selection_score"] = selection_score(summary)
        summary_rows.append(summary)
        write_csv(output_dir / f"case_metrics_thr_{thr:.2f}.csv", rows)

    summary_rows.sort(key=lambda r: float(r["threshold"]))
    write_csv(output_dir / "threshold_sweep_global.csv", summary_rows)
    best = max(summary_rows, key=lambda r: float(r["selection_score"]))
    best_global = max(summary_rows, key=lambda r: float(r["global_dice"]))
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": state.get("epoch", None),
        "config": str(args.config),
        "split": str(args.split or data_cfg.get("val_split")),
        "zoom_refine": bool(args.zoom_refine),
        "min_component_size": min_component_size,
        "best_by_selection_score": best,
        "best_by_global_dice": best_global,
        "thresholds": summary_rows,
    }
    (output_dir / "summary_global.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 88)
    print("PCaSAM-3D-ProFound Global Evaluation")
    print("=" * 88)
    for name, row in [("Best selection", best), ("Best global Dice", best_global)]:
        print(
            f"{name}: thr={row['threshold']:.2f} | global_dice={row['global_dice']:.4f} | "
            f"macro_pos_dice={row['macro_dice_positive']:.4f} | lesion_recall={row['lesion_recall']:.4f} | "
            f"fp/case={row['fp_components_per_case']:.4f} | global_precision={row['global_precision']:.4f} | "
            f"global_recall={row['global_recall']:.4f}"
        )
    print(f"Results saved to: {output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
