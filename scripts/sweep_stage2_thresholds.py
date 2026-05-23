#!/usr/bin/env python
"""Sweep Stage-2 mask/objectness thresholds at case level.

This script runs Stage-2 refinement once over all proposal prompts, caches the
valid refined patch probabilities, then evaluates many threshold combinations.
It is meant to select a coarse-to-fine inference policy that balances lesion
recall against false-positive components per case.
"""
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.collate import stage2_prompt_collate_fn
from src.datasets.stage2_prompt_dataset import Stage2PromptDataset
from src.models.build_model import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config_utils import load_config


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def _resolve_case_path(processed_root: Path, case_id: str) -> Path:
    direct = processed_root / f"{case_id}.npz"
    if direct.is_file():
        return direct
    matches = list(processed_root.rglob(f"{case_id}.npz"))
    if not matches:
        raise FileNotFoundError(f"Missing case npz for {case_id} under {processed_root}")
    return matches[0]


def _load_label(processed_root: Path, case_id: str) -> np.ndarray:
    with np.load(_resolve_case_path(processed_root, case_id), allow_pickle=False) as data:
        label = data["label"].astype(np.float32)
    return label[0] if label.ndim == 4 else label


def _extract_output(model_out):
    if isinstance(model_out, dict):
        return model_out["logits"], model_out.get("objectness_logits")
    return model_out, None


def _counts(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    p = pred > 0
    g = gt > 0
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    return tp, fp, fn


def _component_stats(pred: np.ndarray, gt: np.ndarray, min_component_size: int = 0) -> dict[str, float]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_cc, n_gt = ndimage.label(gt > 0, structure=structure)
    pred_cc, n_pred = ndimage.label(pred > 0, structure=structure)
    hit = 0
    pred_components = 0
    tp_pred = 0
    fp_pred = 0
    valid_pred = np.zeros_like(pred, dtype=bool)
    for pid in range(1, n_pred + 1):
        comp = pred_cc == pid
        if min_component_size > 0 and int(comp.sum()) < min_component_size:
            continue
        pred_components += 1
        valid_pred |= comp
        if np.logical_and(comp, gt > 0).any():
            tp_pred += 1
        else:
            fp_pred += 1
    for gid in range(1, n_gt + 1):
        lesion = gt_cc == gid
        if np.logical_and(valid_pred, lesion).any():
            hit += 1
    return {
        "hit_lesions": float(hit),
        "total_gt_lesions": float(n_gt),
        "pred_components": float(pred_components),
        "tp_pred_components": float(tp_pred),
        "fp_pred_components": float(fp_pred),
    }


def _score(row: dict[str, Any], target_lesion_recall: float, fp_weight: float, dice_weight: float) -> float:
    lesion_recall = float(row["lesion_recall"])
    shortfall = max(0.0, target_lesion_recall - lesion_recall)
    return (
        lesion_recall
        - 2.0 * shortfall
        - fp_weight * float(row["fp_components_per_case"])
        + dice_weight * float(row["dice"])
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]], best: dict[str, Any], target_lesion_recall: float) -> None:
    ranked = sorted(rows, key=lambda r: float(r["selection_score"]), reverse=True)[:15]
    lines = [
        "# Stage-2 Threshold Sweep",
        "",
        f"- Target lesion recall: {target_lesion_recall:.4f}",
        f"- Recommended mask threshold: {best['mask_threshold']:.2f}",
        f"- Recommended objectness threshold: {best['objectness_threshold']}",
        f"- Recommended use objectness filter: {best['use_objectness_filter']}",
        f"- Best selection score: {best['selection_score']:.4f}",
        f"- Best lesion recall: {best['lesion_recall']:.4f}",
        f"- Best fp/components per case: {best['fp_components_per_case']:.4f}",
        f"- Best Dice: {best['dice']:.4f}",
        "",
        "| mask_thr | obj_filter | obj_thr | Dice | Precision | Recall | Lesion recall | FP/case | Comp precision | Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        obj_thr = row["objectness_threshold"] if row["objectness_threshold"] != "none" else "-"
        lines.append(
            f"| {float(row['mask_threshold']):.2f} | {row['use_objectness_filter']} | {obj_thr} | "
            f"{float(row['dice']):.4f} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['lesion_recall']):.4f} | {float(row['fp_components_per_case']):.4f} | "
            f"{float(row['component_precision']):.4f} | {float(row['selection_score']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _evaluate_strategy(
    proposals: list[dict[str, Any]],
    labels: dict[str, np.ndarray],
    mask_threshold: float,
    objectness_threshold: float | None,
    weight_by_objectness: bool,
    min_prompts_per_case: int,
    min_component_size: int,
) -> dict[str, Any]:
    merged = {case_id: np.zeros_like(label, dtype=np.float32) for case_id, label in labels.items()}
    kept_prompts = 0
    fallback_keep: set[int] = set()
    if objectness_threshold is not None and min_prompts_per_case > 0:
        by_case: dict[str, list[tuple[int, float]]] = {}
        for idx, item in enumerate(proposals):
            by_case.setdefault(item["case_id"], []).append((idx, float(item["objectness"])))
        for case_items in by_case.values():
            case_items.sort(key=lambda pair: pair[1], reverse=True)
            fallback_keep.update(idx for idx, _ in case_items[:min_prompts_per_case])
    for item in proposals:
        idx = int(item["proposal_index"])
        if objectness_threshold is not None and item["objectness"] < objectness_threshold and idx not in fallback_keep:
            continue
        prob = item["prob"].astype(np.float32)
        if weight_by_objectness:
            prob = prob * float(item["objectness"])
        cs, ce = item["crop_start_zyx"], item["crop_end_zyx"]
        target = merged[item["case_id"]][cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]]
        merged[item["case_id"]][cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]] = np.maximum(target, prob)
        kept_prompts += 1

    total_tp = total_fp = total_fn = 0.0
    total_hit = total_gt_lesions = total_pred_comp = total_tp_comp = total_fp_comp = 0.0
    positive_cases = 0
    for case_id, prob in merged.items():
        gt = labels[case_id] > 0
        positive_cases += int(gt.any())
        pred = prob >= mask_threshold
        tp, fp, fn = _counts(pred, gt)
        comp = _component_stats(pred, gt, min_component_size=min_component_size)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_hit += comp["hit_lesions"]
        total_gt_lesions += comp["total_gt_lesions"]
        total_pred_comp += comp["pred_components"]
        total_tp_comp += comp["tp_pred_components"]
        total_fp_comp += comp["fp_pred_components"]
    cases = max(len(labels), 1)
    return {
        "mask_threshold": float(mask_threshold),
        "use_objectness_filter": objectness_threshold is not None,
        "objectness_threshold": "none" if objectness_threshold is None else float(objectness_threshold),
        "weight_by_objectness": bool(weight_by_objectness),
        "cases": len(labels),
        "positive_cases": positive_cases,
        "kept_prompts": kept_prompts,
        "min_prompts_per_case": int(min_prompts_per_case),
        "dice": 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1e-8),
        "precision": total_tp / max(total_tp + total_fp, 1e-8),
        "recall": total_tp / max(total_tp + total_fn, 1e-8),
        "lesion_recall": total_hit / max(total_gt_lesions, 1e-8),
        "hit_lesions": total_hit,
        "total_gt_lesions": total_gt_lesions,
        "pred_components_per_case": total_pred_comp / cases,
        "fp_components_per_case": total_fp_comp / cases,
        "component_precision": total_tp_comp / max(total_pred_comp, 1e-8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_stage2_refinement.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt-csv", default=None)
    parser.add_argument("--coarse-pred-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mask-thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--objectness-thresholds", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--include-no-objectness-filter", action="store_true")
    parser.add_argument("--weight-by-objectness", action="store_true")
    parser.add_argument("--min-prompts-per-case", type=int, default=0)
    parser.add_argument("--target-lesion-recall", type=float, default=0.80)
    parser.add_argument("--fp-weight", type=float, default=0.08)
    parser.add_argument("--dice-weight", type=float, default=0.25)
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    output_root = Path(cfg.get("logging", {}).get("output_root", "outputs/stage2_refinement"))
    checkpoint = args.checkpoint or str(output_root / "checkpoints" / "best_by_val_recall_safe_dice.pth")
    prompt_csv = args.prompt_csv or data_cfg["val_prompts"]
    coarse_root = args.coarse_pred_root or data_cfg["val_coarse_pred_root"]
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "threshold_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = Path(data_cfg["processed_root"])
    min_component_size = int(cfg.get("metrics", {}).get("min_pred_component_size", 0))

    device = torch.device(cfg.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    ds = Stage2PromptDataset(
        processed_root=data_cfg["processed_root"],
        prompt_csv=prompt_csv,
        coarse_pred_root=coarse_root,
        patch_size=data_cfg.get("patch_size", [64, 128, 128]),
        bbox_margin=data_cfg.get("bbox_margin", [4, 12, 12]),
        point_sigma=data_cfg.get("point_sigma", 3.0),
        max_prompts=args.max_prompts,
        use_overlaps_gt_sampling=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=stage2_prompt_collate_fn)

    labels: dict[str, np.ndarray] = {}
    proposals: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Stage2 forward cache", dynamic_ncols=True):
            case_id = batch["case_id"][0]
            if case_id not in labels:
                labels[case_id] = _load_label(processed_root, case_id)
            logits, obj_logits = _extract_output(model(
                batch["image"].to(device),
                batch["coarse_prob"].to(device),
                batch["box_prior"].to(device),
                batch["point_prior"].to(device),
            ))
            prob_patch = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            obj_prob = 1.0 if obj_logits is None else float(torch.sigmoid(obj_logits)[0].detach().cpu().item())
            cs = batch["crop_start_zyx"][0].numpy().astype(int).tolist()
            ce = batch["crop_end_zyx"][0].numpy().astype(int).tolist()
            ps = batch["patch_valid_start_zyx"][0].numpy().astype(int).tolist()
            pe = batch["patch_valid_end_zyx"][0].numpy().astype(int).tolist()
            patch_valid = prob_patch[ps[0]:pe[0], ps[1]:pe[1], ps[2]:pe[2]].astype(np.float16)
            proposals.append({
                "proposal_index": len(proposals),
                "case_id": case_id,
                "objectness": obj_prob,
                "prob": patch_valid,
                "crop_start_zyx": cs,
                "crop_end_zyx": ce,
            })

    rows: list[dict[str, Any]] = []
    mask_thresholds = _parse_floats(args.mask_thresholds)
    obj_thresholds = _parse_floats(args.objectness_thresholds)
    obj_options: list[float | None] = obj_thresholds
    if args.include_no_objectness_filter:
        obj_options = [None] + obj_options
    for mask_thr in tqdm(mask_thresholds, desc="Mask thresholds", dynamic_ncols=True):
        for obj_thr in obj_options:
            row = _evaluate_strategy(
                proposals,
                labels,
                mask_threshold=mask_thr,
                objectness_threshold=obj_thr,
                weight_by_objectness=bool(args.weight_by_objectness),
                min_prompts_per_case=int(args.min_prompts_per_case),
                min_component_size=min_component_size,
            )
            row["selection_score"] = _score(row, args.target_lesion_recall, args.fp_weight, args.dice_weight)
            rows.append(row)

    rows.sort(key=lambda r: float(r["selection_score"]), reverse=True)
    best = rows[0]
    _write_csv(output_dir / "stage2_threshold_sweep.csv", rows)
    (output_dir / "best_strategy.json").write_text(json.dumps(best, indent=2))
    _write_report(output_dir / "stage2_threshold_sweep_summary.md", rows, best, args.target_lesion_recall)
    print(json.dumps(best, indent=2))
    print(f"Saved Stage-2 threshold sweep to {output_dir}")


if __name__ == "__main__":
    main()
