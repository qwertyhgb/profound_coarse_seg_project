#!/usr/bin/env python
"""Merge Stage-2 refined proposal masks back to cases and evaluate full volumes."""
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
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.collate import stage2_prompt_collate_fn
from src.datasets.stage2_prompt_dataset import Stage2PromptDataset
from src.models.build_model import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config_utils import load_config


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


def _extract_output(model_out):
    if isinstance(model_out, dict):
        return model_out["logits"], model_out.get("objectness_logits")
    return model_out, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_stage2_refinement.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt-csv", default=None)
    parser.add_argument("--coarse-pred-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--use-objectness-filter", action="store_true")
    parser.add_argument(
        "--weight-by-objectness",
        action="store_true",
        help="Multiply each refined patch probability by its objectness probability before merging.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--min-prompts-per-case", type=int, default=0, help="When objectness filtering is enabled, always keep at least the top-N proposals per case.")
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    output_root = Path(cfg.get("logging", {}).get("output_root", "outputs/stage2_refinement"))
    checkpoint = args.checkpoint or str(output_root / "checkpoints" / "best_by_val_dice.pth")
    prompt_csv = args.prompt_csv or data_cfg["val_prompts"]
    coarse_root = args.coarse_pred_root or data_cfg["val_coarse_pred_root"]
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "case_level_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_threshold = float(args.mask_threshold if args.mask_threshold is not None else cfg.get("metrics", {}).get("threshold", 0.5))
    processed_root = Path(data_cfg["processed_root"])

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

    merged: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    objectness_rows = []
    proposal_cache = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Stage2 forward cache", dynamic_ncols=True):
            case_id = batch["case_id"][0]
            if case_id not in labels:
                labels[case_id] = _load_label(processed_root, case_id)
                merged[case_id] = np.zeros_like(labels[case_id], dtype=np.float32)
            image = batch["image"].to(device)
            coarse_prob = batch["coarse_prob"].to(device)
            box_prior = batch["box_prior"].to(device)
            point_prior = batch["point_prior"].to(device)
            logits, obj_logits = _extract_output(model(image, coarse_prob, box_prior, point_prior))
            prob_patch = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            obj_prob = 1.0 if obj_logits is None else float(torch.sigmoid(obj_logits)[0].detach().cpu().item())
            proposal_rank = int(batch["proposal_rank"][0])
            objectness_rows.append({"case_id": case_id, "proposal_rank": proposal_rank, "objectness": obj_prob})
            cs = batch["crop_start_zyx"][0].numpy().tolist()
            ce = batch["crop_end_zyx"][0].numpy().tolist()
            ps = batch["patch_valid_start_zyx"][0].numpy().tolist()
            pe = batch["patch_valid_end_zyx"][0].numpy().tolist()
            patch_valid = prob_patch[ps[0]:pe[0], ps[1]:pe[1], ps[2]:pe[2]].astype(np.float32)
            proposal_cache.append({
                "proposal_index": len(proposal_cache),
                "case_id": case_id,
                "proposal_rank": proposal_rank,
                "objectness": obj_prob,
                "prob": patch_valid,
                "crop_start_zyx": cs,
                "crop_end_zyx": ce,
            })

    fallback_keep: set[int] = set()
    if args.use_objectness_filter and int(args.min_prompts_per_case) > 0:
        by_case: dict[str, list[tuple[int, float]]] = {}
        for item in proposal_cache:
            by_case.setdefault(item["case_id"], []).append((int(item["proposal_index"]), float(item["objectness"])))
        for case_items in by_case.values():
            case_items.sort(key=lambda pair: pair[1], reverse=True)
            fallback_keep.update(idx for idx, _ in case_items[: int(args.min_prompts_per_case)])

    for item in proposal_cache:
        idx = int(item["proposal_index"])
        if args.use_objectness_filter and item["objectness"] < float(args.objectness_threshold) and idx not in fallback_keep:
            continue
        prob_patch = item["prob"]
        if args.weight_by_objectness:
            prob_patch = prob_patch * float(item["objectness"])
        cs = item["crop_start_zyx"]
        ce = item["crop_end_zyx"]
        target = merged[item["case_id"]][cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]]
        merged[item["case_id"]][cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]] = np.maximum(target, prob_patch)

    rows = []
    total_tp = total_fp = total_fn = 0.0
    total_hit = total_gt_lesions = total_fp_comp = total_pred_comp = total_tp_comp = 0.0
    for case_id, prob in sorted(merged.items()):
        gt = labels[case_id]
        pred = prob >= mask_threshold
        tp, fp, fn = _counts(pred, gt > 0)
        comp = _component_stats(pred, gt > 0, min_component_size=int(cfg.get("metrics", {}).get("min_pred_component_size", 0)))
        dice = 2 * tp / max(2 * tp + fp + fn, 1e-8)
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        rows.append({
            "case_id": case_id,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "gt_voxels": int((gt > 0).sum()),
            "pred_voxels": int(pred.sum()),
            **comp,
        })
        total_tp += tp; total_fp += fp; total_fn += fn
        total_hit += comp["hit_lesions"]; total_gt_lesions += comp["total_gt_lesions"]
        total_pred_comp += comp["pred_components"]; total_tp_comp += comp["tp_pred_components"]; total_fp_comp += comp["fp_pred_components"]
        if args.save_predictions:
            np.savez_compressed(output_dir / f"{case_id}_stage2_pred.npz", probability=prob.astype(np.float32), mask=pred.astype(np.uint8), case_id=np.array(case_id))

    summary = {
        "checkpoint": checkpoint,
        "prompt_csv": prompt_csv,
        "mask_threshold": mask_threshold,
        "use_objectness_filter": bool(args.use_objectness_filter),
        "weight_by_objectness": bool(args.weight_by_objectness),
        "objectness_threshold": float(args.objectness_threshold),
        "min_prompts_per_case": int(args.min_prompts_per_case),
        "cases": len(rows),
        "dice": 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1e-8),
        "precision": total_tp / max(total_tp + total_fp, 1e-8),
        "recall": total_tp / max(total_tp + total_fn, 1e-8),
        "lesion_recall": total_hit / max(total_gt_lesions, 1e-8),
        "hit_lesions": total_hit,
        "total_gt_lesions": total_gt_lesions,
        "pred_components_per_case": total_pred_comp / max(len(rows), 1),
        "fp_components_per_case": total_fp_comp / max(len(rows), 1),
        "component_precision": total_tp_comp / max(total_pred_comp, 1e-8),
    }

    with (output_dir / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "proposal_objectness.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(objectness_rows[0].keys()) if objectness_rows else ["case_id", "proposal_rank", "objectness"])
        writer.writeheader(); writer.writerows(objectness_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Stage-2 Case-Level Evaluation", ""] + [f"- {k}: {v}" for k, v in summary.items()]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved Stage-2 case-level evaluation to {output_dir}")


if __name__ == "__main__":
    main()
