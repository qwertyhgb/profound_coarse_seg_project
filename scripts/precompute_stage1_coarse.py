#!/usr/bin/env python
"""Precompute Stage-1 coarse probability maps for a split.

This prepares Stage-2 training data by saving one coarse probability npz per
case and optional component detail files for prompt generation.
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
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.collate import picai_collate_fn
from src.datasets.picai_npz_dataset import PICAINPZDataset
from src.models.build_model import build_model
from src.trainers.evaluator import Evaluator
from src.utils.checkpoint import load_checkpoint
from src.utils.config_utils import load_config


def _component_box(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)]


def _box_volume(box: list[int]) -> int:
    return int((box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4]))


def apply_gland_mask_to_probability(prob: np.ndarray, npz_path: str, cfg: dict) -> np.ndarray:
    """Apply the same gland-mask suppression used in PCaSAM3D validation."""
    pp_cfg = cfg.get("metrics", {}).get("gland_mask_postprocess", {})
    if not bool(pp_cfg.get("enabled", False)) or not npz_path:
        return prob
    path = Path(npz_path)
    if not path.is_file():
        return prob
    with np.load(path, allow_pickle=False) as data:
        if "gland_mask" not in data.files:
            return prob
        gland = data["gland_mask"].astype(np.float32)
    if gland.ndim == 4:
        gland = gland[0]
    prob3 = prob[0] if prob.ndim == 4 else prob
    if gland.shape != prob3.shape:
        return prob
    mask = gland > float(pp_cfg.get("min_mask_value", 0.5))
    margin = int(pp_cfg.get("margin_voxels", 0))
    if margin > 0:
        structure = np.ones((2 * margin + 1, 2 * margin + 1, 2 * margin + 1), dtype=bool)
        mask = ndimage.binary_dilation(mask, structure=structure)
    outside_logit = float(pp_cfg.get("outside_logit", -12.0))
    outside_prob = 1.0 / (1.0 + np.exp(-outside_logit))
    out = prob.copy()
    if out.ndim == 4:
        out[0, ~mask] = outside_prob
    else:
        out[~mask] = outside_prob
    return out


def analyze_components(prob: np.ndarray, label: np.ndarray, threshold: float, min_component_size: int) -> dict:
    prob3 = prob[0] if prob.ndim == 4 else prob
    gt3 = (label[0] if label.ndim == 4 else label) > 0.5
    pred3 = prob3 >= threshold
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_cc, n_gt = ndimage.label(gt3, structure=structure)
    pred_cc, n_pred = ndimage.label(pred3, structure=structure)
    pred_components = []
    valid_pred_mask = np.zeros_like(pred3, dtype=bool)
    for pred_id in range(1, n_pred + 1):
        comp = pred_cc == pred_id
        voxels = int(comp.sum())
        if voxels < min_component_size:
            continue
        box = _component_box(comp)
        if box is None:
            continue
        valid_pred_mask |= comp
        pred_components.append({
            "component_id": int(pred_id),
            "voxels": voxels,
            "box_zyxzyx": box,
            "box_volume": _box_volume(box),
            "max_probability": float(prob3[comp].max()),
            "mean_probability": float(prob3[comp].mean()),
            "overlaps_gt": bool(np.logical_and(comp, gt3).any()),
        })
    hit = 0
    gt_components = []
    for gt_id in range(1, n_gt + 1):
        lesion = gt_cc == gt_id
        lesion_hit = bool(np.logical_and(valid_pred_mask, lesion).any())
        hit += int(lesion_hit)
        gt_components.append({
            "gt_component_id": int(gt_id),
            "voxels": int(lesion.sum()),
            "box_zyxzyx": _component_box(lesion),
            "hit": lesion_hit,
        })
    tp = sum(1 for c in pred_components if c["overlaps_gt"])
    fp = len(pred_components) - tp
    return {
        "total_gt_lesions": int(n_gt),
        "hit_gt_lesions": int(hit),
        "pred_components": int(len(pred_components)),
        "tp_pred_components": int(tp),
        "fp_pred_components": int(fp),
        "prompt_voxels": int(valid_pred_mask.sum()),
        "gt_voxels": int(gt3.sum()),
        "pred_components_detail": pred_components,
        "gt_components_detail": gt_components,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer_single_case.yaml")
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True, help="Directory for <case_id>_coarse_pred.npz")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-component-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--save-logits", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inf = cfg["inference"]
    checkpoint = args.checkpoint or inf.get("checkpoint_path")
    if not checkpoint:
        raise ValueError("Missing checkpoint. Pass --checkpoint or set inference.checkpoint_path.")
    threshold = float(args.threshold if args.threshold is not None else inf.get("prompt_generation_threshold", 0.25))
    min_component_size = int(args.min_component_size if args.min_component_size is not None else cfg.get("metrics", {}).get("min_pred_component_size", 2))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir) if args.report_dir else out_dir.parent / "proposal_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    ds = PICAINPZDataset(
        processed_root=cfg.get("data", {}).get("processed_root", "../picai_preprocessing_project/data/processed/picai_profound_prompt_v2"),
        split_file=args.split,
        mode="val",
        max_cases=args.max_cases,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=picai_collate_fn)
    predictor = Evaluator(model, loss_fn=None, device=device, inference_cfg=inf, metrics_cfg=cfg.get("metrics", {}))

    rows = []
    component_details = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Precompute Stage1", dynamic_ncols=True):
            image = batch["image"].to(device)
            label = batch["label"].cpu().numpy()[0]
            case_id = batch["case_id"][0]
            logits = predictor.predict(image)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            prob = apply_gland_mask_to_probability(prob, batch.get("npz_path", [""])[0], cfg)
            payload = {"probability": prob.astype(np.float32), "case_id": np.array(case_id), "threshold": np.array(threshold, dtype=np.float32)}
            if args.save_logits:
                payload["logits"] = logits.cpu().numpy()[0].astype(np.float32)
            np.savez_compressed(out_dir / f"{case_id}_coarse_pred.npz", **payload)
            stats = analyze_components(prob, label, threshold=threshold, min_component_size=min_component_size)
            rows.append({
                "case_id": case_id,
                "total_gt_lesions": stats["total_gt_lesions"],
                "hit_gt_lesions": stats["hit_gt_lesions"],
                "pred_components": stats["pred_components"],
                "tp_pred_components": stats["tp_pred_components"],
                "fp_pred_components": stats["fp_pred_components"],
                "prompt_voxels": stats["prompt_voxels"],
                "gt_voxels": stats["gt_voxels"],
            })
            component_details[case_id] = {"pred_components": stats["pred_components_detail"], "gt_components": stats["gt_components_detail"]}
    _write_csv(report_dir / "case_proposal_metrics.csv", rows)
    (report_dir / "component_details.json").write_text(json.dumps(component_details, indent=2))
    print(f"Saved coarse predictions to {out_dir}")
    print(f"Saved component reports to {report_dir}")


if __name__ == "__main__":
    main()
