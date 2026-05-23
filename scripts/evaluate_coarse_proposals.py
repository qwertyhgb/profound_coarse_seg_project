#!/usr/bin/env python
"""Evaluate coarse connected-component proposals for Stage-2 prompt generation."""
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


def _component_box(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def _box_volume(box: tuple[int, int, int, int, int, int]) -> int:
    z0, z1, y0, y1, x0, x1 = box
    return int((z1 - z0) * (y1 - y0) * (x1 - x0))


def _safe_case_id(value) -> str:
    if isinstance(value, list):
        value = value[0]
    return str(value)


def analyze_case(prob: np.ndarray, label: np.ndarray, threshold: float, min_component_size: int) -> dict:
    """Analyze 3D proposal components for one case."""
    prob3 = prob[0] if prob.ndim == 4 else prob
    gt3 = (label[0] if label.ndim == 4 else label) > 0.5
    pred3 = prob3 >= float(threshold)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_cc, n_gt = ndimage.label(gt3, structure=structure)
    pred_cc, n_pred = ndimage.label(pred3, structure=structure)

    pred_components = []
    valid_pred_mask = np.zeros_like(pred3, dtype=bool)
    for pred_id in range(1, n_pred + 1):
        component = pred_cc == pred_id
        voxels = int(component.sum())
        if voxels < int(min_component_size):
            continue
        box = _component_box(component)
        if box is None:
            continue
        overlaps_gt = bool(np.logical_and(component, gt3).any())
        valid_pred_mask |= component
        pred_components.append(
            {
                "component_id": int(pred_id),
                "voxels": voxels,
                "box_zyxzyx": box,
                "box_volume": _box_volume(box),
                "max_probability": float(prob3[component].max()),
                "mean_probability": float(prob3[component].mean()),
                "overlaps_gt": overlaps_gt,
            }
        )

    hit_gt = 0
    gt_components = []
    for gt_id in range(1, n_gt + 1):
        lesion = gt_cc == gt_id
        hit = bool(np.logical_and(valid_pred_mask, lesion).any())
        hit_gt += int(hit)
        box = _component_box(lesion)
        gt_components.append(
            {
                "gt_component_id": int(gt_id),
                "voxels": int(lesion.sum()),
                "box_zyxzyx": box,
                "hit": hit,
            }
        )

    tp_pred = sum(1 for comp in pred_components if comp["overlaps_gt"])
    fp_pred = len(pred_components) - tp_pred
    return {
        "total_gt_lesions": int(n_gt),
        "hit_gt_lesions": int(hit_gt),
        "lesion_recall": float(hit_gt / n_gt) if n_gt > 0 else 0.0,
        "pred_components": int(len(pred_components)),
        "tp_pred_components": int(tp_pred),
        "fp_pred_components": int(fp_pred),
        "component_precision": float(tp_pred / len(pred_components)) if pred_components else 0.0,
        "prompt_voxels": int(valid_pred_mask.sum()),
        "gt_voxels": int(gt3.sum()),
        "pred_components_detail": pred_components,
        "gt_components_detail": gt_components,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: dict, recommended_checkpoint: str, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Coarse Proposal Evaluation",
        "",
        f"- Checkpoint: `{recommended_checkpoint}`",
        f"- Prompt threshold: {threshold:.2f}",
        f"- Cases: {summary['cases']}",
        f"- Positive cases: {summary['positive_cases']}",
        f"- Lesion recall: {summary['lesion_recall']:.4f} ({summary['hit_gt_lesions']}/{summary['total_gt_lesions']})",
        f"- Candidate components per case: {summary['pred_components_per_case']:.4f}",
        f"- FP components per case: {summary['fp_components_per_case']:.4f}",
        f"- Component precision: {summary['component_precision']:.4f}",
        f"- Mean prompt voxels per case: {summary['mean_prompt_voxels']:.2f}",
        "",
        "For Stage 2, this report checks whether the low-threshold coarse mask keeps GT lesions while producing a manageable number of candidate components.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer_single_case.yaml")
    parser.add_argument("--split", default="data/splits/5fold/fold_0/val.txt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Prompt threshold. Defaults to inference.prompt_generation_threshold")
    parser.add_argument("--output-dir", default=None, help="Default: <run-dir>/inference/proposal_reports")
    parser.add_argument("--min-component-size", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--save-component-json", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inf = cfg["inference"]
    checkpoint = args.checkpoint or inf.get("checkpoint_path")
    if not checkpoint:
        raise ValueError("Missing checkpoint. Pass --checkpoint or set inference.checkpoint_path.")
    threshold = float(args.threshold if args.threshold is not None else inf.get("prompt_generation_threshold", 0.15))
    min_component_size = int(args.min_component_size if args.min_component_size is not None else cfg.get("metrics", {}).get("min_pred_component_size", 2))

    ckpt_path = Path(checkpoint)
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else Path("outputs")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "inference" / "proposal_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    component_payload = {}
    totals = {
        "cases": 0,
        "positive_cases": 0,
        "total_gt_lesions": 0,
        "hit_gt_lesions": 0,
        "pred_components": 0,
        "tp_pred_components": 0,
        "fp_pred_components": 0,
        "prompt_voxels": 0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Coarse proposals", dynamic_ncols=True):
            image = batch["image"].to(device)
            label = batch["label"].cpu().numpy()[0]
            case_id = _safe_case_id(batch["case_id"])
            logits = predictor.predict(image)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
            stats = analyze_case(prob, label, threshold=threshold, min_component_size=min_component_size)

            totals["cases"] += 1
            totals["positive_cases"] += int(stats["total_gt_lesions"] > 0)
            for key in ["total_gt_lesions", "hit_gt_lesions", "pred_components", "tp_pred_components", "fp_pred_components", "prompt_voxels"]:
                totals[key] += int(stats[key])

            rows.append(
                {
                    "case_id": case_id,
                    "total_gt_lesions": stats["total_gt_lesions"],
                    "hit_gt_lesions": stats["hit_gt_lesions"],
                    "lesion_recall": stats["lesion_recall"],
                    "pred_components": stats["pred_components"],
                    "tp_pred_components": stats["tp_pred_components"],
                    "fp_pred_components": stats["fp_pred_components"],
                    "component_precision": stats["component_precision"],
                    "prompt_voxels": stats["prompt_voxels"],
                    "gt_voxels": stats["gt_voxels"],
                }
            )
            if args.save_component_json:
                component_payload[case_id] = {
                    "pred_components": stats["pred_components_detail"],
                    "gt_components": stats["gt_components_detail"],
                }

    summary = {
        **totals,
        "lesion_recall": totals["hit_gt_lesions"] / totals["total_gt_lesions"] if totals["total_gt_lesions"] else 0.0,
        "pred_components_per_case": totals["pred_components"] / max(totals["cases"], 1),
        "fp_components_per_case": totals["fp_pred_components"] / max(totals["cases"], 1),
        "component_precision": totals["tp_pred_components"] / totals["pred_components"] if totals["pred_components"] else 0.0,
        "mean_prompt_voxels": totals["prompt_voxels"] / max(totals["cases"], 1),
    }

    _write_csv(output_dir / "case_proposal_metrics.csv", rows)
    _write_summary(output_dir / "proposal_summary.md", summary, checkpoint, threshold)
    if args.save_component_json:
        (output_dir / "component_details.json").write_text(json.dumps(component_payload, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved proposal report to {output_dir}")


if __name__ == "__main__":
    main()
