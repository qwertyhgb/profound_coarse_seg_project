#!/usr/bin/env python
"""Generate Stage-2 coarse prompts from multiple probability thresholds.

PCaSAM-style automatic prompting converts a coarse segmentation mask into box
prompts. A single high threshold can miss weak prostate lesions, so this script
extracts candidates at several thresholds, deduplicates overlapping boxes, and
exports a prompt CSV for Stage-2 training/inference.
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
from scipy import ndimage
from tqdm import tqdm


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _resolve_npz(root: Path, case_id: str) -> Path:
    direct = root / f"{case_id}.npz"
    if direct.is_file():
        return direct
    matches = list(root.rglob(f"{case_id}.npz"))
    if not matches:
        raise FileNotFoundError(f"Could not find {case_id}.npz under {root}")
    return matches[0]


def _resolve_coarse(root: Path, case_id: str) -> Path:
    for name in (f"{case_id}_coarse_pred.npz", f"{case_id}.npz"):
        direct = root / name
        if direct.is_file():
            return direct
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find coarse prediction for {case_id} under {root}")


def _load_probability(root: Path, case_id: str) -> np.ndarray:
    with np.load(_resolve_coarse(root, case_id), allow_pickle=False) as data:
        if "probability" not in data.files:
            raise KeyError(f"Coarse prediction for {case_id} has no probability array")
        prob = data["probability"].astype(np.float32)
    if prob.ndim == 4:
        prob = prob[0]
    if prob.ndim != 3:
        raise ValueError(f"Expected probability [D,H,W] for {case_id}, got {prob.shape}")
    return np.clip(prob, 0.0, 1.0)


def _load_label(root: Path, case_id: str) -> np.ndarray:
    with np.load(_resolve_npz(root, case_id), allow_pickle=False) as data:
        label = data["label"].astype(np.float32)
    if label.ndim == 4:
        label = label[0]
    return label > 0.5


def _bbox(mask: np.ndarray) -> list[int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)]


def _box_volume(box: list[int]) -> int:
    return max(box[1] - box[0], 0) * max(box[3] - box[2], 0) * max(box[5] - box[4], 0)


def _box_iou(a: list[int], b: list[int]) -> float:
    iz0, iz1 = max(a[0], b[0]), min(a[1], b[1])
    iy0, iy1 = max(a[2], b[2]), min(a[3], b[3])
    ix0, ix1 = max(a[4], b[4]), min(a[5], b[5])
    inter = max(iz1 - iz0, 0) * max(iy1 - iy0, 0) * max(ix1 - ix0, 0)
    union = _box_volume(a) + _box_volume(b) - inter
    return inter / max(union, 1e-8)


def _center_from_mask(mask: np.ndarray, prob: np.ndarray) -> list[float]:
    coords = np.argwhere(mask)
    weights = prob[mask].astype(np.float64)
    if float(weights.sum()) <= 1e-8:
        center = coords.mean(axis=0)
    else:
        center = (coords * weights[:, None]).sum(axis=0) / weights.sum()
    return [float(v) for v in center]


def _component_rows(case_id: str, prob: np.ndarray, threshold: float, min_size: int, gt: np.ndarray | None) -> list[dict[str, Any]]:
    pred = prob >= threshold
    labeled, n_pred = ndimage.label(pred, structure=np.ones((3, 3, 3), dtype=np.uint8))
    rows = []
    for component_id in range(1, n_pred + 1):
        mask = labeled == component_id
        voxels = int(mask.sum())
        if voxels < min_size:
            continue
        box = _bbox(mask)
        if box is None:
            continue
        center = _center_from_mask(mask, prob)
        overlaps_gt = bool(gt is not None and np.logical_and(mask, gt).any())
        rows.append(
            {
                "case_id": case_id,
                "component_id": component_id,
                "z0": box[0],
                "z1": box[1],
                "y0": box[2],
                "y1": box[3],
                "x0": box[4],
                "x1": box[5],
                "center_z": center[0],
                "center_y": center[1],
                "center_x": center[2],
                "component_voxels": voxels,
                "box_volume": _box_volume(box),
                "max_probability": float(prob[mask].max()),
                "mean_probability": float(prob[mask].mean()),
                "source_threshold": float(threshold),
                "prompt_type": "coarse_multithreshold",
                "overlaps_gt": overlaps_gt,
            }
        )
    return rows


def _rank(row: dict[str, Any], rank_by: str) -> float:
    if rank_by == "max_probability":
        return float(row["max_probability"])
    if rank_by == "mean_probability":
        return float(row["mean_probability"])
    if rank_by == "component_volume":
        return float(row["component_voxels"])
    if rank_by == "maxprob_x_volume":
        return float(row["max_probability"]) * float(row["component_voxels"])
    raise ValueError(f"Unsupported rank_by: {rank_by}")


def _deduplicate(rows: list[dict[str, Any]], iou_threshold: float, rank_by: str, top_k: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda r: _rank(r, rank_by), reverse=True)
    kept: list[dict[str, Any]] = []
    for row in rows:
        box = [int(row[k]) for k in ["z0", "z1", "y0", "y1", "x0", "x1"]]
        duplicate = False
        for prev in kept:
            prev_box = [int(prev[k]) for k in ["z0", "z1", "y0", "y1", "x0", "x1"]]
            if _box_iou(box, prev_box) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
        if top_k > 0 and len(kept) >= top_k:
            break
    for rank, row in enumerate(kept, start=1):
        row["proposal_rank"] = rank
    return kept


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    preferred = [
        "case_id",
        "proposal_rank",
        "component_id",
        "z0",
        "z1",
        "y0",
        "y1",
        "x0",
        "x1",
        "center_z",
        "center_y",
        "center_x",
        "component_voxels",
        "box_volume",
        "max_probability",
        "mean_probability",
        "source_threshold",
        "prompt_type",
        "overlaps_gt",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="../picai_preprocessing_project/data/processed/picai_profound_prompt_v2")
    parser.add_argument("--coarse-pred-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25")
    parser.add_argument("--min-component-sizes", default="5,10,20,50")
    parser.add_argument("--top-k-per-case", type=int, default=8)
    parser.add_argument("--dedup-box-iou", type=float, default=0.20)
    parser.add_argument("--rank-by", default="max_probability")
    parser.add_argument("--include-gt-hit", action="store_true")
    args = parser.parse_args()

    thresholds = _parse_floats(args.thresholds)
    min_sizes = _parse_ints(args.min_component_sizes)
    if len(min_sizes) == 1:
        min_sizes = min_sizes * len(thresholds)
    if len(min_sizes) != len(thresholds):
        raise ValueError("--min-component-sizes must have length 1 or match --thresholds")

    processed_root = Path(args.processed_root)
    coarse_root = Path(args.coarse_pred_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_ids = _read_split(Path(args.split))

    all_rows = []
    summary_rows = []
    for case_id in tqdm(case_ids, desc="Generate multi-threshold prompts", dynamic_ncols=True):
        prob = _load_probability(coarse_root, case_id)
        gt = _load_label(processed_root, case_id) if args.include_gt_hit else None
        raw = []
        for threshold, min_size in zip(thresholds, min_sizes):
            raw.extend(_component_rows(case_id, prob, threshold, min_size, gt))
        kept = _deduplicate(raw, iou_threshold=args.dedup_box_iou, rank_by=args.rank_by, top_k=args.top_k_per_case)
        all_rows.extend(kept)
        summary_rows.append(
            {
                "case_id": case_id,
                "raw_components": len(raw),
                "kept_prompts": len(kept),
                "gt_prompts": sum(1 for row in kept if str(row.get("overlaps_gt")).lower() == "true"),
                "gt_voxels": int(gt.sum()) if gt is not None else "",
            }
        )

    _write_csv(output_dir / "coarse_prompts_multithreshold.csv", all_rows)
    _write_csv(output_dir / "prompt_generation_summary.csv", summary_rows)
    (output_dir / "settings.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Saved {len(all_rows)} prompts to {output_dir / 'coarse_prompts_multithreshold.csv'}")


if __name__ == "__main__":
    main()
