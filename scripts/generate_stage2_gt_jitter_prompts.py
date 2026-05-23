#!/usr/bin/env python
"""Generate GT-jitter prompts for Stage-2 supervised refinement.

MedSAM-style training commonly perturbs boxes to make the mask decoder robust
to imperfect prompts. For Stage 2, these prompts complement noisy coarse
proposals so the model also learns how to refine near-correct 3D boxes.
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
from scipy import ndimage
from tqdm import tqdm


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


def _clip_box(box: list[int], shape: tuple[int, int, int]) -> list[int]:
    d, h, w = shape
    z0, z1, y0, y1, x0, x1 = box
    z0, z1 = max(0, z0), min(d, z1)
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if z1 <= z0:
        z1 = min(d, z0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    return [z0, z1, y0, y1, x0, x1]


def _jitter_box(box: list[int], shape: tuple[int, int, int], rng: np.random.Generator, margin: tuple[int, int, int], jitter: tuple[int, int, int]) -> list[int]:
    z0, z1, y0, y1, x0, x1 = box
    mz, my, mx = margin
    jz, jy, jx = jitter
    dz0, dz1 = rng.integers(-jz, jz + 1, size=2)
    dy0, dy1 = rng.integers(-jy, jy + 1, size=2)
    dx0, dx1 = rng.integers(-jx, jx + 1, size=2)
    return _clip_box(
        [
            int(z0 - mz + dz0),
            int(z1 + mz + dz1),
            int(y0 - my + dy0),
            int(y1 + my + dy1),
            int(x0 - mx + dx0),
            int(x1 + mx + dx1),
        ],
        shape,
    )


def _center(box: list[int]) -> list[float]:
    z0, z1, y0, y1, x0, x1 = box
    return [(z0 + z1 - 1) / 2.0, (y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0]


def _box_volume(box: list[int]) -> int:
    return max(box[1] - box[0], 0) * max(box[3] - box[2], 0) * max(box[5] - box[4], 0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No GT-jitter prompt rows to write: {path}")
    fieldnames = [
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="../picai_preprocessing_project/data/processed/picai_profound_prompt_v2")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jitter-per-lesion", type=int, default=3)
    parser.add_argument("--margin-zyx", default="3,8,8")
    parser.add_argument("--jitter-zyx", default="2,6,6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    margin = tuple(int(v.strip()) for v in args.margin_zyx.split(","))
    jitter = tuple(int(v.strip()) for v in args.jitter_zyx.split(","))
    if len(margin) != 3 or len(jitter) != 3:
        raise ValueError("margin-zyx and jitter-zyx must each contain three integers")

    rng = np.random.default_rng(args.seed)
    processed_root = Path(args.processed_root)
    rows = []
    for case_id in tqdm(_read_split(Path(args.split)), desc="Generate GT-jitter prompts", dynamic_ncols=True):
        label = _load_label(processed_root, case_id)
        cc, n_gt = ndimage.label(label, structure=np.ones((3, 3, 3), dtype=np.uint8))
        rank = 1
        for gt_id in range(1, n_gt + 1):
            lesion = cc == gt_id
            base_box = _bbox(lesion)
            if base_box is None:
                continue
            for jitter_id in range(args.jitter_per_lesion):
                box = _jitter_box(base_box, label.shape, rng, margin=margin, jitter=jitter)
                center = _center(box)
                rows.append(
                    {
                        "case_id": case_id,
                        "proposal_rank": rank,
                        "component_id": int(gt_id * 1000 + jitter_id),
                        "z0": box[0],
                        "z1": box[1],
                        "y0": box[2],
                        "y1": box[3],
                        "x0": box[4],
                        "x1": box[5],
                        "center_z": center[0],
                        "center_y": center[1],
                        "center_x": center[2],
                        "component_voxels": int(lesion.sum()),
                        "box_volume": _box_volume(box),
                        "max_probability": 1.0,
                        "mean_probability": 1.0,
                        "source_threshold": "gt",
                        "prompt_type": "gt_jitter",
                        "overlaps_gt": True,
                    }
                )
                rank += 1

    output = Path(args.output)
    _write_csv(output, rows)
    (output.with_suffix(".settings.json")).write_text(json.dumps(vars(args), indent=2))
    print(f"Saved {len(rows)} GT-jitter prompts to {output}")


if __name__ == "__main__":
    main()
