"""Stage-2 SAM-Med3D dataset.

Reuses the proposal CSVs from existing Stage 2 pipeline, but:
- Crops a generous region around the proposal at native resolution
- Resizes the crop to (128, 128, 128) cube for SAM-Med3D
- Computes point prompt in resized coordinates
- Optionally jitters the prompt point for training robustness

Key difference from Stage2PromptDataset:
- Output spatial size is fixed at (128,128,128) regardless of native shape
- No box_prior or point_prior dense maps (SAM-Med3D uses sparse point prompts)
- Provides point_coords in (z,y,x) within the 128^3 cube
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Stage2SAMMed3DDataset(Dataset):
    """Proposal-centric crop dataset for SAM-Med3D fine-tuning.

    Each sample:
        image       : [3, 128, 128, 128] normalized mpMRI
        label       : [1, 128, 128, 128] binary GT
        point_coords: [1, 3] (z, y, x) in 128^3 coords (component centroid)
        point_label : [1] always 1 (positive prompt)
        objectness  : float scalar (does proposal overlap GT?)
        case_id, proposal_rank, component_id, etc.
    """

    def __init__(
        self,
        processed_root: str | Path,
        prompt_csv: str | Path,
        crop_margin_ratio: float = 1.5,
        target_size: Sequence[int] = (128, 128, 128),
        normalize: str = "channelwise_nonzero",
        max_prompts: int | None = None,
        positive_only: bool = False,
        negative_ratio: float | None = None,
        point_jitter_voxels: int = 0,
        seed: int = 42,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.prompt_csv = Path(prompt_csv)
        self.crop_margin_ratio = float(crop_margin_ratio)
        self.target_size = tuple(int(v) for v in target_size)
        self.normalize = str(normalize)
        self.positive_only = bool(positive_only)
        self.negative_ratio = negative_ratio
        self.point_jitter_voxels = int(point_jitter_voxels)
        self.rng = np.random.default_rng(seed)

        if not self.processed_root.is_dir():
            raise FileNotFoundError(f"Processed root not found: {self.processed_root}")
        if not self.prompt_csv.is_file():
            raise FileNotFoundError(f"Prompt CSV not found: {self.prompt_csv}")

        with self.prompt_csv.open("r", newline="") as f:
            self.prompts = list(csv.DictReader(f))
        self.prompts = self._filter_prompt_rows(self.prompts)
        if max_prompts is not None:
            self.prompts = self.prompts[: int(max_prompts)]
        if not self.prompts:
            raise ValueError(f"No prompts after filtering: {self.prompt_csv}")
        self._npz_cache: dict[str, Path] = {}

    @staticmethod
    def _row_overlaps_gt(row: dict[str, str]) -> bool:
        return str(row.get("overlaps_gt", "False")).strip().lower() in {"1", "true", "yes", "y"}

    def _filter_prompt_rows(self, rows: list[dict]) -> list[dict]:
        if not (self.positive_only or self.negative_ratio is not None):
            return rows
        if rows and "overlaps_gt" not in rows[0]:
            raise ValueError(
                f"{self.prompt_csv} has no overlaps_gt column. Need GT-aware prompt CSV."
            )
        positives = [r for r in rows if self._row_overlaps_gt(r)]
        negatives = [r for r in rows if not self._row_overlaps_gt(r)]
        if self.positive_only:
            return positives
        if self.negative_ratio is None:
            return rows
        max_neg = int(round(len(positives) * float(self.negative_ratio)))
        if max_neg <= 0:
            selected = list(positives)
        else:
            negatives_shuffled = list(negatives)
            self.rng.shuffle(negatives_shuffled)
            selected = list(positives) + negatives_shuffled[:max_neg]
        self.rng.shuffle(selected)
        return selected

    def _resolve_npz(self, case_id: str) -> Path:
        if case_id in self._npz_cache:
            return self._npz_cache[case_id]
        direct = self.processed_root / f"{case_id}.npz"
        if direct.is_file():
            self._npz_cache[case_id] = direct
            return direct
        matches = list(self.processed_root.rglob(f"{case_id}.npz"))
        if not matches:
            raise FileNotFoundError(f"NPZ for {case_id} not found under {self.processed_root}")
        self._npz_cache[case_id] = matches[0]
        return matches[0]

    @staticmethod
    def _normalize_channelwise_nonzero(image: np.ndarray) -> np.ndarray:
        """Per-channel zero-mean unit-variance over non-zero voxels."""
        out = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            ch = image[c]
            mask = ch != 0
            if mask.sum() == 0:
                out[c] = ch
                continue
            mean = ch[mask].mean()
            std = ch[mask].std()
            if std < 1e-6:
                out[c] = ch - mean
            else:
                out[c] = (ch - mean) / std
        return out

    @staticmethod
    def _expand_bbox_isotropic(
        bbox_zyxzyx: Sequence[int],
        shape: Sequence[int],
        margin_ratio: float,
        min_extent: int = 8,
    ) -> tuple[int, int, int, int, int, int]:
        """Expand the bbox by margin_ratio*extent on each side, clipped to shape."""
        z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox_zyxzyx]
        D, H, W = [int(v) for v in shape]
        ez = max(z1 - z0, min_extent)
        ey = max(y1 - y0, min_extent)
        ex = max(x1 - x0, min_extent)
        mz = int(ez * margin_ratio)
        my = int(ey * margin_ratio)
        mx = int(ex * margin_ratio)
        return (
            max(0, z0 - mz), min(D, z1 + mz),
            max(0, y0 - my), min(H, y1 + my),
            max(0, x0 - mx), min(W, x1 + mx),
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.prompts[index]
        case_id = str(row["case_id"])
        npz_path = self._resolve_npz(case_id)
        with np.load(npz_path, allow_pickle=False) as d:
            image = d["image"].astype(np.float32)  # [3, D, H, W]
            label = d["label"].astype(np.float32)  # [1, D, H, W]
            metadata = json.loads(str(d["metadata_json"])) if "metadata_json" in d.files else {}

        if self.normalize == "channelwise_nonzero":
            image = self._normalize_channelwise_nonzero(image)

        # Parse bbox and center
        bbox_native = [int(float(row[k])) for k in ["z0", "z1", "y0", "y1", "x0", "x1"]]
        center_native = [
            float(row.get("center_z", (bbox_native[0] + bbox_native[1] - 1) / 2.0)),
            float(row.get("center_y", (bbox_native[2] + bbox_native[3] - 1) / 2.0)),
            float(row.get("center_x", (bbox_native[4] + bbox_native[5] - 1) / 2.0)),
        ]

        # Expand bbox for crop region
        spatial_shape = image.shape[1:]
        crop_box = self._expand_bbox_isotropic(
            bbox_native, spatial_shape, self.crop_margin_ratio
        )
        z0, z1, y0, y1, x0, x1 = crop_box

        # Crop both image and label
        img_crop = image[:, z0:z1, y0:y1, x0:x1]
        lbl_crop = label[:, z0:z1, y0:y1, x0:x1]

        # Resize to target_size (128^3)
        img_t = torch.from_numpy(img_crop).float().unsqueeze(0)  # [1,3,d,h,w]
        lbl_t = torch.from_numpy(lbl_crop).float().unsqueeze(0)  # [1,1,d,h,w]
        img_resized = F.interpolate(img_t, size=self.target_size, mode="trilinear", align_corners=False)[0]
        lbl_resized = F.interpolate(lbl_t, size=self.target_size, mode="trilinear", align_corners=False)[0]
        lbl_resized = (lbl_resized > 0.5).float()

        # Compute point coords in the resized 128^3 cube
        # native center → relative to crop origin → scale to target size
        crop_size = (z1 - z0, y1 - y0, x1 - x0)
        rel_z = (center_native[0] - z0) / max(crop_size[0], 1) * self.target_size[0]
        rel_y = (center_native[1] - y0) / max(crop_size[1], 1) * self.target_size[1]
        rel_x = (center_native[2] - x0) / max(crop_size[2], 1) * self.target_size[2]

        # Optional jitter (training only)
        if self.point_jitter_voxels > 0:
            j = self.point_jitter_voxels
            rel_z += float(self.rng.uniform(-j, j))
            rel_y += float(self.rng.uniform(-j, j))
            rel_x += float(self.rng.uniform(-j, j))

        rel_z = float(np.clip(rel_z, 0, self.target_size[0] - 1))
        rel_y = float(np.clip(rel_y, 0, self.target_size[1] - 1))
        rel_x = float(np.clip(rel_x, 0, self.target_size[2] - 1))

        objectness = 1.0 if self._row_overlaps_gt(row) else 0.0

        return {
            "image": img_resized.contiguous(),                                    # [3, 128, 128, 128]
            "label": lbl_resized.contiguous(),                                    # [1, 128, 128, 128]
            "point_coords": torch.tensor([[rel_z, rel_y, rel_x]], dtype=torch.float32),  # [1, 3]
            "point_label": torch.tensor([1], dtype=torch.long),                   # [1]
            "objectness": torch.tensor(objectness, dtype=torch.float32),
            "case_id": case_id,
            "proposal_rank": int(float(row.get("proposal_rank", index + 1))),
            "component_id": int(float(row.get("component_id", index + 1))),
            "bbox_native_zyxzyx": torch.tensor(bbox_native, dtype=torch.long),
            "crop_box_zyxzyx": torch.tensor(crop_box, dtype=torch.long),
            "original_shape_zyx": torch.tensor(spatial_shape, dtype=torch.long),
            "metadata": metadata,
        }


def stage2_sam_med3d_collate_fn(batch: list[dict]) -> dict:
    """Collate function for Stage2SAMMed3DDataset."""
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "point_coords": torch.stack([b["point_coords"] for b in batch], dim=0),    # [B, 1, 3]
        "point_label": torch.stack([b["point_label"] for b in batch], dim=0),       # [B, 1]
        "objectness": torch.stack([b["objectness"] for b in batch], dim=0),
        "case_id": [b["case_id"] for b in batch],
        "proposal_rank": torch.tensor([b["proposal_rank"] for b in batch], dtype=torch.long),
        "component_id": torch.tensor([b["component_id"] for b in batch], dtype=torch.long),
        "bbox_native_zyxzyx": torch.stack([b["bbox_native_zyxzyx"] for b in batch], dim=0),
        "crop_box_zyxzyx": torch.stack([b["crop_box_zyxzyx"] for b in batch], dim=0),
        "original_shape_zyx": torch.stack([b["original_shape_zyx"] for b in batch], dim=0),
        "metadata": [b.get("metadata", {}) for b in batch],
    }
