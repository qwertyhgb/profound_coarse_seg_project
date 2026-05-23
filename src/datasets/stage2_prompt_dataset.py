"""Stage-2 prompt-conditioned refinement dataset.

Each sample is one coarse proposal component. The dataset crops image, label,
coarse probability, and dense prompt priors around the proposal so a refinement
model can learn to accept true lesion proposals and reject false-positive ones.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import csv
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from .samplers import compute_patch_slices


class Stage2PromptDataset(Dataset):
    """Dataset for Stage-2 coarse-to-fine prompt-conditioned refinement."""

    def __init__(
        self,
        processed_root: str | Path,
        prompt_csv: str | Path,
        coarse_pred_root: str | Path,
        patch_size: Sequence[int] = (64, 128, 128),
        bbox_margin: Sequence[int] = (4, 12, 12),
        point_sigma: float = 3.0,
        max_prompts: int | None = None,
        allow_missing_coarse: bool = False,
        use_overlaps_gt_sampling: bool = False,
        positive_only: bool = False,
        negative_ratio: float | None = None,
        seed: int = 42,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.prompt_csv = Path(prompt_csv)
        self.coarse_pred_root = Path(coarse_pred_root)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.bbox_margin = tuple(int(v) for v in bbox_margin)
        self.point_sigma = float(point_sigma)
        self.allow_missing_coarse = bool(allow_missing_coarse)
        self.use_overlaps_gt_sampling = bool(use_overlaps_gt_sampling)
        self.positive_only = bool(positive_only)
        self.negative_ratio = negative_ratio
        self.rng = np.random.default_rng(seed)

        if not self.processed_root.is_dir():
            raise FileNotFoundError(f"Processed root not found: {self.processed_root}")
        if not self.prompt_csv.is_file():
            raise FileNotFoundError(f"Prompt CSV not found: {self.prompt_csv}")
        if not self.coarse_pred_root.is_dir() and not self.allow_missing_coarse:
            raise FileNotFoundError(f"Coarse prediction root not found: {self.coarse_pred_root}")

        with self.prompt_csv.open("r", newline="") as f:
            self.prompts = list(csv.DictReader(f))
        self.prompts = self._filter_prompt_rows(self.prompts)
        if max_prompts is not None:
            self.prompts = self.prompts[: int(max_prompts)]
        if not self.prompts:
            raise ValueError(f"No prompt rows found in {self.prompt_csv} after filtering")
        self._npz_cache: dict[str, Path] = {}
        self._coarse_cache: dict[str, Path] = {}


    def _filter_prompt_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Optionally rebalance true-positive and false-positive proposal rows.

        `overlaps_gt` is only available for supervised train/validation prompt
        files. It should never be used to make decisions at real test time.
        """
        if not (self.use_overlaps_gt_sampling or self.positive_only or self.negative_ratio is not None):
            return rows
        if rows and "overlaps_gt" not in rows[0]:
            raise ValueError(
                f"{self.prompt_csv} has no overlaps_gt column. Regenerate prompts with "
                "scripts/sweep_proposal_postprocess.py --include-gt-hit-in-prompts for Stage-2 supervised training."
            )
        positives = [r for r in rows if self._row_overlaps_gt(r)]
        negatives = [r for r in rows if not self._row_overlaps_gt(r)]
        if self.positive_only:
            return positives
        if self.negative_ratio is None:
            return rows
        max_neg = int(round(len(positives) * float(self.negative_ratio)))
        if max_neg <= 0:
            selected = positives
        else:
            selected_neg = list(negatives)
            self.rng.shuffle(selected_neg)
            selected = positives + selected_neg[: min(max_neg, len(selected_neg))]
        self.rng.shuffle(selected)
        return selected

    @staticmethod
    def _row_overlaps_gt(row: dict[str, str]) -> bool:
        return str(row.get("overlaps_gt", "False")).strip().lower() in {"1", "true", "yes", "y"}

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.prompts[index]
        case_id = str(row["case_id"])
        image, label, metadata = self._load_case(case_id)
        coarse_prob = self._load_coarse(case_id, image.shape[1:])

        bbox = self._bbox_from_row(row)
        center = self._center_from_row(row, bbox)
        bbox = self._expand_bbox(bbox, image.shape[1:])
        crop_slices = self._crop_slices(center, image.shape[1:])

        box_prior = self._box_prior(bbox, image.shape[1:])
        point_prior = self._point_prior(center, image.shape[1:])

        image_patch = self._crop_and_pad(image, crop_slices, pad_value=0.0)
        label_patch = self._crop_and_pad(label, crop_slices, pad_value=0.0)
        coarse_patch = self._crop_and_pad(coarse_prob, crop_slices, pad_value=0.0)
        box_patch = self._crop_and_pad(box_prior, crop_slices, pad_value=0.0)
        point_patch = self._crop_and_pad(point_prior, crop_slices, pad_value=0.0)
        patch_info = self._patch_mapping(crop_slices, image.shape[1:])
        objectness = 1.0 if self._row_overlaps_gt(row) else 0.0

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_patch.astype(np.float32))),
            "label": torch.from_numpy(np.ascontiguousarray(label_patch.astype(np.float32))),
            "coarse_prob": torch.from_numpy(np.ascontiguousarray(coarse_patch.astype(np.float32))),
            "box_prior": torch.from_numpy(np.ascontiguousarray(box_patch.astype(np.float32))),
            "point_prior": torch.from_numpy(np.ascontiguousarray(point_patch.astype(np.float32))),
            "case_id": case_id,
            "proposal_rank": int(float(row.get("proposal_rank", index + 1))),
            "component_id": int(float(row.get("component_id", index + 1))),
            "bbox_zyxzyx": torch.tensor(bbox, dtype=torch.float32),
            "center_zyx": torch.tensor(center, dtype=torch.float32),
            "objectness_label": torch.tensor(objectness, dtype=torch.float32),
            "crop_start_zyx": torch.tensor(patch_info["crop_start_zyx"], dtype=torch.long),
            "crop_end_zyx": torch.tensor(patch_info["crop_end_zyx"], dtype=torch.long),
            "patch_valid_start_zyx": torch.tensor(patch_info["patch_valid_start_zyx"], dtype=torch.long),
            "patch_valid_end_zyx": torch.tensor(patch_info["patch_valid_end_zyx"], dtype=torch.long),
            "original_shape_zyx": torch.tensor(image.shape[1:], dtype=torch.long),
            "metadata": metadata,
        }

    def _resolve_case_path(self, case_id: str) -> Path:
        if case_id in self._npz_cache:
            return self._npz_cache[case_id]
        direct = self.processed_root / f"{case_id}.npz"
        if direct.is_file():
            self._npz_cache[case_id] = direct
            return direct
        matches = list(self.processed_root.rglob(f"{case_id}.npz"))
        if not matches:
            raise FileNotFoundError(f"Could not find NPZ for {case_id} under {self.processed_root}")
        self._npz_cache[case_id] = matches[0]
        return matches[0]

    def _resolve_coarse_path(self, case_id: str) -> Path | None:
        if case_id in self._coarse_cache:
            return self._coarse_cache[case_id]
        patterns = [f"{case_id}_coarse_pred.npz", f"{case_id}.npz"]
        for pattern in patterns:
            direct = self.coarse_pred_root / pattern
            if direct.is_file():
                self._coarse_cache[case_id] = direct
                return direct
            matches = list(self.coarse_pred_root.rglob(pattern)) if self.coarse_pred_root.is_dir() else []
            if matches:
                self._coarse_cache[case_id] = matches[0]
                return matches[0]
        if self.allow_missing_coarse:
            return None
        raise FileNotFoundError(f"Missing coarse prediction for {case_id} under {self.coarse_pred_root}")

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
        path = self._resolve_case_path(case_id)
        with np.load(path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            label = data["label"].astype(np.float32)
            metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
        return image, label, metadata

    def _load_coarse(self, case_id: str, spatial_shape: tuple[int, int, int]) -> np.ndarray:
        path = self._resolve_coarse_path(case_id)
        if path is None:
            return np.zeros((1, *spatial_shape), dtype=np.float32)
        with np.load(path, allow_pickle=False) as data:
            if "probability" not in data.files:
                raise KeyError(f"{path} missing 'probability' coarse map")
            prob = data["probability"].astype(np.float32)
        if prob.ndim == 3:
            prob = prob[None]
        if prob.shape[1:] != spatial_shape:
            raise ValueError(f"Coarse shape {prob.shape[1:]} differs from image shape {spatial_shape} for {case_id}")
        return prob

    @staticmethod
    def _bbox_from_row(row: dict[str, str]) -> list[int]:
        return [int(float(row[k])) for k in ["z0", "z1", "y0", "y1", "x0", "x1"]]

    @staticmethod
    def _center_from_row(row: dict[str, str], bbox: Sequence[int]) -> list[float]:
        if all(k in row and row[k] not in (None, "") for k in ["center_z", "center_y", "center_x"]):
            return [float(row["center_z"]), float(row["center_y"]), float(row["center_x"])]
        z0, z1, y0, y1, x0, x1 = bbox
        return [(z0 + z1 - 1) / 2.0, (y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0]

    def _expand_bbox(self, bbox: Sequence[int], shape: tuple[int, int, int]) -> list[int]:
        z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox]
        mz, my, mx = self.bbox_margin
        d, h, w = shape
        return [max(0, z0 - mz), min(d, z1 + mz), max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)]

    def _crop_slices(self, center: Sequence[float], shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
        center_int = tuple(int(round(float(v))) for v in center)
        return compute_patch_slices(center_int, self.patch_size, shape)

    def _box_prior(self, bbox: Sequence[int], shape: tuple[int, int, int]) -> np.ndarray:
        prior = np.zeros((1, *shape), dtype=np.float32)
        z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox]
        prior[:, z0:z1, y0:y1, x0:x1] = 1.0
        return prior

    def _point_prior(self, center: Sequence[float], shape: tuple[int, int, int]) -> np.ndarray:
        d, h, w = shape
        zc, yc, xc = [float(v) for v in center]
        z = np.arange(d, dtype=np.float32)[:, None, None]
        y = np.arange(h, dtype=np.float32)[None, :, None]
        x = np.arange(w, dtype=np.float32)[None, None, :]
        sigma2 = max(self.point_sigma ** 2, 1e-6)
        prior = np.exp(-((z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2) / (2.0 * sigma2))
        return prior[None].astype(np.float32)


    def _patch_mapping(self, crop_slices: tuple[slice, slice, slice], shape: tuple[int, int, int]) -> dict[str, list[int]]:
        """Return how a padded patch maps back to the original volume."""
        crop_start = [int(s.start) for s in crop_slices]
        crop_end = [int(s.stop) for s in crop_slices]
        valid_shape = [e - s for s, e in zip(crop_start, crop_end)]
        patch_valid_start = []
        patch_valid_end = []
        for got, target in zip(valid_shape, self.patch_size):
            total = max(int(target) - int(got), 0)
            before = total // 2
            patch_valid_start.append(before)
            patch_valid_end.append(before + int(got))
        return {
            "crop_start_zyx": crop_start,
            "crop_end_zyx": crop_end,
            "patch_valid_start_zyx": patch_valid_start,
            "patch_valid_end_zyx": patch_valid_end,
        }

    def _crop_and_pad(self, array: np.ndarray, crop_slices: tuple[slice, slice, slice], pad_value: float) -> np.ndarray:
        cropped = array[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        pads = []
        for got, target in zip(cropped.shape[1:], self.patch_size):
            total = max(int(target) - int(got), 0)
            pads.append((total // 2, total - total // 2))
        if any(a or b for a, b in pads):
            cropped = np.pad(cropped, [(0, 0), *pads], mode="constant", constant_values=pad_value)
        return cropped
