"""Dataset for PCaSAM-3D-ProFound end-to-end training.

Unlike the Stage-2 proposal datasets, this dataset loads full volumes and
crops them to a fixed patch size for training. The model itself handles
coarse segmentation and prompt generation internally.

For training: random patches with lesion-aware sampling (same as Stage 1)
For validation: resize to 128^3 cube (SAM-Med3D compatible) or use sliding window
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PCaSAM3DDataset(Dataset):
    """Dataset for PCaSAM-3D-ProFound end-to-end training.

    Each sample returns a fixed-size 3D patch (training) or resized volume (val/test).
    The model handles coarse segmentation and prompt generation internally.
    """

    def __init__(
        self,
        processed_root: str | Path,
        split_file: str | Path | None = None,
        case_list: Sequence[str] | None = None,
        mode: str = "train",
        patch_size: Sequence[int] = (128, 128, 128),
        use_lesion_aware_sampling: bool = True,
        pos_patch_ratio: float = 0.7,
        positive_case_ratio: float = 0.6,
        normalize: str = "channelwise_nonzero",
        gland_aware_negative_sampling: bool = False,
        gland_negative_prob: float = 0.8,
        augmentation: dict | None = None,
        max_cases: int | None = None,
        seed: int = 42,
    ) -> None:
        self.processed_root = Path(processed_root)
        if not self.processed_root.is_dir():
            raise FileNotFoundError(f"Processed root not found: {self.processed_root}")

        self.mode = mode
        self.patch_size = tuple(int(v) for v in patch_size)
        self.use_lesion_aware_sampling = use_lesion_aware_sampling
        self.pos_patch_ratio = float(pos_patch_ratio)
        self.positive_case_ratio = float(positive_case_ratio)
        self.normalize = normalize
        self.gland_aware_negative_sampling = bool(gland_aware_negative_sampling)
        self.gland_negative_prob = float(gland_negative_prob)
        self.augmentation = augmentation or {}
        self.rng = np.random.default_rng(seed)

        entries = self._load_entries(split_file, case_list)
        if max_cases is not None:
            entries = entries[: int(max_cases)]
        self.samples = [self._resolve_npz(e) for e in entries]
        if not self.samples:
            raise ValueError("No NPZ samples found.")

        self._positive_indices = self._scan_positive_indices()
        self._negative_indices = [
            i for i in range(len(self.samples)) if i not in set(self._positive_indices)
        ]

    def _load_entries(self, split_file, case_list) -> list[str]:
        if case_list is not None:
            return [str(x).strip() for x in case_list if str(x).strip()]
        if split_file is None:
            return [p.stem for p in sorted(self.processed_root.rglob("*.npz"))]
        split_file = Path(split_file)
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    def _resolve_npz(self, entry: str) -> Path:
        p = Path(entry)
        if p.is_file():
            return p
        direct = self.processed_root / f"{entry}.npz"
        if direct.is_file():
            return direct
        matches = list(self.processed_root.rglob(f"{entry}.npz"))
        if not matches:
            raise FileNotFoundError(f"NPZ for '{entry}' not found under {self.processed_root}")
        return matches[0]

    def _scan_positive_indices(self) -> list[int]:
        positives = []
        for i, path in enumerate(self.samples):
            with np.load(path, allow_pickle=False) as data:
                if int(data["label"].sum()) > 0:
                    positives.append(i)
        return positives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Balanced case sampling for training
        if self.mode == "train" and self._positive_indices and self._negative_indices:
            if self.rng.random() < self.positive_case_ratio:
                index = int(self.rng.choice(self._positive_indices))
            else:
                index = int(self.rng.choice(self._negative_indices))

        path = self.samples[index]
        with np.load(path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)  # [3, D, H, W]
            label = data["label"].astype(np.float32)  # [1, D, H, W]
            gland_mask = (
                data["gland_mask"].astype(np.float32)
                if "gland_mask" in data.files
                else np.ones_like(label, dtype=np.float32)
            )
            boundary_mask = (
                data["boundary_uncertainty_mask"].astype(np.float32)
                if "boundary_uncertainty_mask" in data.files
                else np.zeros_like(label, dtype=np.float32)
            )
            case_id = str(data["case_id"])
            metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}

        if gland_mask.ndim == 3:
            gland_mask = gland_mask[None]
        if boundary_mask.ndim == 3:
            boundary_mask = boundary_mask[None]

        # The PI-CAI preprocessing pipeline can already percentile-clip and z-score
        # each modality. Keep normalization configurable so experiments can avoid
        # erasing ADC calibration with a second per-sample z-score pass.
        if self.normalize == "channelwise_nonzero":
            image = self._normalize_channelwise_nonzero(image)
        elif self.normalize in (None, "", "none", "preprocessed"):
            pass
        else:
            raise ValueError(f"Unsupported normalization mode: {self.normalize}")

        if self.mode == "train":
            image, label, gland_mask, boundary_mask = self._sample_and_resize_patch(
                image, label, gland_mask, boundary_mask
            )
            image, label, gland_mask, boundary_mask = self._apply_augmentation(
                image, label, gland_mask, boundary_mask
            )
        else:
            # Validation: resize entire volume to patch_size (128^3)
            image, label, gland_mask, boundary_mask = self._resize_volume(
                image, label, gland_mask, boundary_mask
            )

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "gland_mask": torch.from_numpy(np.ascontiguousarray(gland_mask)),
            "boundary_uncertainty_mask": torch.from_numpy(np.ascontiguousarray(boundary_mask)),
            "case_id": case_id,
            "metadata": metadata,
            "npz_path": str(path),
        }

    def _apply_augmentation(
        self,
        image: np.ndarray,
        label: np.ndarray,
        gland_mask: np.ndarray,
        boundary_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply 3D data augmentation for training regularization.

        Implements random flips, intensity perturbation, and Gaussian noise.
        Reference: arXiv 2505.04097 shows augmentation reduces overfitting ~2.7%.
        """
        if not self.augmentation.get("enabled", False):
            return image, label, gland_mask, boundary_mask

        # Random flips along each axis
        flip_prob = float(self.augmentation.get("random_flip_prob", 0.5))
        flip_axes = self.augmentation.get("random_flip_axes", [0, 1, 2])
        for axis in flip_axes:
            if self.rng.random() < flip_prob:
                spatial_axis = axis + 1  # tensors are [C, D, H, W]
                image = np.flip(image, axis=spatial_axis).copy()
                label = np.flip(label, axis=spatial_axis).copy()
                gland_mask = np.flip(gland_mask, axis=spatial_axis).copy()
                boundary_mask = np.flip(boundary_mask, axis=spatial_axis).copy()

        # Random intensity shift and scale (per channel)
        shift_range = float(self.augmentation.get("intensity_shift_range", 0.0))
        scale_range = float(self.augmentation.get("intensity_scale_range", 0.0))
        if shift_range > 0 or scale_range > 0:
            for c in range(image.shape[0]):
                if shift_range > 0:
                    shift = self.rng.uniform(-shift_range, shift_range)
                    image[c] = image[c] + shift
                if scale_range > 0:
                    scale = self.rng.uniform(1.0 - scale_range, 1.0 + scale_range)
                    image[c] = image[c] * scale

        # Gaussian noise
        noise_std = float(self.augmentation.get("gaussian_noise_std", 0.0))
        if noise_std > 0:
            noise = self.rng.normal(0, noise_std, size=image.shape).astype(np.float32)
            image = image + noise

        return image, label, gland_mask, boundary_mask

    def _sample_and_resize_patch(
        self,
        image: np.ndarray,
        label: np.ndarray,
        gland_mask: np.ndarray,
        boundary_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a patch around a lesion or random location, then resize to patch_size."""
        volume_shape = image.shape[1:]  # (D, H, W)

        want_positive = (
            self.use_lesion_aware_sampling
            and label.sum() > 0
            and self.rng.random() < self.pos_patch_ratio
        )

        if want_positive:
            center = self._sample_lesion_center(label)
        else:
            center = self._sample_negative_center(volume_shape, gland_mask)

        crop_size = tuple(min(int(ps * 1.5), vs) for ps, vs in zip(self.patch_size, volume_shape))
        slices = self._compute_crop_slices(center, crop_size, volume_shape)

        img_crop = image[:, slices[0], slices[1], slices[2]]
        lbl_crop = label[:, slices[0], slices[1], slices[2]]
        gland_crop = gland_mask[:, slices[0], slices[1], slices[2]]
        boundary_crop = boundary_mask[:, slices[0], slices[1], slices[2]]

        img_t = torch.from_numpy(img_crop).float().unsqueeze(0)
        lbl_t = torch.from_numpy(lbl_crop).float().unsqueeze(0)
        gland_t = torch.from_numpy(gland_crop).float().unsqueeze(0)
        boundary_t = torch.from_numpy(boundary_crop).float().unsqueeze(0)

        img_resized = F.interpolate(img_t, size=self.patch_size, mode="trilinear", align_corners=False)[0]
        lbl_resized = F.interpolate(lbl_t, size=self.patch_size, mode="trilinear", align_corners=False)[0]
        lbl_resized = (lbl_resized > 0.5).float()
        gland_resized = F.interpolate(gland_t, size=self.patch_size, mode="nearest")[0]
        boundary_resized = F.interpolate(boundary_t, size=self.patch_size, mode="trilinear", align_corners=False)[0]
        boundary_resized = boundary_resized.clamp(0.0, 1.0)

        return img_resized.numpy(), lbl_resized.numpy(), gland_resized.numpy(), boundary_resized.numpy()

    def _resize_volume(
        self,
        image: np.ndarray,
        label: np.ndarray,
        gland_mask: np.ndarray,
        boundary_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resize entire volume to patch_size for validation."""
        img_t = torch.from_numpy(image).float().unsqueeze(0)
        lbl_t = torch.from_numpy(label).float().unsqueeze(0)
        gland_t = torch.from_numpy(gland_mask).float().unsqueeze(0)
        boundary_t = torch.from_numpy(boundary_mask).float().unsqueeze(0)

        img_resized = F.interpolate(img_t, size=self.patch_size, mode="trilinear", align_corners=False)[0]
        lbl_resized = F.interpolate(lbl_t, size=self.patch_size, mode="trilinear", align_corners=False)[0]
        lbl_resized = (lbl_resized > 0.5).float()
        gland_resized = F.interpolate(gland_t, size=self.patch_size, mode="nearest")[0]
        boundary_resized = F.interpolate(boundary_t, size=self.patch_size, mode="trilinear", align_corners=False)[0]
        boundary_resized = boundary_resized.clamp(0.0, 1.0)

        return img_resized.numpy(), lbl_resized.numpy(), gland_resized.numpy(), boundary_resized.numpy()

    def _sample_lesion_center(self, label: np.ndarray) -> tuple[int, int, int]:
        """Sample a random voxel from the lesion mask as crop center."""
        coords = np.argwhere(label[0] > 0)
        if len(coords) == 0:
            return self._sample_random_center(label.shape[1:])
        idx = self.rng.integers(len(coords))
        return tuple(int(v) for v in coords[idx])

    def _sample_random_center(self, shape: tuple[int, ...]) -> tuple[int, int, int]:
        """Sample a random center within the volume."""
        return tuple(int(self.rng.integers(0, max(s, 1))) for s in shape)

    def _sample_negative_center(
        self,
        shape: tuple[int, ...],
        gland_mask: np.ndarray | None = None,
    ) -> tuple[int, int, int]:
        """Sample a negative crop center, optionally biased into the prostate gland.

        Random whole-volume negatives are often dominated by easy background after
        gland-bbox cropping. Sampling inside the gland yields harder negatives and
        better matches the false-positive modes seen in PI-CAI lesion detection.
        """
        if (
            self.gland_aware_negative_sampling
            and gland_mask is not None
            and self.rng.random() < self.gland_negative_prob
        ):
            coords = np.argwhere(gland_mask[0] > 0.5)
            if len(coords) > 0:
                idx = self.rng.integers(len(coords))
                return tuple(int(v) for v in coords[idx])
        return self._sample_random_center(shape)

    @staticmethod
    def _compute_crop_slices(
        center: tuple[int, ...], crop_size: tuple[int, ...], volume_shape: tuple[int, ...]
    ) -> tuple[slice, ...]:
        """Compute crop slices centered at `center`, clipped to volume bounds."""
        slices = []
        for c, cs, vs in zip(center, crop_size, volume_shape):
            half = cs // 2
            start = max(0, min(c - half, vs - cs))
            end = min(vs, start + cs)
            slices.append(slice(start, end))
        return tuple(slices)

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


def pcasam3d_collate_fn(batch: list[dict]) -> dict:
    """Collate function for PCaSAM3DDataset."""
    out = {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "case_id": [b["case_id"] for b in batch],
        "metadata": [b.get("metadata", {}) for b in batch],
    }
    if "gland_mask" in batch[0]:
        out["gland_mask"] = torch.stack([b["gland_mask"] for b in batch], dim=0)
    if "boundary_uncertainty_mask" in batch[0]:
        out["boundary_uncertainty_mask"] = torch.stack(
            [b["boundary_uncertainty_mask"] for b in batch], dim=0
        )
    return out
