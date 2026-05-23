"""PI-CAI preprocessed NPZ dataset with lesion-aware 3D patch sampling."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Any
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .samplers import compute_patch_slices, sample_lesion_center, sample_random_center


class PICAINPZDataset(Dataset):
    """Dataset for PI-CAI `.npz` volumes.

    Training mode returns fixed-size patches. Validation/test mode returns whole
    cropped volumes and should be evaluated with full-volume or sliding-window inference.
    """

    def __init__(
        self,
        processed_root: str | Path,
        split_file: str | Path | None = None,
        case_list: Sequence[str] | None = None,
        mode: str = "train",
        train_patch_size: Sequence[int] | None = None,
        use_lesion_aware_sampling: bool = True,
        pos_patch_ratio: float = 0.7,
        positive_case_ratio: float = 0.6,
        max_cases: int | None = None,
        seed: int = 42,
    ) -> None:
        self.processed_root = Path(processed_root)
        if not self.processed_root.is_dir():
            raise FileNotFoundError(f"Processed root not found: {self.processed_root}")
        self.mode = mode
        self.train_patch_size = tuple(int(v) for v in train_patch_size) if train_patch_size else None
        self.use_lesion_aware_sampling = use_lesion_aware_sampling
        self.pos_patch_ratio = float(pos_patch_ratio)
        self.positive_case_ratio = float(positive_case_ratio)
        self.rng = np.random.default_rng(seed)

        entries = self._load_entries(split_file, case_list)
        if max_cases is not None:
            entries = entries[: int(max_cases)]
        self.samples = [self._resolve_npz(e) for e in entries]
        if not self.samples:
            raise ValueError("No NPZ samples found. Create splits or pass a non-empty case_list.")
        self._positive_indices = self._scan_positive_indices()
        self._negative_indices = [i for i in range(len(self.samples)) if i not in set(self._positive_indices)]

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
            raise FileNotFoundError(f"Could not find NPZ for case '{entry}' under {self.processed_root}")
        return matches[0]

    def _scan_positive_indices(self) -> list[int]:
        positives = []
        for i, path in enumerate(self.samples):
            with np.load(path, allow_pickle=False) as data:
                self._validate_npz(data, path)
                if int(data["label"].sum()) > 0:
                    positives.append(i)
        return positives

    @staticmethod
    def _validate_npz(data: Any, path: Path) -> None:
        for key in ("image", "label", "case_id"):
            if key not in data.files:
                raise KeyError(f"{path} missing required field '{key}'")
        if data["image"].ndim != 4 or data["image"].shape[0] != 3:
            raise ValueError(f"{path} image must have shape [3,D,H,W], got {data['image'].shape}")
        if data["label"].ndim != 4 or data["label"].shape[0] != 1:
            raise ValueError(f"{path} label must have shape [1,D,H,W], got {data['label'].shape}")
        if data["image"].shape[1:] != data["label"].shape[1:]:
            raise ValueError(f"{path} image/label spatial shapes differ: {data['image'].shape}, {data['label'].shape}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.mode == "train" and self._positive_indices and self._negative_indices:
            if self.rng.random() < self.positive_case_ratio:
                index = int(self.rng.choice(self._positive_indices))
            else:
                index = int(self.rng.choice(self._negative_indices))
        path = self.samples[index]
        with np.load(path, allow_pickle=False) as data:
            self._validate_npz(data, path)
            image = data["image"].astype(np.float32)
            label = data["label"].astype(np.float32)
            case_id = str(data["case_id"])
            metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}

        if self.mode == "train" and self.train_patch_size is not None:
            image, label = self._sample_patch(image, label)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "case_id": case_id,
            "metadata": metadata,
            "npz_path": str(path),
        }

    def _sample_patch(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        patch_size = self.train_patch_size
        assert patch_size is not None
        volume_shape = image.shape[1:]
        want_positive = self.use_lesion_aware_sampling and self.rng.random() < self.pos_patch_ratio
        center = sample_lesion_center(label, self.rng) if want_positive else None
        if center is None:
            center = sample_random_center(volume_shape, self.rng)
        slc = compute_patch_slices(center, patch_size, volume_shape)
        image_patch = image[:, slc[0], slc[1], slc[2]]
        label_patch = label[:, slc[0], slc[1], slc[2]]
        return self._pad_to_patch(image_patch, label_patch, patch_size)

    @staticmethod
    def _pad_to_patch(image: np.ndarray, label: np.ndarray, patch_size: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        pads = []
        for got, target in zip(image.shape[1:], patch_size):
            total = max(int(target) - int(got), 0)
            pads.append((total // 2, total - total // 2))
        if any(a or b for a, b in pads):
            image = np.pad(image, [(0, 0), *pads], mode="constant")
            label = np.pad(label, [(0, 0), *pads], mode="constant")
        return image, label
