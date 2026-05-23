"""Sampling helpers for PI-CAI lesion-aware training."""
from __future__ import annotations
from typing import Sequence
import numpy as np


def compute_patch_slices(center: Sequence[int], patch_size: Sequence[int], volume_shape: Sequence[int]) -> tuple[slice, slice, slice]:
    """Compute clamped [D,H,W] patch slices around a center voxel."""
    starts = []
    for c, p, s in zip(center, patch_size, volume_shape):
        start = int(c) - int(p) // 2
        start = max(0, min(start, max(int(s) - int(p), 0)))
        starts.append(start)
    return tuple(slice(st, min(st + int(p), int(s))) for st, p, s in zip(starts, patch_size, volume_shape))


def sample_lesion_center(label: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """Sample a random foreground voxel from [1,D,H,W] or [D,H,W] label."""
    lab = label[0] if label.ndim == 4 else label
    coords = np.argwhere(lab > 0)
    if coords.size == 0:
        return None
    return coords[int(rng.integers(0, len(coords)))]


def sample_random_center(volume_shape: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Sample a random center inside a [D,H,W] volume."""
    return np.array([int(rng.integers(0, max(int(s), 1))) for s in volume_shape], dtype=np.int64)
