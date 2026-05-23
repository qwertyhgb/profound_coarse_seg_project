"""Automatic 3D prompt generation from coarse lesion probability maps.

This module turns a Stage-1 coarse probability map into proposal prompts used by
Stage 2. It follows the same practical pattern as prompt-free-to-prompt-guided
medical SAM variants: threshold the coarse mask, clean connected components,
then export a 3D bounding box, component centroid, and optional uncertainty
points for each candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PromptComponent3D:
    """One connected-component proposal converted to 3D spatial prompts."""

    component_id: int
    bbox_zyxzyx: tuple[int, int, int, int, int, int]
    centroid_zyx: tuple[float, float, float]
    voxels: int
    box_volume: int
    max_probability: float
    mean_probability: float
    uncertainty_points_zyx: tuple[tuple[int, int, int], ...] = ()

    def to_row(self, case_id: str, proposal_rank: int, **extra) -> dict:
        """Return a CSV/JSON friendly row."""
        z0, z1, y0, y1, x0, x1 = self.bbox_zyxzyx
        cz, cy, cx = self.centroid_zyx
        row = {
            "case_id": case_id,
            "proposal_rank": int(proposal_rank),
            "component_id": int(self.component_id),
            "z0": z0,
            "z1": z1,
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
            "center_z": cz,
            "center_y": cy,
            "center_x": cx,
            "component_voxels": int(self.voxels),
            "box_volume": int(self.box_volume),
            "max_probability": float(self.max_probability),
            "mean_probability": float(self.mean_probability),
            "uncertainty_points_zyx": [list(p) for p in self.uncertainty_points_zyx],
        }
        row.update(extra)
        return row


class AutoPromptGenerator3D:
    """Generate box, centroid, and uncertainty prompts from a 3D probability map.

    The default settings are intentionally conservative for Stage-2 candidate
    generation: keep small but confident components, rank candidates by maximum
    probability, and optionally cap prompts per case.
    """

    def __init__(
        self,
        threshold: float = 0.25,
        min_component_size: int = 50,
        min_max_probability: float = 0.5,
        top_k_per_case: int = 5,
        rank_by: str = "max_probability",
        bbox_margin: Sequence[int] = (0, 0, 0),
        uncertainty_threshold: float = 0.5,
        max_uncertainty_points: int = 0,
    ) -> None:
        self.threshold = float(threshold)
        self.min_component_size = int(min_component_size)
        self.min_max_probability = float(min_max_probability)
        self.top_k_per_case = int(top_k_per_case)
        self.rank_by = str(rank_by)
        self.bbox_margin = tuple(int(v) for v in bbox_margin)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.max_uncertainty_points = int(max_uncertainty_points)

    def generate(self, probability: np.ndarray) -> list[PromptComponent3D]:
        """Return sorted prompt components for one case.

        Args:
            probability: coarse probability map with shape ``[D,H,W]`` or
                ``[1,D,H,W]``.
        """
        prob = _as_3d_probability(probability)
        pred = prob >= self.threshold
        labeled, n_components = ndimage.label(pred, structure=np.ones((3, 3, 3), dtype=np.uint8))
        components: list[PromptComponent3D] = []
        for component_id in range(1, n_components + 1):
            mask = labeled == component_id
            voxels = int(mask.sum())
            if voxels < self.min_component_size:
                continue
            max_prob = float(prob[mask].max())
            if max_prob < self.min_max_probability:
                continue
            bbox = _component_bbox(mask, prob.shape, self.bbox_margin)
            centroid = _weighted_centroid(mask, prob)
            uncertainty_points = _uncertainty_points(
                prob,
                mask,
                threshold=self.uncertainty_threshold,
                max_points=self.max_uncertainty_points,
            )
            components.append(
                PromptComponent3D(
                    component_id=component_id,
                    bbox_zyxzyx=bbox,
                    centroid_zyx=centroid,
                    voxels=voxels,
                    box_volume=_box_volume(bbox),
                    max_probability=max_prob,
                    mean_probability=float(prob[mask].mean()),
                    uncertainty_points_zyx=uncertainty_points,
                )
            )
        components.sort(key=lambda c: _rank_value(c, self.rank_by), reverse=True)
        if self.top_k_per_case > 0:
            components = components[: self.top_k_per_case]
        return components

    def to_rows(self, case_id: str, probability: np.ndarray, **extra) -> list[dict]:
        """Generate CSV/JSON friendly rows for one case."""
        return [component.to_row(case_id, rank, **extra) for rank, component in enumerate(self.generate(probability), start=1)]


def generate_prompts_from_probability(probability: np.ndarray, **kwargs) -> list[PromptComponent3D]:
    """Functional convenience wrapper around :class:`AutoPromptGenerator3D`."""
    return AutoPromptGenerator3D(**kwargs).generate(probability)


def _as_3d_probability(probability: np.ndarray) -> np.ndarray:
    prob = np.asarray(probability, dtype=np.float32)
    if prob.ndim == 4 and prob.shape[0] == 1:
        prob = prob[0]
    if prob.ndim != 3:
        raise ValueError(f"Expected probability shape [D,H,W] or [1,D,H,W], got {probability.shape}")
    return np.clip(prob, 0.0, 1.0)


def _component_bbox(mask: np.ndarray, shape: Sequence[int], margin: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute bbox for empty component")
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    mz, my, mx = margin
    d, h, w = [int(v) for v in shape]
    return (
        max(0, int(z0) - mz),
        min(d, int(z1) + mz),
        max(0, int(y0) - my),
        min(h, int(y1) + my),
        max(0, int(x0) - mx),
        min(w, int(x1) + mx),
    )


def _weighted_centroid(mask: np.ndarray, prob: np.ndarray) -> tuple[float, float, float]:
    coords = np.argwhere(mask)
    weights = prob[mask].astype(np.float64)
    if coords.size == 0:
        raise ValueError("Cannot compute centroid for empty component")
    if float(weights.sum()) <= 1e-8:
        center = coords.mean(axis=0)
    else:
        center = (coords * weights[:, None]).sum(axis=0) / weights.sum()
    return tuple(float(v) for v in center)


def _uncertainty_points(prob: np.ndarray, mask: np.ndarray, threshold: float, max_points: int) -> tuple[tuple[int, int, int], ...]:
    if max_points <= 0:
        return ()
    coords = np.argwhere(mask)
    if coords.size == 0:
        return ()
    values = np.abs(prob[mask] - float(threshold))
    order = np.argsort(values)[:max_points]
    return tuple(tuple(int(v) for v in coords[i]) for i in order)


def _box_volume(box: Sequence[int]) -> int:
    z0, z1, y0, y1, x0, x1 = [int(v) for v in box]
    return max(z1 - z0, 0) * max(y1 - y0, 0) * max(x1 - x0, 0)


def _rank_value(component: PromptComponent3D, rank_by: str) -> float:
    if rank_by == "max_probability":
        return component.max_probability
    if rank_by == "mean_probability":
        return component.mean_probability
    if rank_by == "component_volume":
        return float(component.voxels)
    if rank_by == "box_volume":
        return float(component.box_volume)
    if rank_by == "maxprob_x_volume":
        return component.max_probability * float(component.voxels)
    raise ValueError(f"Unsupported rank_by: {rank_by}")
