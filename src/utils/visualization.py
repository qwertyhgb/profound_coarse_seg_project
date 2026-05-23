"""Visualization helpers for coarse lesion predictions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _as_3d(array: np.ndarray | None) -> np.ndarray | None:
    """Convert [1,D,H,W] or [D,H,W] arrays to [D,H,W]."""
    if array is None:
        return None
    if array.ndim == 4:
        return array[0]
    return array


def save_case_png(
    image: np.ndarray,
    label: np.ndarray | None,
    prob: np.ndarray | None,
    output_path: str | Path,
    case_id: str = "case",
    segmentation_threshold: float = 0.5,
    prompt_threshold: float | None = None,
) -> None:
    """Save an axial visualization for GT, probability and thresholded masks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image3d = image[0] if image.ndim == 4 else image
    label3d = _as_3d(label)
    prob3d = _as_3d(prob)

    score = None
    if label3d is not None and label3d.sum() > 0:
        score = label3d
    elif prob3d is not None:
        score = prob3d
    z = int(np.argmax(score.reshape(score.shape[0], -1).sum(axis=1))) if score is not None else image3d.shape[0] // 2

    ncols = 5 if prob3d is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(image3d[z], cmap="gray")
    axes[0].set_title(f"{case_id} T2W z={z}")

    axes[1].imshow(image3d[z], cmap="gray")
    if label3d is not None:
        gt = label3d[z] > 0
        axes[1].imshow(np.ma.masked_where(~gt, gt), alpha=0.45, cmap="Greens")
    axes[1].set_title("GT")

    axes[2].imshow(image3d[z], cmap="gray")
    if prob3d is not None:
        axes[2].imshow(prob3d[z], alpha=0.45, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Coarse prob")

    if prob3d is not None:
        seg_mask = prob3d[z] >= float(segmentation_threshold)
        axes[3].imshow(image3d[z], cmap="gray")
        axes[3].imshow(np.ma.masked_where(~seg_mask, seg_mask), alpha=0.45, cmap="Reds")
        axes[3].set_title(f"Mask @{segmentation_threshold:.2f}")

        pthr = float(prompt_threshold if prompt_threshold is not None else segmentation_threshold)
        prompt_mask = prob3d[z] >= pthr
        axes[4].imshow(image3d[z], cmap="gray")
        axes[4].imshow(np.ma.masked_where(~prompt_mask, prompt_mask), alpha=0.45, cmap="Blues")
        axes[4].set_title(f"Prompt @{pthr:.2f}")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
