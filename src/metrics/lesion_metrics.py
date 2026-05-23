"""Lesion-level connected-component metrics for coarse proposal evaluation."""
from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage


class LesionRecallAccumulator:
    """Accumulate lesion-wise recall and predicted-component burden.

    A GT lesion component is counted as hit if any predicted voxel overlaps it.
    A predicted component is counted as false positive if it overlaps no GT
    lesion. The latter matters for two-stage promptable segmentation because
    coarse connected components often become downstream boxes/points.
    """

    def __init__(self, threshold: float = 0.5, min_pred_component_size: int = 0) -> None:
        self.threshold = threshold
        self.min_pred_component_size = int(min_pred_component_size)
        self.reset()

    def reset(self) -> None:
        self.hit = 0
        self.total = 0
        self.pred_components = 0
        self.tp_pred_components = 0
        self.fp_pred_components = 0
        self.cases = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        preds_np = probs > self.threshold
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        for pred, gt in zip(preds_np, targets_np):
            self.cases += 1
            pred3 = pred[0] if pred.ndim == 4 else pred
            gt3 = (gt[0] if gt.ndim == 4 else gt) > 0.5
            gt_comp, n_gt = ndimage.label(gt3, structure=structure)
            pred_comp, n_pred = ndimage.label(pred3, structure=structure)
            self.total += int(n_gt)

            valid_pred_ids = []
            for pred_id in range(1, n_pred + 1):
                pred_component = pred_comp == pred_id
                if self.min_pred_component_size > 0 and int(pred_component.sum()) < self.min_pred_component_size:
                    continue
                valid_pred_ids.append(pred_id)
                self.pred_components += 1
                if np.logical_and(pred_component, gt3).any():
                    self.tp_pred_components += 1
                else:
                    self.fp_pred_components += 1

            valid_pred_mask = np.isin(pred_comp, valid_pred_ids) if valid_pred_ids else np.zeros_like(pred3, dtype=bool)
            for gt_id in range(1, n_gt + 1):
                lesion = gt_comp == gt_id
                if np.logical_and(valid_pred_mask, lesion).any():
                    self.hit += 1

    def compute(self) -> dict[str, float]:
        recall = self.hit / self.total if self.total > 0 else 0.0
        component_precision = self.tp_pred_components / self.pred_components if self.pred_components > 0 else 0.0
        fp_per_case = self.fp_pred_components / max(self.cases, 1)
        pred_per_case = self.pred_components / max(self.cases, 1)
        return {
            "lesion_recall": recall,
            "hit_lesions": float(self.hit),
            "total_gt_lesions": float(self.total),
            "pred_components": float(self.pred_components),
            "tp_pred_components": float(self.tp_pred_components),
            "fp_pred_components": float(self.fp_pred_components),
            "component_precision": component_precision,
            "fp_components_per_case": fp_per_case,
            "pred_components_per_case": pred_per_case,
        }
