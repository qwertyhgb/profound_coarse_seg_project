"""Voxel-level binary segmentation metrics."""
from __future__ import annotations
import torch


class SegmentationMetricAccumulator:
    """Accumulate Dice/Precision/Recall over batches using global counts."""
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0; self.fp = 0.0; self.fn = 0.0
        self.pos_dice_sum = 0.0; self.pos_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        preds = probs > self.threshold
        targets = targets > 0.5
        self.tp += float((preds & targets).sum().item())
        self.fp += float((preds & ~targets).sum().item())
        self.fn += float((~preds & targets).sum().item())
        for p, t in zip(preds, targets):
            if t.sum() > 0:
                tp = float((p & t).sum().item())
                fp = float((p & ~t).sum().item())
                fn = float((~p & t).sum().item())
                self.pos_dice_sum += (2 * tp) / max(2 * tp + fp + fn, 1e-8)
                self.pos_count += 1

    def compute(self) -> dict[str, float]:
        dice = (2 * self.tp) / max(2 * self.tp + self.fp + self.fn, 1e-8)
        precision = self.tp / max(self.tp + self.fp, 1e-8)
        recall = self.tp / max(self.tp + self.fn, 1e-8)
        pos_dice = self.pos_dice_sum / max(self.pos_count, 1)
        return {"dice": dice, "precision": precision, "recall": recall, "positive_case_dice": pos_dice}
