"""Tversky-family losses for recall-oriented small-lesion segmentation.

Tversky loss generalizes Dice by assigning different penalties to false
positives and false negatives. For coarse lesion proposal generation, false
negatives are usually more harmful than false positives because missed lesions
cannot produce useful downstream prompts. `fn_weight > fp_weight` therefore
biases optimization toward higher recall.
"""
from __future__ import annotations
import torch
from torch import nn


class TverskyLoss(nn.Module):
    """Binary Tversky loss for logits.

    Args:
        fp_weight: Penalty weight for false-positive probability mass.
        fn_weight: Penalty weight for false-negative probability mass.
        smooth: Numerical stabilizer.
    """

    def __init__(self, fp_weight: float = 0.3, fn_weight: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        self.fp_weight = float(fp_weight)
        self.fn_weight = float(fn_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        dims = tuple(range(1, probs.ndim))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        score = (tp + self.smooth) / (tp + self.fp_weight * fp + self.fn_weight * fn + self.smooth)
        return 1.0 - score.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for sparse lesion segmentation.

    The focal exponent emphasizes hard examples. With `fn_weight > fp_weight`,
    the loss is recall-oriented and better aligned with coarse proposal use.
    """

    def __init__(
        self,
        fp_weight: float = 0.3,
        fn_weight: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(fp_weight=fp_weight, fn_weight=fn_weight, smooth=smooth)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.tversky(logits, targets), self.gamma)


class DiceFocalTverskyBCELoss(nn.Module):
    """Compound Dice + Focal-Tversky + BCEWithLogits loss.

    This keeps the stable BCE term, preserves Dice-style overlap learning, and
    adds a false-negative-aware term for coarse lesion recall.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_tversky_weight: float = 1.0,
        bce_weight: float = 0.3,
        fp_weight: float = 0.3,
        fn_weight: float = 0.7,
        gamma: float = 4.0 / 3.0,
        pos_weight: float | None = 3.0,
    ) -> None:
        super().__init__()
        from .dice_loss import DiceLoss

        self.dice_weight = float(dice_weight)
        self.focal_tversky_weight = float(focal_tversky_weight)
        self.bce_weight = float(bce_weight)
        self.dice = DiceLoss()
        self.focal_tversky = FocalTverskyLoss(fp_weight=fp_weight, fn_weight=fn_weight, gamma=gamma)
        pw = torch.tensor([float(pos_weight)]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        targets = targets.float()
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.focal_tversky_weight * self.focal_tversky(logits, targets)
            + self.bce_weight * self.bce(logits, targets)
        )
