"""Dice + BCEWithLogits loss for binary lesion segmentation."""
from __future__ import annotations
import torch
from torch import nn
from .dice_loss import DiceLoss


class DiceBCELoss(nn.Module):
    """Combine soft Dice loss and BCEWithLogitsLoss.

    The model must output raw logits. Sigmoid is applied only inside Dice and
    inference, while BCE uses PyTorch's numerically stable logits implementation.
    """
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice = DiceLoss()
        pw = torch.tensor([float(pos_weight)]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.bce.pos_weight is not None and self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        targets = targets.float()
        return self.dice_weight * self.dice(logits, targets) + self.bce_weight * self.bce(logits, targets)
