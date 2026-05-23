"""Multi-task loss for PCaSAM-3D-ProFound training.

The model produces two segmentation outputs:
1. coarse_logits: from the lightweight coarse branch (auxiliary supervision)
2. refined_logits: from the SAM mask decoder (primary supervision)

Plus an optional IoU prediction loss.

Loss = refined_weight * L_refined + coarse_weight * L_coarse + iou_weight * L_iou

Where:
- L_refined = Dice + Focal-Tversky + BCE (recall-oriented for lesion detection)
- L_coarse  = Dice + Focal-Tversky + light BCE (recall-oriented prompt source)
- L_iou     = MSE(iou_pred, actual_dice) (mask quality estimation)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCaSAM3DLoss(nn.Module):
    """Multi-task loss for PCaSAM-3D-ProFound.

    Combines refined mask loss, coarse mask loss, and IoU prediction loss.
    """

    def __init__(
        self,
        # Refined mask loss weights
        refined_dice_weight: float = 1.0,
        refined_focal_tversky_weight: float = 0.7,
        refined_bce_weight: float = 0.3,
        refined_tversky_fn_weight: float = 0.7,
        refined_tversky_fp_weight: float = 0.3,
        refined_focal_gamma: float = 4.0 / 3.0,
        refined_bce_pos_weight: float | None = 3.0,
        refined_boundary_bce_weight: float = 0.0,
        refined_boundary_weight: float = 4.0,
        boundary_kernel_size: int = 3,
        # Coarse mask loss weights
        coarse_dice_weight: float = 1.0,
        coarse_focal_tversky_weight: float = 0.8,
        coarse_bce_weight: float = 0.3,
        coarse_tversky_fn_weight: float = 0.8,
        coarse_tversky_fp_weight: float = 0.2,
        coarse_focal_gamma: float = 4.0 / 3.0,
        coarse_bce_pos_weight: float | None = 6.0,
        coarse_aux_weight: float = 0.25,
        # Task weights
        refined_weight: float = 1.0,
        coarse_weight: float = 0.3,
        iou_weight: float = 0.1,
        objectness_weight: float = 0.10,
        objectness_pos_weight: float | None = 2.0,
        # Options
        use_iou_loss: bool = True,
    ) -> None:
        super().__init__()
        self.refined_weight = refined_weight
        self.coarse_weight = coarse_weight
        self.iou_weight = iou_weight
        self.objectness_weight = objectness_weight
        self.use_iou_loss = use_iou_loss
        pw_objectness = (
            torch.tensor([objectness_pos_weight])
            if objectness_pos_weight is not None
            else None
        )
        self.register_buffer("objectness_pos_weight", pw_objectness)

        # Refined mask loss components
        self.refined_dice_weight = refined_dice_weight
        self.refined_focal_tversky_weight = refined_focal_tversky_weight
        self.refined_bce_weight = refined_bce_weight
        self.refined_fn_weight = refined_tversky_fn_weight
        self.refined_fp_weight = refined_tversky_fp_weight
        self.refined_gamma = refined_focal_gamma
        self.refined_boundary_bce_weight = float(refined_boundary_bce_weight)
        self.refined_boundary_weight = float(refined_boundary_weight)
        self.boundary_kernel_size = int(boundary_kernel_size)

        pw_refined = (
            torch.tensor([refined_bce_pos_weight])
            if refined_bce_pos_weight is not None
            else None
        )
        self.register_buffer("refined_pos_weight", pw_refined)

        # Coarse mask loss components
        self.coarse_dice_weight = coarse_dice_weight
        self.coarse_focal_tversky_weight = coarse_focal_tversky_weight
        self.coarse_bce_weight = coarse_bce_weight
        self.coarse_fn_weight = coarse_tversky_fn_weight
        self.coarse_fp_weight = coarse_tversky_fp_weight
        self.coarse_gamma = coarse_focal_gamma
        self.coarse_aux_weight = float(coarse_aux_weight)
        pw_coarse = (
            torch.tensor([coarse_bce_pos_weight])
            if coarse_bce_pos_weight is not None
            else None
        )
        self.register_buffer("coarse_pos_weight", pw_coarse)

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        target: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
        gland_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss.

        Args:
            model_output: dict from PCaSAM3DProFoundModel.forward()
            target: [B, 1, D, H, W] ground truth binary mask

        Returns:
            dict with 'total_loss', 'refined_loss', 'coarse_loss', 'iou_loss'
        """
        refined_logits = model_output["refined_logits"]
        coarse_logits = model_output["coarse_logits"]
        coarse_aux_logits = model_output.get("coarse_aux_logits")
        iou_pred = model_output.get("iou_pred")
        objectness_logit = model_output.get("objectness_logit")

        target = target.float()

        # ─── Refined mask loss ───
        refined_loss = self._refined_loss(refined_logits, target, boundary_mask=boundary_mask)

        # ─── Coarse mask loss ───
        coarse_loss, coarse_aux_loss = self._coarse_loss_with_aux(coarse_logits, target, coarse_aux_logits)

        # ─── IoU prediction loss ───
        iou_loss = torch.tensor(0.0, device=target.device)
        if self.use_iou_loss and iou_pred is not None:
            with torch.no_grad():
                actual_dice = self._compute_dice(refined_logits, target)
            iou_loss = F.mse_loss(iou_pred.squeeze(-1), actual_dice)

        objectness_loss = self.objectness_loss(objectness_logit, target)

        # ─── Total ───
        total_loss = (
            self.refined_weight * refined_loss
            + self.coarse_weight * coarse_loss
            + self.iou_weight * iou_loss
            + self.objectness_weight * objectness_loss
        )

        return {
            "total_loss": total_loss,
            "refined_loss": refined_loss,
            "coarse_loss": coarse_loss,
            "coarse_aux_loss": coarse_aux_loss,
            "iou_loss": iou_loss,
            "objectness_loss": objectness_loss,
        }

    def _refined_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dice + Focal-Tversky + BCE plus optional boundary-weighted BCE."""
        loss = torch.tensor(0.0, device=logits.device)

        if self.refined_dice_weight > 0:
            loss = loss + self.refined_dice_weight * self._dice_loss(logits, target)

        if self.refined_focal_tversky_weight > 0:
            loss = loss + self.refined_focal_tversky_weight * self._focal_tversky_loss(
                logits, target,
                fn_weight=self.refined_fn_weight,
                fp_weight=self.refined_fp_weight,
                gamma=self.refined_gamma,
            )

        if self.refined_bce_weight > 0:
            pw = self.refined_pos_weight
            if pw is not None:
                pw = pw.to(logits.device)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
            loss = loss + self.refined_bce_weight * bce

        if self.refined_boundary_bce_weight > 0:
            loss = loss + self.refined_boundary_bce_weight * self._boundary_weighted_bce(
                logits, target, boundary_mask=boundary_mask
            )

        return loss

    def coarse_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        aux_logits: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Public coarse-only loss for PCaSAM-style stage-1 training."""
        total, _ = self._coarse_loss_with_aux(logits, target.float(), aux_logits)
        return total

    def _coarse_loss_with_aux(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        aux_logits: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_loss = self._coarse_loss(logits, target.float())
        aux_loss = torch.tensor(0.0, device=logits.device)
        if aux_logits:
            aux_terms = [self._coarse_loss(aux, target.float()) for aux in aux_logits]
            aux_loss = torch.stack(aux_terms).mean() * self.coarse_aux_weight
        return main_loss + aux_loss, aux_loss

    def objectness_loss(self, objectness_logit: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        """Case-level objectness loss for suppressing negative-case prompts."""
        if objectness_logit is None:
            return torch.tensor(0.0, device=target.device)
        target = target.float()
        dims = tuple(range(1, target.ndim))
        case_target = (target.sum(dim=dims) > 0).float()
        pw = self.objectness_pos_weight
        if pw is not None:
            pw = pw.to(objectness_logit.device)
        return F.binary_cross_entropy_with_logits(
            objectness_logit.view(-1), case_target.view(-1), pos_weight=pw
        )

    def _coarse_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Recall-oriented coarse loss for prompt generation.

        Coarse masks are proposal sources, so false negatives are more damaging
        than false positives. A focal Tversky term biases the branch toward
        lesion coverage while a lighter BCE term keeps probabilities calibrated.
        """
        loss = torch.tensor(0.0, device=logits.device)

        if self.coarse_dice_weight > 0:
            loss = loss + self.coarse_dice_weight * self._dice_loss(logits, target)

        if self.coarse_focal_tversky_weight > 0:
            loss = loss + self.coarse_focal_tversky_weight * self._focal_tversky_loss(
                logits, target,
                fn_weight=self.coarse_fn_weight,
                fp_weight=self.coarse_fp_weight,
                gamma=self.coarse_gamma,
            )

        if self.coarse_bce_weight > 0:
            pw = self.coarse_pos_weight
            if pw is not None:
                pw = pw.to(logits.device)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
            loss = loss + self.coarse_bce_weight * bce

        return loss

    def _boundary_weighted_bce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BCE that emphasizes uncertain/boundary voxels.

        Uses the precomputed boundary_uncertainty_mask when available. If an older
        dataset lacks it, fall back to a narrow morphological band around the GT.
        """
        if boundary_mask is None:
            boundary_mask = self._target_boundary_band(target)
        boundary_mask = boundary_mask.to(device=logits.device, dtype=logits.dtype)
        if boundary_mask.shape[2:] != logits.shape[2:]:
            boundary_mask = F.interpolate(
                boundary_mask, size=logits.shape[2:], mode="trilinear", align_corners=False
            )
        boundary_mask = boundary_mask.clamp(0.0, 1.0)
        weight = 1.0 + self.refined_boundary_weight * boundary_mask
        pw = self.refined_pos_weight
        if pw is not None:
            pw = pw.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
        return (bce * weight).sum() / weight.sum().clamp_min(1.0)

    def _target_boundary_band(self, target: torch.Tensor) -> torch.Tensor:
        k = max(int(self.boundary_kernel_size), 3)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        target = target.float()
        dilated = F.max_pool3d(target, kernel_size=k, stride=1, padding=pad)
        eroded = 1.0 - F.max_pool3d(1.0 - target, kernel_size=k, stride=1, padding=pad)
        return (dilated - eroded).clamp(0.0, 1.0)

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def _focal_tversky_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0,
        fn_weight: float = 0.7,
        fp_weight: float = 0.3,
        gamma: float = 4.0 / 3.0,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        tversky = (tp + smooth) / (tp + fp_weight * fp + fn_weight * fn + smooth)
        return torch.pow(1.0 - tversky.mean(), gamma)

    @staticmethod
    def _compute_dice(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Compute per-sample Dice for IoU loss target."""
        preds = (torch.sigmoid(logits) > 0.5).float()
        dims = tuple(range(1, preds.ndim))
        intersection = (preds * target).sum(dim=dims)
        union = preds.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.squeeze(-1) if dice.ndim > 1 else dice


def build_pcasam3d_loss(cfg: dict) -> PCaSAM3DLoss:
    """Build PCaSAM3DLoss from config."""
    loss_cfg = cfg.get("loss", cfg)
    return PCaSAM3DLoss(
        refined_dice_weight=loss_cfg.get("refined_dice_weight", 1.0),
        refined_focal_tversky_weight=loss_cfg.get("refined_focal_tversky_weight", 0.7),
        refined_bce_weight=loss_cfg.get("refined_bce_weight", 0.3),
        refined_tversky_fn_weight=loss_cfg.get("refined_tversky_fn_weight", 0.7),
        refined_tversky_fp_weight=loss_cfg.get("refined_tversky_fp_weight", 0.3),
        refined_focal_gamma=loss_cfg.get("refined_focal_gamma", 4.0 / 3.0),
        refined_bce_pos_weight=loss_cfg.get("refined_bce_pos_weight", 3.0),
        refined_boundary_bce_weight=loss_cfg.get("refined_boundary_bce_weight", 0.0),
        refined_boundary_weight=loss_cfg.get("refined_boundary_weight", 4.0),
        boundary_kernel_size=loss_cfg.get("boundary_kernel_size", 3),
        coarse_dice_weight=loss_cfg.get("coarse_dice_weight", 1.0),
        coarse_focal_tversky_weight=loss_cfg.get("coarse_focal_tversky_weight", 0.8),
        coarse_bce_weight=loss_cfg.get("coarse_bce_weight", 0.3),
        coarse_tversky_fn_weight=loss_cfg.get("coarse_tversky_fn_weight", 0.8),
        coarse_tversky_fp_weight=loss_cfg.get("coarse_tversky_fp_weight", 0.2),
        coarse_focal_gamma=loss_cfg.get("coarse_focal_gamma", 4.0 / 3.0),
        coarse_bce_pos_weight=loss_cfg.get("coarse_bce_pos_weight", 6.0),
        coarse_aux_weight=loss_cfg.get("coarse_aux_weight", 0.25),
        refined_weight=loss_cfg.get("refined_weight", 1.0),
        coarse_weight=loss_cfg.get("coarse_weight", 0.3),
        iou_weight=loss_cfg.get("iou_weight", 0.1),
        objectness_weight=loss_cfg.get("objectness_weight", 0.10),
        objectness_pos_weight=loss_cfg.get("objectness_pos_weight", 2.0),
        use_iou_loss=loss_cfg.get("use_iou_loss", True),
    )
