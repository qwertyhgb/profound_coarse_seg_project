#!/usr/bin/env python
"""Train PCaSAM-3D-ProFound: end-to-end ProFound encoder + SAM-Med3D decoder.

This script trains the unified model that combines:
- ProFound-Conv as domain-specific image encoder
- Lightweight coarse branch for automatic 3D prompt generation
- SAM-Med3D prompt encoder + mask decoder for refined segmentation

Usage:
    /root/anaconda3/envs/lm/bin/python scripts/train_pcasam3d_profound.py \
        --config configs/train_pcasam3d_profound.yaml

    # With fold specification:
    /root/anaconda3/envs/lm/bin/python scripts/train_pcasam3d_profound.py \
        --config configs/train_pcasam3d_profound.yaml \
        --fold 0 \
        --train-split data/splits/5fold/fold_0/train.txt \
        --val-split data/splits/5fold/fold_0/val.txt
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.datasets.pcasam3d_dataset import PCaSAM3DDataset, pcasam3d_collate_fn
from src.models.pcasam3d_profound import build_pcasam3d_profound
from src.models.pcasam3d_profound.pcasam3d_loss import build_pcasam3d_loss
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config_utils import load_config
from src.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(cfg: dict, args) -> tuple[DataLoader, DataLoader]:
    data = cfg["data"]
    processed_root = data["processed_root"]
    patch_size = tuple(data.get("patch_size", [128, 128, 128]))
    seed = int(cfg.get("project", {}).get("seed", 42))

    train_split = args.train_split or data.get("train_split")
    val_split = args.val_split or data.get("val_split")

    train_ds = PCaSAM3DDataset(
        processed_root=processed_root,
        split_file=train_split,
        mode="train",
        patch_size=patch_size,
        use_lesion_aware_sampling=data.get("use_lesion_aware_sampling", True),
        pos_patch_ratio=data.get("pos_patch_ratio", 0.7),
        positive_case_ratio=data.get("positive_case_ratio", 0.6),
        normalize=data.get("normalize", "channelwise_nonzero"),
        gland_aware_negative_sampling=data.get("gland_aware_negative_sampling", False),
        gland_negative_prob=data.get("gland_negative_prob", 0.8),
        augmentation=data.get("augmentation"),
        max_cases=data.get("max_train_cases"),
        seed=seed,
    )

    val_ds = PCaSAM3DDataset(
        processed_root=processed_root,
        split_file=val_split,
        mode="val",
        patch_size=patch_size,
        normalize=data.get("normalize", "channelwise_nonzero"),
        max_cases=data.get("max_val_cases"),
        seed=seed,
    )

    print(f"[Data] Train: {len(train_ds)} cases | Val: {len(val_ds)} cases")
    print(f"[Data] Patch size: {patch_size} | Normalize: {data.get('normalize', 'channelwise_nonzero')}")
    if data.get("gland_aware_negative_sampling", False):
        print("[Data] Gland-aware negative sampling enabled | prob=" + str(data.get("gland_negative_prob", 0.8)))

    num_workers = int(data.get("num_workers", 4))
    persistent = num_workers > 0
    prefetch = int(data.get("prefetch_factor", 2)) if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data.get("batch_size", 2)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(data.get("pin_memory", True)),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=pcasam3d_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data.get("pin_memory", True)),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        collate_fn=pcasam3d_collate_fn,
    )
    return train_loader, val_loader




def apply_training_stage(model: nn.Module, cfg: dict) -> str:
    """Freeze modules according to the PCaSAM-style training stage.

    PCaSAM trains prompt-free coarse localization and prompt-guided refinement
    as distinct problems before joint use. The 3D version follows that idea:
    - coarse: train only the coarse/prompt-source path.
    - prompt: train bridge/refinement with GT-jitter prompts; freeze coarse.
    - joint: train all non-frozen adapters end-to-end.
    """
    stage = str(cfg.get("training", {}).get("stage", "joint")).lower()
    if stage not in {"coarse", "prompt", "joint"}:
        raise ValueError(f"Unknown training.stage: {stage}")

    for p in model.parameters():
        p.requires_grad = False

    train_prefixes: tuple[str, ...]
    if stage == "coarse":
        train_prefixes = (
            "modality_fusion.",
            "encoder.",
            "enhancement.",
            "coarse_branch.",
            "objectness_head.",
        )
    elif stage == "prompt":
        train_prefixes = (
            "feature_bridge.",
            "modality_cross_attention.",
            "self_gated_multiscale.",
            "decoder_alignment.",
            "high_res_refinement.",
            "prompt_encoder.",
            "mask_decoder.",
            "prompt_adapter.",
        )
    else:
        train_prefixes = (
            "modality_fusion.",
            "encoder.",
            "enhancement.",
            "feature_bridge.",
            "modality_cross_attention.",
            "self_gated_multiscale.",
            "decoder_alignment.",
            "high_res_refinement.",
            "coarse_branch.",
            "objectness_head.",
            "auto_prompt.",
            "prompt_encoder.",
            "mask_decoder.",
            "prompt_adapter.",
        )

    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in train_prefixes):
            p.requires_grad = True

    # Respect frozen external foundation encoders regardless of stage.
    for name, p in model.named_parameters():
        if name.startswith("encoder.") and cfg.get("model", {}).get("freeze_encoder", True):
            p.requires_grad = False
        model_cfg = cfg.get("model", {})
        legacy_freeze_sam = bool(model_cfg.get("freeze_sam_decoder", False))
        freeze_prompt_encoder = bool(model_cfg.get("freeze_prompt_encoder", legacy_freeze_sam))
        freeze_mask_decoder = bool(model_cfg.get("freeze_mask_decoder", legacy_freeze_sam))
        train_mask_decoder_adapters = bool(model_cfg.get("train_mask_decoder_adapters", False))
        if name.startswith("prompt_encoder.") and freeze_prompt_encoder:
            p.requires_grad = False
        if name.startswith("mask_decoder.") and freeze_mask_decoder:
            is_adapter = "adapter" in name
            p.requires_grad = bool(train_mask_decoder_adapters and is_adapter)

    return stage

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict):
    """Three parameter groups: encoder (lowest lr), bridge+coarse (mid), SAM decoder (highest)."""
    opt = cfg["optimizer"]
    encoder_lr = float(opt.get("encoder_lr", 1e-5))
    bridge_lr = float(opt.get("bridge_lr", 5e-5))
    decoder_lr = float(opt.get("decoder_lr", 1e-4))
    weight_decay = float(opt.get("weight_decay", 1e-4))

    encoder_params, bridge_params, decoder_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        elif any(name.startswith(prefix) for prefix in (
            "feature_bridge.", "coarse_branch.", "objectness_head.", "enhancement.", "auto_prompt.",
            "modality_fusion.", "modality_cross_attention.", "self_gated_multiscale.", "decoder_alignment.", "high_res_refinement.",
        )):
            bridge_params.append(p)
        else:
            # prompt_encoder, mask_decoder, prompt_adapter
            decoder_params.append(p)

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "initial_lr": encoder_lr})
    if bridge_params:
        groups.append({"params": bridge_params, "lr": bridge_lr, "initial_lr": bridge_lr})
    if decoder_params:
        groups.append({"params": decoder_params, "lr": decoder_lr, "initial_lr": decoder_lr})

    if not groups:
        raise ValueError("No trainable parameters found.")

    n_enc = sum(p.numel() for p in encoder_params)
    n_bridge = sum(p.numel() for p in bridge_params)
    n_dec = sum(p.numel() for p in decoder_params)
    print(f"[Optimizer] encoder: {n_enc/1e6:.2f}M (lr={encoder_lr:.1e}) | "
          f"bridge+coarse: {n_bridge/1e6:.2f}M (lr={bridge_lr:.1e}) | "
          f"SAM decoder: {n_dec/1e6:.2f}M (lr={decoder_lr:.1e})")

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_logits: torch.Tensor,
    label: torch.Tensor,
    threshold: float = 0.5,
    min_pred_size: int = 0,
    include_components: bool = True,
):
    """Compute metrics with positive-case recall separated from true negatives.

    The validation split is mostly negative. Treating every true-negative case as
    recall=0 makes recall-safe model selection impossible. This function keeps
    all-case Dice for sanity, but exposes positive-only Dice/Recall and
    negative-case FP rate for checkpoint selection.
    """
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    gt = (label > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    tp = (pred * gt).sum(dim=dims)
    fp = (pred * (1 - gt)).sum(dim=dims)
    fn = ((1 - pred) * gt).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    gt_sum = gt.sum(dim=dims)

    has_gt = gt_sum > 0
    has_pred = pred_sum > 0
    n = int(pred.shape[0])
    n_pos = int(has_gt.sum().item())
    n_neg = n - n_pos

    denom = 2 * tp + fp + fn
    dice = torch.where(denom > 0, 2 * tp / (denom + 1e-6), torch.ones_like(tp))
    precision = torch.where(pred_sum > 0, tp / (tp + fp + 1e-6), (~has_gt).float())
    recall = torch.where(has_gt, tp / (tp + fn + 1e-6), (~has_pred).float())

    if n_pos > 0:
        positive_dice = dice[has_gt].mean().item()
        positive_precision = precision[has_gt].mean().item()
        positive_recall = recall[has_gt].mean().item()
        detection_rate = has_pred[has_gt].float().mean().item()
    else:
        positive_dice = 0.0
        positive_precision = 0.0
        positive_recall = 0.0
        detection_rate = 0.0

    if n_neg > 0:
        fp_case_rate = has_pred[~has_gt].float().mean().item()
        negative_dice = dice[~has_gt].mean().item()
    else:
        fp_case_rate = 0.0
        negative_dice = 0.0

    if include_components:
        lesion_m = compute_lesion_component_metrics(pred, gt, min_pred_component_size=min_pred_size)
    else:
        lesion_m = {
            "lesion_recall": 0.0,
            "fp_components_per_case": 0.0,
            "pred_components_per_case": 0.0,
        }

    total_tp = float(tp.sum().item())
    total_fp = float(fp.sum().item())
    total_fn = float(fn.sum().item())
    global_dice = 2.0 * total_tp / max(2.0 * total_tp + total_fp + total_fn, 1e-8)

    return {
        "dice": dice.mean().item(),
        "global_dice": global_dice,
        "global_tp": total_tp,
        "global_fp": total_fp,
        "global_fn": total_fn,
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "positive_dice": positive_dice,
        "positive_precision": positive_precision,
        "positive_recall": positive_recall,
        "detection_rate": detection_rate,
        "negative_dice": negative_dice,
        "fp_case_rate": fp_case_rate,
        "lesion_recall": lesion_m["lesion_recall"],
        "fp_components_per_case": lesion_m["fp_components_per_case"],
        "pred_components_per_case": lesion_m["pred_components_per_case"],
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def compute_lesion_component_metrics(
    pred: torch.Tensor, gt: torch.Tensor, min_pred_component_size: int = 0
) -> dict[str, float]:
    """Lesion-level recall and false-positive components for prompt-source selection.

    Args:
        pred: [B, 1, D, H, W] binary prediction
        gt: [B, 1, D, H, W] binary ground truth
        min_pred_component_size: discard predicted components smaller than this (voxels).
            Filters noise without affecting lesion recall (real lesions are typically >50 voxels).
    """
    from scipy import ndimage

    pred_np = pred.detach().cpu().numpy()[:, 0] > 0.5
    gt_np = gt.detach().cpu().numpy()[:, 0] > 0.5
    hit_lesions = 0
    total_lesions = 0
    fp_components = 0
    pred_components = 0

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for pred_case, gt_case in zip(pred_np, gt_np):
        gt_labeled, n_gt = ndimage.label(gt_case, structure=structure)
        pred_labeled, n_pred = ndimage.label(pred_case, structure=structure)
        total_lesions += int(n_gt)

        # Filter small predicted components
        valid_pred_ids = []
        for comp_id in range(1, n_pred + 1):
            if min_pred_component_size > 0:
                comp_size = int((pred_labeled == comp_id).sum())
                if comp_size < min_pred_component_size:
                    continue
            valid_pred_ids.append(comp_id)

        pred_components += len(valid_pred_ids)

        # Build filtered pred mask for lesion recall computation
        if min_pred_component_size > 0 and len(valid_pred_ids) < n_pred:
            filtered_pred = np.zeros_like(pred_case)
            for comp_id in valid_pred_ids:
                filtered_pred |= (pred_labeled == comp_id)
        else:
            filtered_pred = pred_case

        for lesion_id in range(1, n_gt + 1):
            if (filtered_pred & (gt_labeled == lesion_id)).any():
                hit_lesions += 1
        for comp_id in valid_pred_ids:
            if not (gt_case & (pred_labeled == comp_id)).any():
                fp_components += 1

    n_cases = max(int(pred.shape[0]), 1)
    return {
        "lesion_recall": float(hit_lesions / total_lesions) if total_lesions > 0 else 0.0,
        "fp_components_per_case": float(fp_components / n_cases),
        "pred_components_per_case": float(pred_components / n_cases),
    }


def compute_objectness_metrics(objectness_logit: torch.Tensor | None, label: torch.Tensor) -> dict[str, float]:
    """Case-level objectness quality for negative prompt suppression."""
    if objectness_logit is None:
        return {"objectness_acc": 0.0, "objectness_pos_prob": 0.0, "objectness_neg_prob": 0.0}
    target = (label.float().sum(dim=tuple(range(1, label.ndim))) > 0).float().view(-1)
    prob = torch.sigmoid(objectness_logit.view(-1).float())
    pred = (prob >= 0.5).float()
    acc = (pred == target).float().mean().item()
    pos_prob = prob[target > 0.5].mean().item() if (target > 0.5).any() else 0.0
    neg_prob = prob[target <= 0.5].mean().item() if (target <= 0.5).any() else 0.0
    return {
        "objectness_acc": float(acc),
        "objectness_pos_prob": float(pos_prob),
        "objectness_neg_prob": float(neg_prob),
    }


def compute_modality_weight_metrics(modality_weights: torch.Tensor | None) -> dict[str, float]:
    """Extract per-modality gate weights for monitoring fusion behavior."""
    if modality_weights is None:
        return {"modality_weight_t2w": 0.0, "modality_weight_adc": 0.0, "modality_weight_hbv": 0.0}
    # modality_weights: [B, 3] mean gate activations (already detached)
    w = modality_weights.float().mean(dim=0)
    return {
        "modality_weight_t2w": float(w[0].item()),
        "modality_weight_adc": float(w[1].item()),
        "modality_weight_hbv": float(w[2].item()),
    }


def _new_metric_sums() -> dict[str, float]:
    return {
        "dice": 0.0, "global_tp": 0.0, "global_fp": 0.0, "global_fn": 0.0,
        "precision": 0.0, "recall": 0.0,
        "positive_case_dice": 0.0, "positive_precision": 0.0, "positive_recall": 0.0,
        "detection_rate": 0.0, "negative_dice": 0.0, "fp_case_rate": 0.0,
        "lesion_recall": 0.0, "fp_components_per_case": 0.0, "pred_components_per_case": 0.0,
        "n": 0.0, "pos_n": 0.0, "neg_n": 0.0,
    }


def _accumulate_metric_sums(sums: dict[str, float], metrics: dict[str, float]) -> None:
    n_b = max(int(metrics.get("n", 1)), 1)
    pos_b = int(metrics.get("n_pos", 0))
    neg_b = int(metrics.get("n_neg", 0))
    for key in ("dice", "precision", "recall", "fp_components_per_case", "pred_components_per_case"):
        sums[key] += float(metrics.get(key, 0.0)) * n_b
    sums["global_tp"] += float(metrics.get("global_tp", 0.0))
    sums["global_fp"] += float(metrics.get("global_fp", 0.0))
    sums["global_fn"] += float(metrics.get("global_fn", 0.0))
    sums["n"] += n_b
    if pos_b:
        sums["positive_case_dice"] += float(metrics.get("positive_dice", 0.0)) * pos_b
        sums["positive_precision"] += float(metrics.get("positive_precision", 0.0)) * pos_b
        sums["positive_recall"] += float(metrics.get("positive_recall", 0.0)) * pos_b
        sums["detection_rate"] += float(metrics.get("detection_rate", 0.0)) * pos_b
        sums["lesion_recall"] += float(metrics.get("lesion_recall", 0.0)) * pos_b
        sums["pos_n"] += pos_b
    if neg_b:
        sums["negative_dice"] += float(metrics.get("negative_dice", 0.0)) * neg_b
        sums["fp_case_rate"] += float(metrics.get("fp_case_rate", 0.0)) * neg_b
        sums["neg_n"] += neg_b


def _finalize_metric_sums(sums: dict[str, float]) -> dict[str, float]:
    n = max(float(sums.get("n", 0.0)), 1.0)
    pos_n = max(float(sums.get("pos_n", 0.0)), 1.0)
    neg_n = max(float(sums.get("neg_n", 0.0)), 1.0)
    global_tp = float(sums.get("global_tp", 0.0))
    global_fp = float(sums.get("global_fp", 0.0))
    global_fn = float(sums.get("global_fn", 0.0))
    global_dice = 2.0 * global_tp / max(2.0 * global_tp + global_fp + global_fn, 1e-8)
    return {
        "dice": sums["dice"] / n,
        "global_dice": global_dice,
        "global_tp": global_tp,
        "global_fp": global_fp,
        "global_fn": global_fn,
        "precision": sums["precision"] / n,
        "recall": sums["recall"] / n,
        "positive_case_dice": sums["positive_case_dice"] / pos_n,
        "positive_precision": sums["positive_precision"] / pos_n,
        "positive_recall": sums["positive_recall"] / pos_n,
        "detection_rate": sums["detection_rate"] / pos_n,
        "negative_dice": sums["negative_dice"] / neg_n,
        "fp_case_rate": sums["fp_case_rate"] / neg_n,
        "lesion_recall": sums["lesion_recall"] / pos_n,
        "fp_components_per_case": sums["fp_components_per_case"] / n,
        "pred_components_per_case": sums["pred_components_per_case"] / n,
        "n_positive": sums["pos_n"],
        "n_negative": sums["neg_n"],
    }


def coarse_score_from_metrics(metrics: dict[str, float], score_cfg: dict | None = None) -> float:
    score_cfg = score_cfg or {}
    lesion_recall = float(metrics.get("lesion_recall", 0.0))
    positive_dice = float(metrics.get("positive_case_dice", 0.0))
    fp_components = float(metrics.get("fp_components_per_case", 0.0))
    target_recall = float(score_cfg.get("target_lesion_recall", 0.90))
    target_positive_dice = float(score_cfg.get("target_positive_case_dice", 0.0))
    max_fp_components = float(score_cfg.get("max_fp_components_per_case", float("inf")))
    recall_gap = max(0.0, target_recall - lesion_recall)
    dice_gap = max(0.0, target_positive_dice - positive_dice)
    fp_gap = max(0.0, fp_components - max_fp_components)
    return (
        float(score_cfg.get("lesion_recall_weight", 1.0)) * lesion_recall
        + float(score_cfg.get("positive_dice_weight", 0.20)) * positive_dice
        - float(score_cfg.get("fp_components_penalty", 0.03)) * fp_components
        - float(score_cfg.get("below_target_recall_penalty", 1.5)) * recall_gap
        - float(score_cfg.get("below_target_dice_penalty", 0.0)) * dice_gap
        - float(score_cfg.get("above_target_fp_penalty", 0.0)) * fp_gap
    )


def select_global_threshold_sweep(
    threshold_sums: dict[float, dict[str, float]],
    score_cfg: dict | None = None,
) -> dict[str, float]:
    """Select one fixed threshold after accumulating metrics over the whole val set."""
    best = None
    for threshold, sums in threshold_sums.items():
        metrics = _finalize_metric_sums(sums)
        score = coarse_score_from_metrics(metrics, score_cfg)
        row = {
            "threshold_sweep_best_coarse_score": score,
            "threshold_sweep_best_threshold": float(threshold),
            "threshold_sweep_best_lesion_recall": float(metrics.get("lesion_recall", 0.0)),
            "threshold_sweep_best_positive_case_dice": float(metrics.get("positive_case_dice", 0.0)),
            "threshold_sweep_best_positive_recall": float(metrics.get("positive_recall", 0.0)),
            "threshold_sweep_best_global_dice": float(metrics.get("global_dice", 0.0)),
            "threshold_sweep_best_fp_components_per_case": float(metrics.get("fp_components_per_case", 0.0)),
        }
        if best is None or row["threshold_sweep_best_coarse_score"] > best["threshold_sweep_best_coarse_score"]:
            best = row
    return best or {
        "threshold_sweep_best_coarse_score": 0.0,
        "threshold_sweep_best_threshold": 0.0,
        "threshold_sweep_best_lesion_recall": 0.0,
        "threshold_sweep_best_positive_case_dice": 0.0,
        "threshold_sweep_best_positive_recall": 0.0,
        "threshold_sweep_best_global_dice": 0.0,
        "threshold_sweep_best_fp_components_per_case": 0.0,
    }




def refined_score_from_metrics(metrics: dict[str, float], score_cfg: dict | None = None) -> float:
    """Precision-oriented score for selecting refined segmentation thresholds/checkpoints.

    The refined stage should keep lesion recall acceptable while improving Dice,
    positive precision, and false-positive component control.
    """
    score_cfg = score_cfg or {}
    positive_dice = float(metrics.get("positive_case_dice", 0.0))
    global_dice = float(metrics.get("global_dice", 0.0))
    lesion_recall = float(metrics.get("lesion_recall", 0.0))
    positive_precision = float(metrics.get("positive_precision", 0.0))
    fp_components = float(metrics.get("fp_components_per_case", 0.0))
    target_recall = float(score_cfg.get("target_lesion_recall", 0.80))
    target_precision = float(score_cfg.get("target_positive_precision", 0.0))
    max_fp_components = float(score_cfg.get("max_fp_components_per_case", float("inf")))
    recall_gap = max(0.0, target_recall - lesion_recall)
    precision_gap = max(0.0, target_precision - positive_precision)
    fp_gap = max(0.0, fp_components - max_fp_components)
    return (
        float(score_cfg.get("positive_dice_weight", 1.0)) * positive_dice
        + float(score_cfg.get("global_dice_weight", 0.0)) * global_dice
        + float(score_cfg.get("lesion_recall_weight", 0.20)) * lesion_recall
        + float(score_cfg.get("positive_precision_weight", 0.20)) * positive_precision
        - float(score_cfg.get("fp_components_penalty", 0.05)) * fp_components
        - float(score_cfg.get("below_target_recall_penalty", 1.0)) * recall_gap
        - float(score_cfg.get("below_target_precision_penalty", 0.0)) * precision_gap
        - float(score_cfg.get("above_target_fp_penalty", 0.0)) * fp_gap
    )


def select_global_refined_threshold_sweep(
    threshold_sums: dict[float, dict[str, float]],
    score_cfg: dict | None = None,
) -> dict[str, float]:
    """Select one refined-mask threshold after full validation accumulation."""
    best = None
    for threshold, sums in threshold_sums.items():
        metrics = _finalize_metric_sums(sums)
        score = refined_score_from_metrics(metrics, score_cfg)
        row = {
            "refined_sweep_best_score": score,
            "refined_sweep_best_threshold": float(threshold),
            "refined_sweep_best_positive_case_dice": float(metrics.get("positive_case_dice", 0.0)),
            "refined_sweep_best_positive_precision": float(metrics.get("positive_precision", 0.0)),
            "refined_sweep_best_positive_recall": float(metrics.get("positive_recall", 0.0)),
            "refined_sweep_best_global_dice": float(metrics.get("global_dice", 0.0)),
            "refined_sweep_best_lesion_recall": float(metrics.get("lesion_recall", 0.0)),
            "refined_sweep_best_fp_components_per_case": float(metrics.get("fp_components_per_case", 0.0)),
        }
        if best is None or row["refined_sweep_best_score"] > best["refined_sweep_best_score"]:
            best = row
    return best or {
        "refined_sweep_best_score": 0.0,
        "refined_sweep_best_threshold": 0.0,
        "refined_sweep_best_positive_case_dice": 0.0,
        "refined_sweep_best_positive_precision": 0.0,
        "refined_sweep_best_positive_recall": 0.0,
        "refined_sweep_best_global_dice": 0.0,
        "refined_sweep_best_lesion_recall": 0.0,
        "refined_sweep_best_fp_components_per_case": 0.0,
    }


def build_gt_point_prompts(
    label: torch.Tensor,
    max_points: int = 3,
    jitter_std: float = 0.03,
    box_margin_voxels: int = 4,
    box_jitter_std: float = 0.03,
    max_negative_points: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build GT-derived point, box, and dense mask prompts.

    Positive cases receive lesion points, optional hard-negative background
    points inside the loose target box, and a jittered lesion box. Negative
    points teach the prompt-guided decoder to suppress nearby background and
    reduce false-positive spillover in the refinement stage.
    """
    B, _, D, H, W = label.shape
    device = label.device
    max_negative_points = max(int(max_negative_points), 0)
    total_points = max(int(max_points), 1) + max_negative_points
    coords_out = torch.full((B, total_points, 3), 0.5, dtype=label.dtype, device=device)
    labels_out = torch.full((B, total_points), -1, dtype=torch.long, device=device)
    boxes_out = torch.full((B, 2, 3), 0.5, dtype=label.dtype, device=device)
    box_valid = torch.zeros(B, dtype=torch.bool, device=device)

    scale = torch.tensor(
        [max(D - 1, 1), max(H - 1, 1), max(W - 1, 1)],
        dtype=label.dtype,
        device=device,
    )

    for b in range(B):
        coords = (label[b, 0] > 0.5).nonzero(as_tuple=False)
        if coords.numel() == 0:
            continue

        centroid = coords.float().mean(dim=0)
        selected = [centroid]
        if max_points > 1 and coords.shape[0] > 1:
            take = min(max_points - 1, coords.shape[0])
            perm = torch.randperm(coords.shape[0], device=device)[:take]
            selected.extend(coords[perm].float())

        pts = torch.stack(selected[:max_points], dim=0) / scale.view(1, 3)
        if jitter_std > 0:
            pts = pts + torch.randn_like(pts) * float(jitter_std)
        pts = pts.clamp(0.0, 1.0)
        n_pts = pts.shape[0]
        coords_out[b, :n_pts] = pts
        labels_out[b, :n_pts] = 1

        lo_raw = coords.min(dim=0).values.float()
        hi_raw = coords.max(dim=0).values.float()
        margin = torch.tensor([box_margin_voxels, box_margin_voxels, box_margin_voxels], dtype=label.dtype, device=device)
        lo = (lo_raw - margin).clamp_min(0)
        hi = torch.minimum(hi_raw + margin, scale)

        if max_negative_points > 0:
            lo_i = lo.long()
            hi_i = hi.long()
            region = torch.zeros((D, H, W), dtype=torch.bool, device=device)
            region[lo_i[0]:hi_i[0] + 1, lo_i[1]:hi_i[1] + 1, lo_i[2]:hi_i[2] + 1] = True
            bg = region & ~(label[b, 0] > 0.5)
            bg_coords = bg.nonzero(as_tuple=False)
            if bg_coords.numel() == 0:
                bg_coords = (label[b, 0] <= 0.5).nonzero(as_tuple=False)
            if bg_coords.numel() > 0:
                n_neg = min(max_negative_points, bg_coords.shape[0])
                neg = bg_coords[torch.randperm(bg_coords.shape[0], device=device)[:n_neg]].float() / scale.view(1, 3)
                if jitter_std > 0:
                    neg = neg + torch.randn_like(neg) * float(jitter_std) * 0.5
                neg = neg.clamp(0.0, 1.0)
                coords_out[b, n_pts:n_pts + n_neg] = neg
                labels_out[b, n_pts:n_pts + n_neg] = 0

        box = torch.stack([lo / scale, hi / scale], dim=0)
        if box_jitter_std > 0:
            box = box + torch.randn_like(box) * float(box_jitter_std)
            low = torch.minimum(box[0], box[1]).clamp(0.0, 1.0)
            high = torch.maximum(box[0], box[1]).clamp(0.0, 1.0)
            min_extent = torch.tensor([1 / max(D - 1, 1), 1 / max(H - 1, 1), 1 / max(W - 1, 1)], dtype=label.dtype, device=device)
            high = torch.maximum(high, (low + min_extent).clamp(0.0, 1.0))
            box = torch.stack([low, high], dim=0)
        boxes_out[b] = box
        box_valid[b] = True

    return coords_out, labels_out, boxes_out, box_valid, label.float()

def apply_gland_mask_postprocess(logits: torch.Tensor, gland_mask: torch.Tensor | None, cfg: dict) -> torch.Tensor:
    """Suppress logits outside the prostate gland for metric/postprocess evaluation.

    The preprocessing pipeline stores a gland mask for each case. Using it only at
    evaluation/postprocessing time reduces anatomically implausible false positives
    without changing the training loss or checkpoint compatibility.
    """
    pp_cfg = cfg.get("metrics", {}).get("gland_mask_postprocess", {})
    if not bool(pp_cfg.get("enabled", False)) or gland_mask is None:
        return logits
    mask = gland_mask.to(device=logits.device, dtype=logits.dtype)
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.shape[-3:] != logits.shape[-3:]:
        mask = F.interpolate(mask, size=logits.shape[-3:], mode="nearest")
    margin = int(pp_cfg.get("margin_voxels", 0))
    if margin > 0:
        kernel = 2 * margin + 1
        mask = F.max_pool3d(mask, kernel_size=kernel, stride=1, padding=margin)
    mask = mask > float(pp_cfg.get("min_mask_value", 0.5))
    outside_logit = float(pp_cfg.get("outside_logit", -12.0))
    return torch.where(mask, logits, torch.full_like(logits, outside_logit))


def gt_prompt_probability(epoch: int, cfg: dict) -> float:
    prompt_cfg = cfg.get("prompt_curriculum", {})
    if not bool(prompt_cfg.get("enabled", False)):
        return 0.0
    warmup_epochs = int(prompt_cfg.get("gt_prompt_epochs", 5))
    anneal_epochs = int(prompt_cfg.get("anneal_epochs", 10))
    start_prob = float(prompt_cfg.get("gt_prompt_prob", 1.0))
    min_prob = float(prompt_cfg.get("min_gt_prompt_prob", 0.0))
    if epoch <= warmup_epochs:
        return start_prob
    if anneal_epochs <= 0:
        return min_prob
    t = min(max(epoch - warmup_epochs, 0), anneal_epochs) / anneal_epochs
    return start_prob * (1.0 - t) + min_prob * t


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg, epoch):
    model.train()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    grad_clip = cfg["training"].get("grad_clip_norm")

    sums = {"total_loss": 0.0, "refined_loss": 0.0, "coarse_loss": 0.0,
            "objectness_loss": 0.0, "refined_dice": 0.0, "coarse_dice": 0.0, "n": 0}

    progress = tqdm(loader, desc=f"Train E{epoch:03d}", dynamic_ncols=True, leave=False)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        boundary_mask = batch.get("boundary_uncertainty_mask")
        if boundary_mask is not None:
            boundary_mask = boundary_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        stage = str(cfg.get("training", {}).get("stage", "joint")).lower()
        prompt_prob = gt_prompt_probability(epoch, cfg)
        use_gt_prompt = prompt_prob > 0 and torch.rand((), device=device).item() < prompt_prob
        external_coords = external_labels = external_boxes = external_box_valid = external_mask = None
        if use_gt_prompt and stage != "coarse":
            prompt_cfg = cfg.get("prompt_curriculum", {})
            external_coords, external_labels, external_boxes, external_box_valid, external_mask = build_gt_point_prompts(
                label,
                max_points=int(prompt_cfg.get("max_gt_points", 3)),
                jitter_std=float(prompt_cfg.get("jitter_std", 0.03)),
                box_margin_voxels=int(prompt_cfg.get("box_margin_voxels", cfg.get("model", {}).get("box_margin_voxels", 4))),
                box_jitter_std=float(prompt_cfg.get("box_jitter_std", 0.03)),
                max_negative_points=int(prompt_cfg.get("max_gt_negative_points", 0)),
            )
            if not bool(prompt_cfg.get("use_gt_mask_prior", True)):
                external_mask = None

        with torch.amp.autocast("cuda", enabled=amp):
            if stage == "coarse":
                output = model.forward_coarse(image)
                coarse_loss = loss_fn.coarse_loss(output["coarse_logits"], label, output.get("coarse_aux_logits"))
                objectness_loss = loss_fn.objectness_loss(output.get("objectness_logit"), label)
                total_loss = coarse_loss + loss_fn.objectness_weight * objectness_loss
                losses = {
                    "total_loss": total_loss,
                    "refined_loss": torch.zeros((), device=device),
                    "coarse_loss": coarse_loss,
                    "iou_loss": torch.zeros((), device=device),
                    "objectness_loss": objectness_loss,
                }
                loss = losses["total_loss"]
            else:
                output = model(
                    image,
                    external_point_coords=external_coords,
                    external_point_labels=external_labels,
                    external_box_coords=external_boxes,
                    external_box_valid=external_box_valid,
                    external_mask_prior=external_mask,
                )
                losses = loss_fn(output, label, boundary_mask=boundary_mask)
                loss = losses["total_loss"]

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            if stage == "coarse":
                refined_m = compute_metrics(output["coarse_logits"].detach(), label, include_components=False)
                coarse_m = refined_m
            else:
                refined_m = compute_metrics(output["refined_logits"].detach(), label, include_components=False)
                coarse_m = compute_metrics(output["coarse_logits"].detach(), label, include_components=False)

        sums["total_loss"] += float(loss.item())
        sums["refined_loss"] += float(losses["refined_loss"].item())
        sums["coarse_loss"] += float(losses["coarse_loss"].item())
        sums["objectness_loss"] += float(losses.get("objectness_loss", torch.zeros((), device=device)).item())
        sums["refined_dice"] += refined_m["positive_dice"] if refined_m["n_pos"] else refined_m["dice"]
        sums["coarse_dice"] += coarse_m["positive_dice"] if coarse_m["n_pos"] else coarse_m["dice"]
        sums["n"] += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            r_dice=f"{(refined_m['positive_dice'] if refined_m['n_pos'] else refined_m['dice']):.3f}",
            c_dice=f"{(coarse_m['positive_dice'] if coarse_m['n_pos'] else coarse_m['dice']):.3f}",
            gt_p=f"{prompt_prob:.2f}",
        )

    n = max(sums["n"], 1)
    return {
        "loss": sums["total_loss"] / n,
        "refined_loss": sums["refined_loss"] / n,
        "coarse_loss": sums["coarse_loss"] / n,
        "objectness_loss": sums["objectness_loss"] / n,
        "refined_dice": sums["refined_dice"] / n,
        "coarse_dice": sums["coarse_dice"] / n,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg, epoch):
    model.eval()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    threshold = float(cfg.get("metrics", {}).get("threshold", 0.5))
    coarse_threshold = float(cfg.get("metrics", {}).get("coarse_threshold", 0.3))
    metrics_cfg = cfg.get("metrics", {})
    min_pred_size = int(metrics_cfg.get("min_pred_component_size", 0))
    sweep_thresholds = [float(x) for x in metrics_cfg.get("coarse_threshold_sweep", [])]
    sweep_score_cfg = metrics_cfg.get("coarse_score", {})
    refined_sweep_thresholds = [float(x) for x in metrics_cfg.get("refined_threshold_sweep", [])]
    refined_sweep_score_cfg = metrics_cfg.get("refined_score", {})

    sums = {
        "loss": 0.0, "refined_dice": 0.0, "coarse_dice": 0.0,
        "precision": 0.0, "recall": 0.0,
        "positive_case_dice": 0.0, "positive_precision": 0.0, "positive_recall": 0.0,
        "global_tp": 0.0, "global_fp": 0.0, "global_fn": 0.0,
        "detection_rate": 0.0, "fp_case_rate": 0.0, "negative_dice": 0.0,
        "coarse_positive_dice": 0.0, "coarse_positive_recall": 0.0,
        "lesion_recall": 0.0, "fp_components_per_case": 0.0,
        "coarse_lesion_recall": 0.0, "coarse_fp_components_per_case": 0.0,
        "objectness_loss": 0.0, "objectness_acc": 0.0,
        "objectness_pos_prob": 0.0, "objectness_neg_prob": 0.0,
        "objectness_pos_n": 0.0, "objectness_neg_n": 0.0,
        "modality_weight_t2w": 0.0, "modality_weight_adc": 0.0, "modality_weight_hbv": 0.0,
        "n": 0, "pos_n": 0, "neg_n": 0,
    }
    threshold_sums = {thr: _new_metric_sums() for thr in sweep_thresholds}
    refined_threshold_sums = {thr: _new_metric_sums() for thr in refined_sweep_thresholds}

    progress = tqdm(loader, desc=f"Val   E{epoch:03d}", dynamic_ncols=True, leave=False)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        boundary_mask = batch.get("boundary_uncertainty_mask")
        if boundary_mask is not None:
            boundary_mask = boundary_mask.to(device, non_blocking=True)
        gland_mask = batch.get("gland_mask")
        if gland_mask is not None:
            gland_mask = gland_mask.to(device, non_blocking=True)
        stage = str(cfg.get("training", {}).get("stage", "joint")).lower()
        with torch.amp.autocast("cuda", enabled=amp):
            if stage == "coarse":
                output = model.forward_coarse(image)
                coarse_loss = loss_fn.coarse_loss(output["coarse_logits"], label, output.get("coarse_aux_logits"))
                objectness_loss = loss_fn.objectness_loss(output.get("objectness_logit"), label)
                losses = {
                    "total_loss": coarse_loss + loss_fn.objectness_weight * objectness_loss,
                    "coarse_loss": coarse_loss,
                    "objectness_loss": objectness_loss,
                }
            else:
                output = model(image)
                losses = loss_fn(output, label, boundary_mask=boundary_mask)

        coarse_logits_eval = apply_gland_mask_postprocess(output["coarse_logits"], gland_mask, cfg)
        refined_logits_raw = output["coarse_logits"] if stage == "coarse" else output["refined_logits"]
        refined_logits_eval = apply_gland_mask_postprocess(refined_logits_raw, gland_mask, cfg)

        if stage == "coarse":
            refined_m = compute_metrics(coarse_logits_eval, label, coarse_threshold, min_pred_size)
            coarse_m = refined_m
        else:
            refined_m = compute_metrics(refined_logits_eval, label, threshold, min_pred_size)
            coarse_m = compute_metrics(coarse_logits_eval, label, coarse_threshold, min_pred_size)

        for thr, metric_sums in threshold_sums.items():
            thr_m = compute_metrics(coarse_logits_eval, label, thr, min_pred_size)
            _accumulate_metric_sums(metric_sums, thr_m)
        for thr, metric_sums in refined_threshold_sums.items():
            thr_m = compute_metrics(refined_logits_eval, label, thr, min_pred_size)
            _accumulate_metric_sums(metric_sums, thr_m)

        obj_m = compute_objectness_metrics(output.get("objectness_logit"), label)
        mod_m = compute_modality_weight_metrics(output.get("modality_weights"))
        n_b = max(refined_m["n"], 1)
        pos_b = refined_m["n_pos"]
        neg_b = refined_m["n_neg"]

        sums["loss"] += float(losses["total_loss"].item()) * n_b
        sums["objectness_loss"] += float(losses.get("objectness_loss", torch.zeros((), device=device)).item()) * n_b
        sums["objectness_acc"] += obj_m["objectness_acc"] * n_b
        if pos_b:
            sums["objectness_pos_prob"] += obj_m["objectness_pos_prob"] * pos_b
            sums["objectness_pos_n"] += pos_b
        if neg_b:
            sums["objectness_neg_prob"] += obj_m["objectness_neg_prob"] * neg_b
            sums["objectness_neg_n"] += neg_b
        sums["modality_weight_t2w"] += mod_m["modality_weight_t2w"] * n_b
        sums["modality_weight_adc"] += mod_m["modality_weight_adc"] * n_b
        sums["modality_weight_hbv"] += mod_m["modality_weight_hbv"] * n_b
        sums["refined_dice"] += refined_m["dice"] * n_b
        sums["global_tp"] += float(refined_m.get("global_tp", 0.0))
        sums["global_fp"] += float(refined_m.get("global_fp", 0.0))
        sums["global_fn"] += float(refined_m.get("global_fn", 0.0))
        sums["coarse_dice"] += coarse_m["dice"] * n_b
        sums["precision"] += refined_m["precision"] * n_b
        sums["recall"] += refined_m["recall"] * n_b
        sums["n"] += n_b
        if pos_b:
            sums["positive_case_dice"] += refined_m["positive_dice"] * pos_b
            sums["positive_precision"] += refined_m["positive_precision"] * pos_b
            sums["positive_recall"] += refined_m["positive_recall"] * pos_b
            sums["detection_rate"] += refined_m["detection_rate"] * pos_b
            sums["lesion_recall"] += refined_m["lesion_recall"] * pos_b
            sums["coarse_positive_dice"] += coarse_m["positive_dice"] * pos_b
            sums["coarse_positive_recall"] += coarse_m["positive_recall"] * pos_b
            sums["coarse_lesion_recall"] += coarse_m["lesion_recall"] * pos_b
            sums["pos_n"] += pos_b
        if neg_b:
            sums["fp_case_rate"] += refined_m["fp_case_rate"] * neg_b
            sums["negative_dice"] += refined_m["negative_dice"] * neg_b
            sums["neg_n"] += neg_b
        sums["fp_components_per_case"] += refined_m["fp_components_per_case"] * n_b
        sums["coarse_fp_components_per_case"] += coarse_m["fp_components_per_case"] * n_b
        progress.set_postfix(
            loss=f"{losses['total_loss'].item():.4f}",
            pos_dice=f"{refined_m['positive_dice']:.3f}",
            pos_rec=f"{refined_m['positive_recall']:.3f}",
        )

    n = max(sums["n"], 1)
    pos_n = max(sums["pos_n"], 1)
    neg_n = max(sums["neg_n"], 1)
    global_tp = float(sums.get("global_tp", 0.0))
    global_fp = float(sums.get("global_fp", 0.0))
    global_fn = float(sums.get("global_fn", 0.0))
    global_dice = 2.0 * global_tp / max(2.0 * global_tp + global_fp + global_fn, 1e-8)
    return {
        "loss": sums["loss"] / n,
        "refined_dice": sums["refined_dice"] / n,
        "global_dice": global_dice,
        "global_tp": global_tp,
        "global_fp": global_fp,
        "global_fn": global_fn,
        "coarse_dice": sums["coarse_dice"] / n,
        "precision": sums["precision"] / n,
        "recall": sums["recall"] / n,
        "positive_case_dice": sums["positive_case_dice"] / pos_n,
        "positive_precision": sums["positive_precision"] / pos_n,
        "positive_recall": sums["positive_recall"] / pos_n,
        "detection_rate": sums["detection_rate"] / pos_n,
        "fp_case_rate": sums["fp_case_rate"] / neg_n,
        "negative_dice": sums["negative_dice"] / neg_n,
        "coarse_positive_dice": sums["coarse_positive_dice"] / pos_n,
        "coarse_positive_recall": sums["coarse_positive_recall"] / pos_n,
        "lesion_recall": sums["lesion_recall"] / pos_n,
        "fp_components_per_case": sums["fp_components_per_case"] / n,
        "coarse_lesion_recall": sums["coarse_lesion_recall"] / pos_n,
        "coarse_fp_components_per_case": sums["coarse_fp_components_per_case"] / n,
        "objectness_loss": sums["objectness_loss"] / n,
        "objectness_acc": sums["objectness_acc"] / n,
        "objectness_pos_prob": sums["objectness_pos_prob"] / max(sums["objectness_pos_n"], 1),
        "objectness_neg_prob": sums["objectness_neg_prob"] / max(sums["objectness_neg_n"], 1),
        "modality_weight_t2w": sums["modality_weight_t2w"] / n,
        "modality_weight_adc": sums["modality_weight_adc"] / n,
        "modality_weight_hbv": sums["modality_weight_hbv"] / n,
        **select_global_threshold_sweep(threshold_sums, sweep_score_cfg),
        **select_global_refined_threshold_sweep(refined_threshold_sums, refined_sweep_score_cfg),
        "n_positive": sums["pos_n"],
        "n_negative": sums["neg_n"],
    }


def recall_safe_score(metrics: dict, min_recall: float) -> float:
    """Positive-case Dice penalized when positive recall is below target."""
    recall = float(metrics.get("positive_recall", 0.0))
    dice = float(metrics.get("positive_case_dice", 0.0))
    fp_rate = float(metrics.get("fp_case_rate", 0.0))
    if recall < min_recall:
        dice = dice - 2.0 * (min_recall - recall)
    return dice - 0.10 * fp_rate


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def save_if_best(name: str, value: float, best_values: dict, ckpt_dir: Path,
                 model, optimizer, scheduler, epoch: int, metrics: dict, cfg: dict) -> bool:
    """Save checkpoint if value is a new best. Returns True if saved."""
    if not math.isfinite(float(value)):
        return False
    if value > best_values.get(name, -float("inf")):
        best_values[name] = float(value)
        save_checkpoint(
            ckpt_dir / f"best_by_val_{name}.pth",
            model, optimizer, scheduler, epoch, metrics,
            metadata={"epoch": epoch, name: float(value), **{k: float(v) for k, v in metrics.items()}},
            config_snapshot=cfg,
        )
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Pretty logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, digits=4):
    """Format a float value, return 'n/a' for None/nan."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{float(v):.{digits}f}"



LOG_WIDTH = 140


def _log_rule(char: str = "-") -> None:
    print(char * LOG_WIDTH)


def _kv(label: str, value: object, width: int = 32) -> str:
    text = f"{label}: {value}"
    if len(text) > width - 1:
        text = text[: max(width - 4, 1)] + "..."
    return text.ljust(width)


def _cell(label: str, value: object, width: int = 18) -> str:
    text = f"{label} {_fmt(value) if isinstance(value, float) else value}"
    if len(text) > width - 1:
        text = text[: max(width - 4, 1)] + "..."
    return text.ljust(width)


def _metric_row(title: str, items: list[tuple[str, object, int | None]]) -> None:
    cells = []
    for label, value, digits in items:
        rendered = str(value) if digits is None else _maybe_metric(value, digits)
        cells.append(_cell(label, rendered, 18))
    print(f"  {title:<8}" + " | ".join(cells))


def _maybe_metric(value, digits=4):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def print_run_header(args, cfg, output_dir, n_total, n_train,
                     train_loader, val_loader, epochs, early_cfg):
    W = LOG_WIDTH
    fold_str = f"fold_{args.fold}" if args.fold is not None else "no-fold"
    exp = cfg.get("logging", {}).get("experiment_name", "pcasam3d")
    stage = cfg.get("training", {}).get("stage", "joint")
    data_cfg = cfg.get("data", {})
    opt = cfg.get("optimizer", {})
    model_cfg = cfg.get("model", {})

    print()
    print("=" * W)
    print("PCaSAM-3D-ProFound Training".center(W))
    print("=" * W)
    print("Run")
    print("  " + _kv("Experiment", f"{exp} [{fold_str}]", 58)
          + _kv("Stage", stage, 24)
          + _kv("Device", cfg.get("project", {}).get("device", "cuda"), 20))
    print("  " + _kv("Cases", f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}", 32)
          + _kv("Patch", data_cfg.get("patch_size", [128, 128, 128]), 32)
          + _kv("Batch", data_cfg.get("batch_size", 2), 20)
          + _kv("Workers", data_cfg.get("num_workers", 0), 20))
    print("  " + _kv("Params", f"{n_total:.1f}M total / {n_train:.1f}M trainable", 42)
          + _kv("Epochs", epochs, 20)
          + _kv("AMP", cfg.get("training", {}).get("amp", True), 18)
          + _kv("SAM frozen", model_cfg.get("freeze_sam_decoder", False), 24))
    print("  " + _kv("LR encoder", f"{opt.get('encoder_lr', 0):.1e}", 26)
          + _kv("LR bridge", f"{opt.get('bridge_lr', 5e-5):.1e}", 26)
          + _kv("LR decoder", f"{opt.get('decoder_lr', 1e-4):.1e}", 26)
          + _kv("Weight decay", f"{opt.get('weight_decay', 0):.1e}", 26))
    if early_cfg.get("enabled", False):
        print("  " + _kv("Early stop", f"{early_cfg.get('monitor')} | patience={early_cfg.get('patience')}", 84))
    print("  " + _kv("Output", output_dir, W - 4))
    print("=" * W)


def print_epoch_row(epoch, train_m, val_m, safe, lr, early_counter, early_patience, note="", stage="joint"):
    es_str = f"{early_counter}/{early_patience}" if early_counter > 0 else "fresh"
    saved_str = note.replace("best: ", "") if note else "-"
    _log_rule()
    print(f"Epoch {epoch:03d}  lr={lr:.3e}  early_stop={es_str}  saved={saved_str}")
    _metric_row("Loss", [
        ("train", train_m.get("loss"), 4),
        ("val", val_m.get("loss"), 4),
        ("refined", train_m.get("refined_loss"), 4),
        ("coarse", train_m.get("coarse_loss"), 4),
        ("obj", train_m.get("objectness_loss"), 4),
    ])
    if str(stage).lower() == "coarse":
        _metric_row("Goal", [
            ("val_dice", val_m.get("coarse_dice"), 4),
            ("pos_dice", val_m.get("positive_case_dice"), 4),
            ("pos_rec", val_m.get("coarse_positive_recall"), 3),
            ("lesion", val_m.get("coarse_lesion_recall"), 3),
            ("fp/case", val_m.get("coarse_fp_components_per_case"), 2),
        ])
        if float(val_m.get("threshold_sweep_best_threshold", 0.0) or 0.0) > 0:
            _metric_row("Sweep", [
                ("score", val_m.get("threshold_sweep_best_coarse_score"), 4),
                ("thr", val_m.get("threshold_sweep_best_threshold"), 3),
                ("dice", val_m.get("threshold_sweep_best_positive_case_dice"), 4),
                ("lesion", val_m.get("threshold_sweep_best_lesion_recall"), 3),
                ("fp/case", val_m.get("threshold_sweep_best_fp_components_per_case"), 2),
            ])
        _metric_row("Check", [
            ("global", val_m.get("global_dice"), 4),
            ("prec", val_m.get("positive_precision"), 3),
            ("recall", val_m.get("positive_recall"), 3),
            ("safe", safe, 4),
            ("train_d", train_m.get("coarse_dice"), 4),
        ])
        return
    _metric_row("Dice", [
        ("tr_ref", train_m.get("refined_dice"), 4),
        ("val_macro", val_m.get("positive_case_dice"), 4),
        ("global", val_m.get("global_dice"), 4),
        ("all_case", val_m.get("refined_dice"), 4),
        ("safe", safe, 4),
    ])
    _metric_row("Recall", [
        ("pos", val_m.get("positive_recall"), 3),
        ("lesion", val_m.get("lesion_recall"), 3),
        ("detect", val_m.get("detection_rate"), 3),
        ("precision", val_m.get("positive_precision"), 3),
        ("fp/case", val_m.get("fp_components_per_case"), 2),
    ])
    _metric_row("Coarse", [
        ("tr_dice", train_m.get("coarse_dice"), 4),
        ("val_dice", val_m.get("coarse_dice"), 4),
        ("pos_rec", val_m.get("coarse_positive_recall"), 3),
        ("lesion", val_m.get("coarse_lesion_recall"), 3),
        ("fp/case", val_m.get("coarse_fp_components_per_case"), 2),
    ])
    if float(val_m.get("threshold_sweep_best_threshold", 0.0) or 0.0) > 0:
        _metric_row("C-Sweep", [
            ("score", val_m.get("threshold_sweep_best_coarse_score"), 4),
            ("thr", val_m.get("threshold_sweep_best_threshold"), 3),
            ("dice", val_m.get("threshold_sweep_best_positive_case_dice"), 4),
            ("global", val_m.get("threshold_sweep_best_global_dice"), 4),
            ("lesion", val_m.get("threshold_sweep_best_lesion_recall"), 3),
            ("fp/case", val_m.get("threshold_sweep_best_fp_components_per_case"), 2),
        ])
    if float(val_m.get("refined_sweep_best_threshold", 0.0) or 0.0) > 0:
        _metric_row("R-Sweep", [
            ("score", val_m.get("refined_sweep_best_score"), 4),
            ("thr", val_m.get("refined_sweep_best_threshold"), 3),
            ("dice", val_m.get("refined_sweep_best_positive_case_dice"), 4),
            ("global", val_m.get("refined_sweep_best_global_dice"), 4),
            ("prec", val_m.get("refined_sweep_best_positive_precision"), 3),
        ])
        _metric_row("R-Extra", [
            ("lesion", val_m.get("refined_sweep_best_lesion_recall"), 3),
            ("fp/case", val_m.get("refined_sweep_best_fp_components_per_case"), 2),
        ])


def print_best_saved(name: str, value: float, epoch: int, prev_value: float | None, ckpt_dir: Path):
    fname = f"best_by_val_{name}.pth"
    if prev_value is not None and prev_value != value:
        delta = value - prev_value
        sign = "+" if delta > 0 else ""
        delta_str = f"delta={sign}{delta:.4f}"
    else:
        delta_str = "new"
    print(f"    saved {fname:<48} value={value:.4f}  ep={epoch:03d}  {delta_str}")


def print_early_stop(epoch, monitor, counter, patience, best):
    _log_rule()
    print(f"Early stop at epoch {epoch}: {monitor} best={best:.4f}, no improvement for {counter}/{patience} epochs")


def print_run_footer(best_values, elapsed):
    _log_rule("=")
    print("Training complete")

    def row(label, value):
        print(f"  {label:<36} {value}")

    row("Best val_positive_case_dice",    _fmt(best_values.get("positive_case_dice")))
    row("Best val_recall_safe_dice",      _fmt(best_values.get("recall_safe_dice")))
    row("Best val_positive_recall",       _fmt(best_values.get("positive_recall")))
    row("Best val_lesion_recall",         _fmt(best_values.get("lesion_recall")))
    row("Best val_coarse_lesion_recall",  _fmt(best_values.get("coarse_lesion_recall")))
    row("Best val_threshold_sweep_score", _fmt(best_values.get("threshold_sweep_coarse_score")))
    row("Best val_refined_sweep_score",   _fmt(best_values.get("refined_sweep_score")))
    row("Best val_refined_dice",          _fmt(best_values.get("refined_dice")))
    row("Elapsed",                        f"{elapsed / 3600:.2f} h")
    _log_rule("=")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("PCaSAM-3D-ProFound Training")
    parser.add_argument("--config", default="configs/train_pcasam3d_profound.yaml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))

    # Output directory
    exp_name = args.exp_name or cfg.get("logging", {}).get("experiment_name", "pcasam3d_profound")
    output_root = Path(cfg.get("logging", {}).get("output_root", "outputs"))
    if args.fold is not None:
        output_dir = output_root / exp_name / f"fold_{args.fold}"
    else:
        output_dir = output_root / exp_name
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg.get("project", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )

    # ─── Model ───
    model = build_pcasam3d_profound(cfg).to(device)
    resume_from = cfg.get("training", {}).get("resume_from")
    if resume_from:
        state = load_checkpoint(resume_from, model, map_location=device)
        print(f"[Checkpoint] Loaded model weights from {resume_from} (epoch={state.get('epoch', 'n/a')})")
    stage = apply_training_stage(model, cfg)
    print(f"[Training] stage={stage}")
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # ─── Loss ───
    loss_fn = build_pcasam3d_loss(cfg)

    # ─── Optimizer ───
    optimizer = build_optimizer(model, cfg)
    sched_cfg = cfg.get("scheduler", {})
    epochs = int(cfg["training"].get("epochs", 100))
    if sched_cfg.get("name", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(sched_cfg.get("min_lr", 1e-6))
        )
    else:
        scheduler = None

    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    )

    # ─── Data ───
    train_loader, val_loader = build_loaders(cfg, args)

    # ─── Training Loop ───
    rows: list[dict] = []
    best_values: dict[str, float] = {}
    min_recall = float(cfg.get("selection", {}).get("min_recall", 0.60))

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_monitor = str(early_cfg.get("monitor", "val_recall_safe_dice"))
    early_mode = str(early_cfg.get("mode", "max")).lower()
    early_patience = int(early_cfg.get("patience", 15))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_best = float("inf") if early_mode == "min" else -float("inf")
    early_counter = 0

    # Print run header
    print_run_header(args, cfg, output_dir, n_total, n_train,
                     train_loader, val_loader, epochs, early_cfg)

    start_time = time.time()
    W = 78  # box width

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler,
                                         loss_fn, device, cfg, epoch)
        val_metrics = validate(model, val_loader, loss_fn, device, cfg, epoch)
        if scheduler is not None:
            scheduler.step()

        lr = max(g["lr"] for g in optimizer.param_groups)
        safe = recall_safe_score(val_metrics, min_recall)

        row = {
            "epoch": epoch,
            "lr": lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_recall_safe_dice": safe,
        }

        # Early stopping check
        monitor_val = row.get(early_monitor, float("nan"))
        if early_enabled and math.isfinite(monitor_val):
            improved = (
                (monitor_val > early_best + early_min_delta) if early_mode == "max"
                else (monitor_val < early_best - early_min_delta)
            )
            if improved:
                early_best = monitor_val
                early_counter = 0
            else:
                early_counter += 1
        row["early_stopping_counter"] = early_counter
        row["early_stopping_best"] = early_best
        rows.append(row)
        write_csv(log_dir / "train_log.csv", rows)

        # Save best checkpoints and collect notes
        meta = {**val_metrics, "epoch": epoch, "lr": lr, "recall_safe_dice": safe}
        saved_notes = []
        monitors = [
            ("refined_dice",       val_metrics["refined_dice"]),
            ("positive_case_dice", val_metrics["positive_case_dice"]),
            ("global_dice",        val_metrics.get("global_dice", 0.0)),
            ("recall_safe_dice",   safe),
            ("positive_recall",    val_metrics["positive_recall"]),
            ("coarse_positive_recall", val_metrics["coarse_positive_recall"]),
            ("lesion_recall", val_metrics["lesion_recall"]),
            ("coarse_lesion_recall", val_metrics["coarse_lesion_recall"]),
            ("threshold_sweep_coarse_score", val_metrics["threshold_sweep_best_coarse_score"]),
            ("refined_sweep_score", val_metrics.get("refined_sweep_best_score", 0.0)),
        ]
        for name, value in monitors:
            if save_if_best(name, value, best_values, ckpt_dir,
                            model, optimizer, scheduler, epoch, meta, cfg):
                saved_notes.append(name)

        # Print epoch row
        note = ""
        if saved_notes:
            note = "best: " + ", ".join(saved_notes)
        print_epoch_row(epoch, train_metrics, val_metrics, safe, lr, early_counter, early_patience, note, stage=cfg.get("training", {}).get("stage", "joint"))

        # Print saved checkpoint lines with delta from previous best
        for name in saved_notes:
            # Find previous best for comparison
            prev_val = None
            for prev_row in rows[:-1]:
                prev_key = f"val_{name}"
                if prev_key in prev_row and prev_row[prev_key] is not None:
                    try:
                        candidate = float(prev_row[prev_key])
                        if math.isfinite(candidate):
                            if prev_val is None or candidate > prev_val:
                                prev_val = candidate
                    except (ValueError, TypeError):
                        pass
            print_best_saved(name, best_values[name], epoch, prev_val, ckpt_dir)

        # Save last checkpoint periodically
        save_every = int(cfg["training"].get("save_every", 5))
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(
                ckpt_dir / "last.pth",
                model, optimizer, scheduler, epoch, val_metrics,
                metadata=meta, config_snapshot=cfg,
                extra_state={"best_values": dict(best_values),
                             "early_stopping_best": early_best,
                             "early_stopping_counter": early_counter},
            )
            print(f"Saved last.pth at epoch {epoch}")

        if early_enabled and early_counter >= early_patience:
            print_early_stop(epoch, early_monitor, early_counter, early_patience, early_best)
            # Save final checkpoint on early stop
            save_checkpoint(
                ckpt_dir / "last.pth",
                model, optimizer, scheduler, epoch, val_metrics,
                metadata=meta, config_snapshot=cfg,
                extra_state={"best_values": dict(best_values),
                             "early_stopping_best": early_best,
                             "early_stopping_counter": early_counter},
            )
            break

    elapsed = time.time() - start_time
    print_run_footer(best_values, elapsed)


if __name__ == "__main__":
    main()
