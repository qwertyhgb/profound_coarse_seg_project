#!/usr/bin/env python
"""Phase 1: Train SAM-Med3D Stage-2 baseline on PI-CAI.

This script fine-tunes SAM-Med3D (with multi-channel patch_embed adapter)
using Stage-1 coarse prompts. It is the baseline for Phase 2 encoder replacement.

Usage:
    /root/anaconda3/envs/lm/bin/python scripts/train_stage2_sam_med3d.py \
        --config configs/train_stage2_sam_med3d_v1.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.stage2_sam_med3d_dataset import (
    Stage2SAMMed3DDataset, stage2_sam_med3d_collate_fn,
)
from src.models.sam_med3d_integration import build_sam_med3d_stage2
from src.utils.checkpoint import save_checkpoint
from src.utils.config_utils import load_config
from src.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def build_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    ds = Stage2SAMMed3DDataset(
        processed_root=data["processed_root"],
        prompt_csv=data[f"{split}_prompts"],
        crop_margin_ratio=data.get("crop_margin_ratio", 1.5),
        target_size=tuple(data.get("target_size", [128, 128, 128])),
        normalize=data.get("normalize", "channelwise_nonzero"),
        max_prompts=data.get(f"max_{split}_prompts"),
        positive_only=data.get(f"{split}_positive_only", False),
        negative_ratio=data.get(f"{split}_negative_ratio"),
        point_jitter_voxels=data.get(f"point_jitter_voxels_{split}", 0),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    print(
        f"[{split}] prompts: {len(ds)} "
        f"(positive_only={data.get(f'{split}_positive_only', False)}, "
        f"negative_ratio={data.get(f'{split}_negative_ratio')})"
    )
    return DataLoader(
        ds,
        batch_size=int(data.get("batch_size", 2)),
        shuffle=shuffle,
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=bool(data.get("pin_memory", True)),
        collate_fn=stage2_sam_med3d_collate_fn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict):
    """Two parameter groups: image_encoder (lower lr) and prompt+mask decoder."""
    opt = cfg["optimizer"]
    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("image_encoder."):
            encoder_params.append(p)
        else:
            head_params.append(p)
    groups = []
    if encoder_params:
        groups.append({
            "params": encoder_params,
            "lr": float(opt.get("encoder_lr", 1e-5)),
            "initial_lr": float(opt.get("encoder_lr", 1e-5)),
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": float(opt.get("head_lr", 1e-4)),
            "initial_lr": float(opt.get("head_lr", 1e-4)),
        })
    if not groups:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(groups, weight_decay=float(opt.get("weight_decay", 1e-4)))


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────────────────────────

def forward_batch(model, batch, device):
    image = batch["image"].to(device, non_blocking=True)
    point_coords = batch["point_coords"].to(device, non_blocking=True)
    point_labels = batch["point_label"].to(device, non_blocking=True)
    masks_logits, iou_pred = model(image, point_coords, point_labels)
    return masks_logits, iou_pred


def compute_dice(pred_mask: torch.Tensor, label: torch.Tensor) -> tuple[float, float, float]:
    """Compute Dice / Precision / Recall on binary tensors (per-batch averaged)."""
    pred = pred_mask.float()
    gt = (label > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    tp = (pred * gt).sum(dim=dims)
    fp = (pred * (1 - gt)).sum(dim=dims)
    fn = ((1 - pred) * gt).sum(dim=dims)
    dice = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean().item()
    prec = (tp / (tp + fp + 1e-6)).mean().item()
    rec = (tp / (tp + fn + 1e-6)).mean().item()
    return dice, prec, rec


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg, epoch):
    model.train()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    sums = {"loss": 0.0, "dice": 0.0, "n": 0}
    progress = tqdm(loader, desc=f"Train {epoch}", dynamic_ncols=True, leave=False)
    for batch in progress:
        label = batch["label"].to(device, non_blocking=True)
        objectness = batch["objectness"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            masks_logits, _ = forward_batch(model, batch, device)
            # Skip mask loss for negative-overlap samples (they have no GT to learn)
            # Use objectness as a soft mask: full weight for positives, low for negatives
            pos_mask = objectness > 0.5
            if pos_mask.any():
                pos_loss = loss_fn(masks_logits[pos_mask], label[pos_mask])
            else:
                pos_loss = masks_logits.new_tensor(0.0)
            if (~pos_mask).any():
                neg_loss = loss_fn(masks_logits[~pos_mask], label[~pos_mask])
            else:
                neg_loss = masks_logits.new_tensor(0.0)
            loss = pos_loss + 0.05 * neg_loss
        scaler.scale(loss).backward()
        clip = cfg["training"].get("grad_clip_norm")
        if clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        scaler.step(optimizer); scaler.update()

        with torch.no_grad():
            preds = (torch.sigmoid(masks_logits.detach()) > 0.5).float()
            dice = compute_dice(preds, label)[0]
        sums["loss"] += float(loss.detach().item()); sums["dice"] += float(dice); sums["n"] += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.3f}",
                             avg=f"{sums['loss']/sums['n']:.4f}")
    return {"loss": sums["loss"] / max(sums["n"], 1), "dice": sums["dice"] / max(sums["n"], 1)}


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg, epoch):
    model.eval()
    sums = {"loss": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0,
            "positive_dice": 0.0, "n": 0, "pos_n": 0}
    progress = tqdm(loader, desc=f"Val   {epoch}", dynamic_ncols=True, leave=False)
    for batch in progress:
        label = batch["label"].to(device, non_blocking=True)
        objectness = batch["objectness"].to(device, non_blocking=True)
        masks_logits, _ = forward_batch(model, batch, device)
        loss = loss_fn(masks_logits, label)
        preds = (torch.sigmoid(masks_logits) > 0.5).float()
        d, p, r = compute_dice(preds, label)
        sums["loss"] += float(loss.item()); sums["dice"] += d
        sums["precision"] += p; sums["recall"] += r; sums["n"] += 1

        # Positive case dice (samples whose objectness=1, i.e., overlaps GT)
        pos_idx = objectness > 0.5
        if pos_idx.any():
            pd, _, _ = compute_dice(preds[pos_idx], label[pos_idx])
            sums["positive_dice"] += pd; sums["pos_n"] += 1

        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.3f}",
                             avg=f"{sums['dice']/sums['n']:.3f}")
    return {
        "loss": sums["loss"] / max(sums["n"], 1),
        "dice": sums["dice"] / max(sums["n"], 1),
        "precision": sums["precision"] / max(sums["n"], 1),
        "recall": sums["recall"] / max(sums["n"], 1),
        "positive_case_dice": sums["positive_dice"] / max(sums["pos_n"], 1),
    }


def recall_safe_score(metrics: dict, min_recall: float) -> float:
    recall = float(metrics.get("recall", 0.0))
    dice = float(metrics.get("dice", 0.0))
    if recall < min_recall:
        return dice - 2.0 * (min_recall - recall)
    return dice


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
        w.writeheader(); w.writerows(rows)


def save_if_best(name: str, value: float, best_values: dict, ckpt_dir: Path,
                 model, optimizer, scheduler, epoch: int, metrics: dict, cfg: dict):
    if not math.isfinite(float(value)):
        return
    if value > best_values.get(name, -float("inf")):
        best_values[name] = float(value)
        save_checkpoint(
            ckpt_dir / f"best_by_val_{name}.pth",
            model, optimizer, scheduler, epoch, metrics,
            metadata={"epoch": epoch, name: float(value), **{k: float(v) for k, v in metrics.items()}},
            config_snapshot=cfg,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_stage2_sam_med3d_v1.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))

    output_root = Path(cfg["logging"]["output_root"])
    ckpt_dir = output_root / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("project", {}).get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")

    # Model
    model = build_sam_med3d_stage2(cfg["model"]).to(device)
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: {n_total:.1f}M total, {n_train:.1f}M trainable")

    # Loss
    loss_fn = DiceCELoss(sigmoid=True, include_background=cfg["loss"].get("include_background", True))

    # Optimizer
    optimizer = build_optimizer(model, cfg)
    sched = cfg.get("scheduler", {})
    if sched.get("name", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["training"].get("epochs", 50)),
            eta_min=float(sched.get("min_lr", 1e-6)),
        )
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["training"].get("amp", True))
                                    and device.type == "cuda")

    # Data
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)

    # Training loop
    rows: list[dict] = []
    best_values: dict[str, float] = {}
    min_recall = float(cfg.get("selection", {}).get("min_recall", 0.70))

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_monitor = str(early_cfg.get("monitor", "val_recall_safe_dice"))
    early_mode = str(early_cfg.get("mode", "max")).lower()
    early_patience = int(early_cfg.get("patience", 10))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_best = float("inf") if early_mode == "min" else -float("inf")
    early_counter = 0

    epochs = int(cfg["training"].get("epochs", 50))
    print(f"Starting training for {epochs} epochs")
    print("=" * 100)

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

        # Early stopping
        monitor_val = row.get(early_monitor, float("nan"))
        if early_enabled and math.isfinite(monitor_val):
            improved = (monitor_val > early_best + early_min_delta) if early_mode == "max" \
                       else (monitor_val < early_best - early_min_delta)
            if improved:
                early_best = monitor_val; early_counter = 0
            else:
                early_counter += 1
        row["early_stopping_counter"] = early_counter
        row["early_stopping_best"] = early_best
        rows.append(row)
        write_csv(log_dir / "train_log.csv", rows)

        print(
            f"E{epoch:03d} train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} "
            f"posDice={val_metrics['positive_case_dice']:.4f} "
            f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} "
            f"safe={safe:.4f} | lr={lr:.2e} es={early_counter}/{early_patience}"
        )

        # Save best checkpoints
        meta = {**val_metrics, "epoch": epoch, "lr": lr, "recall_safe_dice": safe}
        save_if_best("dice", val_metrics["dice"], best_values, ckpt_dir,
                     model, optimizer, scheduler, epoch, meta, cfg)
        save_if_best("positive_case_dice", val_metrics["positive_case_dice"], best_values,
                     ckpt_dir, model, optimizer, scheduler, epoch, meta, cfg)
        save_if_best("recall_safe_dice", safe, best_values, ckpt_dir,
                     model, optimizer, scheduler, epoch, meta, cfg)

        # Save last checkpoint periodically
        if epoch % int(cfg["training"].get("save_every", 5)) == 0 or epoch == epochs:
            save_checkpoint(
                ckpt_dir / "last.pth",
                model, optimizer, scheduler, epoch, val_metrics,
                metadata=meta, config_snapshot=cfg,
                extra_state={"best_values": dict(best_values),
                             "early_stopping_best": early_best,
                             "early_stopping_counter": early_counter},
            )

        if early_enabled and early_counter >= early_patience:
            print(f"Early stopping at epoch {epoch}: {early_monitor} no improvement for "
                  f"{early_counter} epochs (best={early_best:.4f}).")
            break

    print("=" * 100)
    print("Training complete.")
    print(f"  Best val_dice              : {best_values.get('dice', float('nan')):.4f}")
    print(f"  Best val_positive_case_dice: {best_values.get('positive_case_dice', float('nan')):.4f}")
    print(f"  Best val_recall_safe_dice  : {best_values.get('recall_safe_dice', float('nan')):.4f}")
    print(f"Outputs: {output_root}")


if __name__ == "__main__":
    main()
