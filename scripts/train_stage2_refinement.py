#!/usr/bin/env python
"""Train Stage-2 prompt-conditioned refinement baseline."""
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
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.collate import stage2_prompt_collate_fn
from src.datasets.stage2_prompt_dataset import Stage2PromptDataset
from src.losses.build_loss import build_loss
from src.metrics.segmentation_metrics import SegmentationMetricAccumulator
from src.models.build_model import build_model
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config_utils import load_config
from src.utils.seed import set_seed


def build_loader(cfg: dict, split_key: str, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    ds = Stage2PromptDataset(
        processed_root=data["processed_root"],
        prompt_csv=data[f"{split_key}_prompts"],
        coarse_pred_root=data[f"{split_key}_coarse_pred_root"],
        patch_size=data.get("patch_size", [64, 128, 128]),
        bbox_margin=data.get("bbox_margin", [4, 12, 12]),
        point_sigma=data.get("point_sigma", 3.0),
        max_prompts=data.get(f"max_{split_key}_prompts"),
        allow_missing_coarse=data.get("allow_missing_coarse", False),
        use_overlaps_gt_sampling=data.get("use_overlaps_gt_sampling", False),
        positive_only=data.get(f"{split_key}_positive_only", False),
        negative_ratio=data.get(f"{split_key}_negative_ratio"),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    print(
        f"{split_key} prompts: {len(ds)} "
        f"(positive_only={data.get(f'{split_key}_positive_only', False)}, "
        f"negative_ratio={data.get(f'{split_key}_negative_ratio')})"
    )
    return DataLoader(
        ds,
        batch_size=int(data.get("batch_size", 2)),
        shuffle=shuffle,
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=bool(data.get("pin_memory", True)),
        collate_fn=stage2_prompt_collate_fn,
    )


def forward_model(model, batch, device) -> dict[str, torch.Tensor]:
    out = model(
        batch["image"].to(device, non_blocking=True),
        batch["coarse_prob"].to(device, non_blocking=True),
        batch["box_prior"].to(device, non_blocking=True),
        batch["point_prior"].to(device, non_blocking=True),
    )
    if isinstance(out, dict):
        return out
    return {"logits": out}


def compute_stage2_loss(out: dict[str, torch.Tensor], label: torch.Tensor, objectness: torch.Tensor, mask_loss_fn, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task Stage-2 loss: mask refinement plus proposal objectness."""
    loss_cfg = cfg.get("stage2_loss", {})
    logits = out["logits"]
    pos_mask = objectness > 0.5
    neg_mask = ~pos_mask
    total = logits.new_tensor(0.0)
    parts: dict[str, float] = {}

    positive_mask_weight = float(loss_cfg.get("positive_mask_weight", 1.0))
    negative_mask_weight = float(loss_cfg.get("negative_mask_weight", 0.05))
    if pos_mask.any():
        pos_loss = mask_loss_fn(logits[pos_mask], label[pos_mask])
        total = total + positive_mask_weight * pos_loss
        parts["mask_pos_loss"] = float(pos_loss.detach().item())
    else:
        parts["mask_pos_loss"] = 0.0
    if negative_mask_weight > 0 and neg_mask.any():
        neg_loss = mask_loss_fn(logits[neg_mask], label[neg_mask])
        total = total + negative_mask_weight * neg_loss
        parts["mask_neg_loss"] = float(neg_loss.detach().item())
    else:
        parts["mask_neg_loss"] = 0.0

    objectness_weight = float(loss_cfg.get("objectness_weight", 0.25))
    if objectness_weight > 0 and "objectness_logits" in out:
        pos_weight_value = loss_cfg.get("objectness_pos_weight")
        pos_weight = torch.tensor([float(pos_weight_value)], device=logits.device) if pos_weight_value is not None else None
        obj_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(out["objectness_logits"], objectness)
        total = total + objectness_weight * obj_loss
        parts["objectness_loss"] = float(obj_loss.detach().item())
    else:
        parts["objectness_loss"] = 0.0
    parts["loss"] = float(total.detach().item())
    return total, parts


def objectness_metrics(out: dict[str, torch.Tensor], objectness: torch.Tensor) -> dict[str, float]:
    if "objectness_logits" not in out:
        return {}
    probs = torch.sigmoid(out["objectness_logits"].detach())
    pred = probs >= 0.5
    target = objectness.detach() >= 0.5
    tp = float((pred & target).sum().item())
    fp = float((pred & ~target).sum().item())
    tn = float((~pred & ~target).sum().item())
    fn = float((~pred & target).sum().item())
    return {
        "objectness_acc": (tp + tn) / max(tp + fp + tn + fn, 1.0),
        "objectness_precision": tp / max(tp + fp, 1e-8),
        "objectness_recall": tp / max(tp + fn, 1e-8),
        "objectness_prob_mean": float(probs.mean().item()),
    }


def train_one_epoch(model, mask_loss_fn, optimizer, scaler, loader, device, cfg, epoch: int) -> dict[str, float]:
    model.train()
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    sums = {"loss": 0.0, "mask_pos_loss": 0.0, "mask_neg_loss": 0.0, "objectness_loss": 0.0}
    n = 0
    progress = tqdm(loader, desc=f"Stage2 train {epoch}", dynamic_ncols=True, leave=False)
    for batch in progress:
        label = batch["label"].to(device, non_blocking=True)
        objectness = batch["objectness_label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = forward_model(model, batch, device)
            loss, parts = compute_stage2_loss(out, label, objectness, mask_loss_fn, cfg)
        scaler.scale(loss).backward()
        clip = cfg["training"].get("grad_clip_norm")
        if clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        scaler.step(optimizer); scaler.update()
        for key in sums:
            sums[key] += float(parts.get(key, 0.0))
        n += 1
        progress.set_postfix(loss=f"{parts['loss']:.4f}", avg=f"{sums['loss'] / max(n, 1):.4f}")
    return {key: value / max(n, 1) for key, value in sums.items()}


@torch.no_grad()
def validate(model, mask_loss_fn, loader, device, cfg, epoch: int) -> dict[str, float]:
    model.eval()
    threshold = float(cfg.get("metrics", {}).get("threshold", 0.5))
    acc = SegmentationMetricAccumulator(threshold=threshold)
    sums = {"loss": 0.0, "mask_pos_loss": 0.0, "mask_neg_loss": 0.0, "objectness_loss": 0.0}
    obj_sums: dict[str, float] = {}
    n = 0
    progress = tqdm(loader, desc=f"Stage2 val {epoch}", dynamic_ncols=True, leave=False)
    for batch in progress:
        label = batch["label"].to(device, non_blocking=True)
        objectness = batch["objectness_label"].to(device, non_blocking=True)
        out = forward_model(model, batch, device)
        loss, parts = compute_stage2_loss(out, label, objectness, mask_loss_fn, cfg)
        for key in sums:
            sums[key] += float(parts.get(key, 0.0))
        for key, value in objectness_metrics(out, objectness).items():
            obj_sums[key] = obj_sums.get(key, 0.0) + value
        n += 1
        acc.update(out["logits"], label)
        progress.set_postfix(loss=f"{float(loss.item()):.4f}", avg=f"{sums['loss'] / max(n, 1):.4f}")
    metrics = {key: value / max(n, 1) for key, value in sums.items()}
    metrics.update(acc.compute())
    metrics.update({key: value / max(n, 1) for key, value in obj_sums.items()})
    return metrics


def recall_safe_score(metrics: dict[str, float], min_recall: float) -> float:
    recall = float(metrics.get("recall", 0.0))
    dice = float(metrics.get("dice", 0.0))
    if recall < min_recall:
        return dice - 2.0 * (min_recall - recall)
    return dice


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def save_if_best(name: str, value: float, best_values: dict[str, float], ckpt_path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, metadata: dict, cfg: dict) -> None:
    if not math.isfinite(float(value)):
        return
    if value > best_values.get(name, -float("inf")):
        best_values[name] = float(value)
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, metrics, metadata=metadata, config_snapshot=cfg)


def _monitor_improved(value: float, best: float, mode: str, min_delta: float) -> bool:
    """Return True when an early-stopping monitor improved enough."""
    if mode == "min":
        return value < best - min_delta
    return value > best + min_delta


def _resolve_monitor_value(row: dict, monitor: str) -> float:
    """Resolve monitor names such as val_dice or val_recall_safe_dice."""
    if monitor in row:
        return float(row[monitor])
    raise KeyError(f"Early stopping monitor '{monitor}' is not present in the Stage-2 log row")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_stage2_refinement.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    if args.output_root:
        cfg.setdefault("logging", {})["output_root"] = args.output_root
    output_root = Path(cfg.get("logging", {}).get("output_root", "outputs/stage2_refinement/fold_0"))
    ckpt_dir = output_root / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"; log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    mask_loss_fn = build_loss(cfg["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"].get("lr", 1e-4)), weight_decay=float(cfg["optimizer"].get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["training"].get("epochs", 100)), eta_min=float(cfg.get("scheduler", {}).get("min_lr", 1e-6))
    ) if cfg.get("scheduler", {}).get("name", "cosine") != "none" else None
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["training"].get("amp", True)) and device.type == "cuda")

    start_epoch = 1
    best_values = {"dice": -1.0, "positive_case_dice": -1.0, "recall_safe_dice": -1.0}
    resume = args.resume or cfg.get("training", {}).get("resume_from")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(state.get("epoch", 0)) + 1
        best_values.update({k: float(v) for k, v in state.get("best_values", {}).items()})

    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    rows = []
    min_recall = float(cfg.get("selection", {}).get("min_recall", 0.60))
    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_monitor = str(early_cfg.get("monitor", "val_recall_safe_dice"))
    early_mode = str(early_cfg.get("mode", "max")).lower()
    early_patience = int(early_cfg.get("patience", 15))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_best = float("inf") if early_mode == "min" else -float("inf")
    early_counter = 0
    for epoch in range(start_epoch, int(cfg["training"].get("epochs", 100)) + 1):
        train_metrics = train_one_epoch(model, mask_loss_fn, optimizer, scaler, train_loader, device, cfg, epoch)
        val_metrics = validate(model, mask_loss_fn, val_loader, device, cfg, epoch)
        if scheduler is not None:
            scheduler.step()
        lr = max(float(g["lr"]) for g in optimizer.param_groups)
        safe_score = recall_safe_score(val_metrics, min_recall=min_recall)
        row = {
            "epoch": epoch,
            "learning_rate": lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_recall_safe_dice": safe_score,
        }
        monitor_value = _resolve_monitor_value(row, early_monitor) if early_enabled else float("nan")
        if early_enabled and _monitor_improved(monitor_value, early_best, early_mode, early_min_delta):
            early_best = monitor_value
            early_counter = 0
        elif early_enabled:
            early_counter += 1
        row["early_stopping_monitor"] = early_monitor if early_enabled else ""
        row["early_stopping_best"] = early_best if early_enabled else ""
        row["early_stopping_counter"] = early_counter if early_enabled else 0
        rows.append(row)
        write_csv(log_dir / "train_stage2_log.csv", rows)
        print(
            f"E{epoch:03d} train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} posDice={val_metrics['positive_case_dice']:.4f} "
            f"prec={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} "
            f"objAcc={val_metrics.get('objectness_acc', float('nan')):.4f} safe={safe_score:.4f} "
            f"es={early_counter}/{early_patience if early_enabled else 0} lr={lr:.6g}"
        )
        metadata = {
            "epoch": epoch,
            "val_dice": val_metrics["dice"],
            "val_positive_case_dice": val_metrics["positive_case_dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_recall_safe_dice": safe_score,
            "learning_rate": lr,
        }
        save_if_best("dice", val_metrics["dice"], best_values, ckpt_dir / "best_by_val_dice.pth", model, optimizer, scheduler, epoch, val_metrics, metadata, cfg)
        save_if_best("positive_case_dice", val_metrics["positive_case_dice"], best_values, ckpt_dir / "best_by_val_positive_case_dice.pth", model, optimizer, scheduler, epoch, val_metrics, metadata, cfg)
        save_if_best("recall_safe_dice", safe_score, best_values, ckpt_dir / "best_by_val_recall_safe_dice.pth", model, optimizer, scheduler, epoch, val_metrics, metadata, cfg)
        extra_state = {"best_values": dict(best_values), "early_stopping_best": early_best, "early_stopping_counter": early_counter}
        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scheduler, epoch, val_metrics, metadata=metadata, config_snapshot=cfg, extra_state=extra_state)
        if early_enabled and early_counter >= early_patience:
            print(f"Stage2 early stopping at epoch {epoch}: {early_monitor} did not improve for {early_counter} epochs; best={early_best:.6f}")
            break


if __name__ == "__main__":
    main()
