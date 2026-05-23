#!/usr/bin/env python
"""Test a checkpoint on a test split."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, torch
from torch.utils.data import DataLoader
from src.utils.config_utils import load_config
from src.models.build_model import build_model
from src.losses.build_loss import build_loss
from src.datasets.picai_npz_dataset import PICAINPZDataset
from src.datasets.collate import picai_collate_fn
from src.trainers.evaluator import Evaluator
from src.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_profound_coarse.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_by_val_dice.pth")
    parser.add_argument("--split", default=None, help="Override test split file, e.g. data/splits/5fold/fold_0/test.txt")
    parser.add_argument("--save-csv", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    loss = build_loss(cfg["loss"]).to(device)
    split = args.split or cfg["data"]["test_split"]
    ds = PICAINPZDataset(cfg["data"]["processed_root"], split, mode="test")
    save_csv = args.save_csv or str(Path(args.checkpoint).parent.parent / "logs" / "test_cases.csv")
    metrics_cfg = dict(cfg.get("metrics", {}))
    if isinstance(cfg.get("coarse_score"), dict):
        metrics_cfg["coarse_score"] = cfg["coarse_score"]
    if "threshold_sweep" in cfg:
        metrics_cfg["threshold_sweep"] = cfg["threshold_sweep"]
    metrics = Evaluator(model, loss, device, cfg["inference"], metrics_cfg).evaluate(
        DataLoader(ds, batch_size=1, collate_fn=picai_collate_fn), save_csv=save_csv
    )
    print(metrics)


if __name__ == "__main__":
    main()
