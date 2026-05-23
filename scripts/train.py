#!/usr/bin/env python
"""Train Stage-1 ProFound coarse lesion segmentation."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from src.utils.config_utils import load_config
from src.utils.seed import set_seed
from src.trainers.trainer import Trainer


def apply_cli_overrides(cfg, args):
    """Apply command line overrides and build outputs/<experiment>/fold_n."""
    if args.train_split:
        cfg["data"]["train_split"] = args.train_split
    if args.val_split:
        cfg["data"]["val_split"] = args.val_split

    logging_cfg = cfg.setdefault("logging", {})
    base_output = args.output_root or logging_cfg.get("output_root", "outputs")
    exp_name = args.exp_name or logging_cfg.get("experiment_name") or cfg.get("project", {}).get("name", "experiment")

    output_root = Path(base_output) / exp_name
    if args.fold is not None:
        cfg.setdefault("project", {})["fold"] = args.fold
        output_root = output_root / f"fold_{args.fold}"
    if args.resume:
        cfg.setdefault("training", {})["resume_from"] = args.resume

    logging_cfg["output_root"] = str(output_root)
    logging_cfg["experiment_name"] = exp_name
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_profound_coarse.yaml")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--output-root", default=None, help="Base output directory, default from config: outputs")
    parser.add_argument("--exp-name", default=None, help="Experiment name under output root, e.g. recall_ftversky")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Resume from a full training checkpoint")
    args = parser.parse_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    print(f"Experiment output: {cfg['logging']['output_root']}")
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    project_cfg = cfg.get("project", {})
    import torch
    if bool(project_cfg.get("cudnn_benchmark", False)):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision(str(project_cfg.get("float32_matmul_precision", "high")))
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
