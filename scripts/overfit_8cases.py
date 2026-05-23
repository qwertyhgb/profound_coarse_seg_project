#!/usr/bin/env python
"""Run the 8-case overfit sanity check."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overfit_8cases.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    Trainer(cfg).fit()

if __name__ == "__main__":
    main()
