#!/usr/bin/env python
"""Create patient-level stratified K-fold splits for PI-CAI NPZ files.

Each fold directory contains train.txt, val.txt, and test.txt. By default, val
and test are the same held-out fold, which is common for cross-validation. Use
--val-from-train-ratio to carve a small validation subset out of the training
folds while keeping test as the held-out fold.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from collections import defaultdict
import numpy as np


def read_meta(path: Path):
    """Return case_id, patient_id, positive flag for one processed NPZ."""
    with np.load(path, allow_pickle=False) as data:
        case_id = str(data["case_id"])
        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
        patient_id = str(meta.get("patient_id", case_id.split("_")[0]))
        positive = bool(data["label"].sum() > 0)
    return case_id, patient_id, positive


def split_groups_stratified(groups, group_pos, k: int, seed: int):
    """Distribute positive and negative patient groups across k folds."""
    rng = np.random.default_rng(seed)
    pos = [g for g in groups if group_pos[g]]
    neg = [g for g in groups if not group_pos[g]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(k)]
    for arr in (pos, neg):
        for i, group in enumerate(arr):
            folds[i % k].append(group)
    for fold in folds:
        rng.shuffle(fold)
    return folds


def groups_to_cases(group_ids, groups, rng):
    cases = [case for gid in group_ids for case in groups[gid]]
    rng.shuffle(cases)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="../picai_preprocessing_project/data/processed/picai_profound_prompt_v2")
    parser.add_argument("--output-dir", default="data/splits/5fold")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-from-train-ratio", type=float, default=0.0,
                        help="If >0, create val.txt from train groups and keep test.txt as held-out fold.")
    args = parser.parse_args()

    root = Path(args.processed_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed root not found: {root}")
    if args.num_folds < 2:
        raise ValueError("--num-folds must be >= 2")
    if not 0.0 <= args.val_from_train_ratio < 0.5:
        raise ValueError("--val-from-train-ratio must be in [0, 0.5)")

    groups = defaultdict(list)
    group_pos = defaultdict(bool)
    for path in sorted(root.rglob("*.npz")):
        case_id, patient_id, positive = read_meta(path)
        groups[patient_id].append(case_id)
        group_pos[patient_id] |= positive

    folds = split_groups_stratified(groups, group_pos, args.num_folds, args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 1009)

    all_group_ids = set(groups)
    for fold_idx, test_groups in enumerate(folds):
        test_groups = list(test_groups)
        train_groups = list(all_group_ids - set(test_groups))
        val_groups = list(test_groups)
        if args.val_from_train_ratio > 0:
            rng.shuffle(train_groups)
            n_val = max(1, int(round(len(train_groups) * args.val_from_train_ratio)))
            val_groups = train_groups[:n_val]
            train_groups = train_groups[n_val:]

        fold_dir = out / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_map = {
            "train": groups_to_cases(train_groups, groups, rng),
            "val": groups_to_cases(val_groups, groups, rng),
            "test": groups_to_cases(test_groups, groups, rng),
        }
        for name, cases in split_map.items():
            (fold_dir / f"{name}.txt").write_text("\n".join(cases) + "\n")
        n_pos_test = sum(bool(group_pos[g]) for g in test_groups)
        print(
            f"fold_{fold_idx}: train={len(split_map['train'])} "
            f"val={len(split_map['val'])} test={len(split_map['test'])} "
            f"test_positive_patients={n_pos_test}/{len(test_groups)}"
        )


if __name__ == "__main__":
    main()
