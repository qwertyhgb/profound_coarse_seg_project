#!/usr/bin/env python
"""Create patient-level train/val/test splits from processed NPZ files."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np


def read_meta(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        case_id = str(data["case_id"])
        meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
        patient_id = str(meta.get("patient_id", case_id.split("_")[0]))
        positive = bool(data["label"].sum() > 0)
    return case_id, patient_id, positive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="../picai_preprocessing_project/data/processed/picai_profound_prompt_v2")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7,0.1,0.2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.processed_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed root not found: {root}")
    groups = defaultdict(list); group_pos = defaultdict(bool)
    for p in sorted(root.rglob("*.npz")):
        case_id, patient_id, positive = read_meta(p)
        groups[patient_id].append(case_id); group_pos[patient_id] |= positive
    rng = np.random.default_rng(args.seed)
    pos = [g for g in groups if group_pos[g]]; neg = [g for g in groups if not group_pos[g]]
    rng.shuffle(pos); rng.shuffle(neg)
    buckets = [[], [], []]
    for arr in (pos, neg):
        n = len(arr); n_train = int(round(n*args.ratios[0])); n_val = int(round(n*args.ratios[1]))
        for i, part in enumerate([arr[:n_train], arr[n_train:n_train+n_val], arr[n_train+n_val:]]):
            buckets[i].extend(part)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, bucket in zip(["train", "val", "test"], buckets):
        cases = [c for g in bucket for c in groups[g]]
        rng.shuffle(cases)
        (out / f"{name}.txt").write_text("\n".join(cases) + "\n")
        print(name, len(cases), "cases")

if __name__ == "__main__":
    main()
