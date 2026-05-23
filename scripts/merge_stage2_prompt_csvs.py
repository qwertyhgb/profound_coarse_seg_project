#!/usr/bin/env python
"""Merge several Stage-2 prompt CSV files into one training CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    fieldnames = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Prompt CSV not found: {path}")
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(list(reader))
    if not rows:
        raise ValueError("No rows found in input prompt CSVs")
    if "proposal_rank" in fieldnames:
        by_case_count = {}
        for row in rows:
            case_id = row["case_id"]
            by_case_count[case_id] = by_case_count.get(case_id, 0) + 1
            row["proposal_rank"] = by_case_count[case_id]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} merged prompts to {output}")


if __name__ == "__main__":
    main()
