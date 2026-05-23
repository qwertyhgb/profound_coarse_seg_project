#!/usr/bin/env python
"""Summarize a Stage-1 training run and recommend a Stage-2 checkpoint."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def _to_float(row: dict, *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return default


def _best(rows: list[dict], *keys: str) -> dict | None:
    valid = [(row, _to_float(row, *keys)) for row in rows]
    valid = [(row, value) for row, value in valid if value == value]
    if not valid:
        return None
    return max(valid, key=lambda item: item[1])[0]


def _epoch(row: dict | None) -> str:
    if row is None:
        return "NA"
    return str(row.get("epoch", "NA"))


def _fmt(value: float) -> str:
    return "NA" if value != value else f"{value:.4f}"


def _row_summary(label: str, row: dict | None, metric_keys: tuple[str, ...]) -> list[str]:
    if row is None:
        return [label, "NA", "NA", "NA", "NA", "NA", "NA"]
    return [
        label,
        _epoch(row),
        _fmt(_to_float(row, *metric_keys)),
        _fmt(_to_float(row, "val_dice")),
        _fmt(_to_float(row, "val_positive_case_dice")),
        _fmt(_to_float(row, "val_lesion_recall")),
        _fmt(_to_float(row, "val_fp_per_case", "val_fp_components_per_case")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None, help="Run directory, e.g. outputs/coarse_score_es_3090/fold_0")
    parser.add_argument("--log", default=None, help="Path to train_log.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Report path. Default: <run-dir>/reports/training_summary.md, or <log parent>/../reports/training_summary.md",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.log:
        log_path = Path(args.log)
        if run_dir is None and log_path.parent.name == "logs":
            run_dir = log_path.parent.parent
    elif run_dir:
        log_path = run_dir / "logs" / "train_log.csv"
    else:
        raise ValueError("Pass --run-dir or --log.")
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    with log_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Training log is empty: {log_path}")

    best_coarse = _best(rows, "val_coarse_score")
    best_dice = _best(rows, "val_dice")
    best_lesion_recall = _best(rows, "val_lesion_recall")
    best_pos_dice = _best(rows, "val_positive_case_dice")
    best_sweep = _best(rows, "val_threshold_sweep_best_coarse_score", "val_best_threshold_coarse_score")
    last = rows[-1]
    recall_drop = _to_float(best_lesion_recall or {}, "val_lesion_recall") - _to_float(last, "val_lesion_recall")
    coarse_drop = _to_float(best_coarse or {}, "val_coarse_score") - _to_float(last, "val_coarse_score")

    recommended_label = "best_by_val_threshold_sweep_coarse_score.pth" if best_sweep is not None else "best_by_val_coarse_score.pth"
    recommended_epoch = _epoch(best_sweep or best_coarse)

    table_rows = [
        _row_summary("best_by_val_coarse_score.pth", best_coarse, ("val_coarse_score",)),
        _row_summary(
            "best_by_val_threshold_sweep_coarse_score.pth",
            best_sweep,
            ("val_threshold_sweep_best_coarse_score", "val_best_threshold_coarse_score"),
        ),
        _row_summary("best_by_val_lesion_recall.pth", best_lesion_recall, ("val_lesion_recall",)),
        _row_summary("best_by_val_positive_case_dice.pth", best_pos_dice, ("val_positive_case_dice",)),
        _row_summary("best_by_val_dice.pth", best_dice, ("val_dice",)),
    ]

    lines = [
        "# Stage-1 Training Summary",
        "",
        f"- Log: `{log_path}`",
        f"- Total epochs in log: {len(rows)}",
        f"- Best coarse_score epoch: {_epoch(best_coarse)}",
        f"- Best Dice epoch: {_epoch(best_dice)}",
        f"- Best lesion_recall epoch: {_epoch(best_lesion_recall)}",
        f"- Best threshold-sweep coarse_score epoch: {_epoch(best_sweep)}",
        f"- Lesion recall late drop: {_fmt(recall_drop)}",
        f"- Coarse score late drop: {_fmt(coarse_drop)}",
        f"- Recommended Stage-2 checkpoint: `{recommended_label}` from epoch {recommended_epoch}",
        "",
        "| checkpoint | epoch | selection metric | val_dice | positive_case_dice | lesion_recall | fp_per_case |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "For Stage 2 prompt generation, prefer the threshold-sweep coarse checkpoint when its lesion recall is high and fp_per_case remains usable."
    )

    if args.output:
        output_path = Path(args.output)
    elif run_dir is not None:
        output_path = run_dir / "reports" / "training_summary.md"
    else:
        output_path = log_path.parent / "training_summary.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
