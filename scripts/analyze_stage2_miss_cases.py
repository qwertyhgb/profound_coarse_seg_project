#!/usr/bin/env python
"""Analyze missed or partially hit positive cases for Stage-2 case-level evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-metrics", required=True)
    parser.add_argument("--prompt-csv", required=True)
    parser.add_argument("--proposal-objectness", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objectness-threshold", type=float, default=0.10)
    parser.add_argument("--min-prompts-per-case", type=int, default=1)
    parser.add_argument("--tag", default="stage2")
    args = parser.parse_args()

    case = pd.read_csv(args.case_metrics)
    prompt = pd.read_csv(args.prompt_csv)
    obj = pd.read_csv(args.proposal_objectness)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt["overlaps_gt_bool"] = _bool_series(prompt["overlaps_gt"])
    merged = prompt.merge(obj, on=["case_id", "proposal_rank"], how="left")
    merged["objectness"] = merged["objectness"].fillna(-1.0)
    merged["fallback_keep"] = False

    if args.min_prompts_per_case > 0:
        for _, group in merged.groupby("case_id"):
            keep_idx = group.sort_values("objectness", ascending=False).head(args.min_prompts_per_case).index
            merged.loc[keep_idx, "fallback_keep"] = True

    merged["kept"] = (merged["objectness"] >= args.objectness_threshold) | merged["fallback_keep"]

    positive = case[case["total_gt_lesions"] > 0].copy()
    miss = positive[positive["hit_lesions"] < positive["total_gt_lesions"]].copy()

    rows: list[dict] = []
    for _, row in miss.iterrows():
        case_id = row["case_id"]
        case_prompts = merged[merged["case_id"] == case_id].copy()
        gt_prompts = case_prompts[case_prompts["overlaps_gt_bool"]]
        kept_prompts = case_prompts[case_prompts["kept"]]
        kept_gt_prompts = gt_prompts[gt_prompts["kept"]]

        if len(gt_prompts) == 0:
            reason = "A_no_gt_overlapping_prompt"
        elif len(kept_gt_prompts) == 0:
            reason = "B_gt_prompt_filtered_by_objectness"
        elif float(row["hit_lesions"]) > 0 and float(row["hit_lesions"]) < float(row["total_gt_lesions"]):
            reason = "E_multi_lesion_partial_hit"
        elif float(row["pred_components"]) == 0:
            reason = "C_mask_threshold_removed_prediction"
        else:
            reason = "D_refined_mask_missed_gt_despite_kept_prompt"

        rows.append(
            {
                "case_id": case_id,
                "reason": reason,
                "gt_lesions": int(row["total_gt_lesions"]),
                "hit_lesions_final": int(row["hit_lesions"]),
                "missed_lesions": int(row["total_gt_lesions"] - row["hit_lesions"]),
                "gt_voxels": int(row["gt_voxels"]),
                "final_dice": float(row["dice"]),
                "final_precision": float(row["precision"]),
                "final_recall": float(row["recall"]),
                "final_pred_voxels": int(row["pred_voxels"]),
                "final_pred_components": int(row["pred_components"]),
                "final_fp_components": int(row["fp_pred_components"]),
                "prompts": int(len(case_prompts)),
                "gt_prompts": int(len(gt_prompts)),
                "kept_prompts": int(len(kept_prompts)),
                "kept_gt_prompts": int(len(kept_gt_prompts)),
                "max_objectness_any_prompt": float(case_prompts["objectness"].max()) if len(case_prompts) else -1.0,
                "max_objectness_gt_prompt": float(gt_prompts["objectness"].max()) if len(gt_prompts) else -1.0,
                "gt_prompt_ranks": ";".join(map(str, gt_prompts["proposal_rank"].astype(int).tolist())),
                "kept_gt_prompt_ranks": ";".join(map(str, kept_gt_prompts["proposal_rank"].astype(int).tolist())),
                "source_thresholds": ";".join(map(str, sorted(case_prompts["source_threshold"].astype(str).unique())))
                if len(case_prompts)
                else "",
            }
        )

    out = pd.DataFrame(rows)
    csv_path = output_dir / f"{args.tag}_miss_case_analysis.csv"
    out.to_csv(csv_path, index=False)

    counts = out["reason"].value_counts().to_dict() if len(out) else {}
    md: list[str] = [
        f"# {args.tag} Miss Positive Case Analysis",
        "",
        f"- Miss / partial-hit positive cases: {len(out)}",
        f"- Missed lesions: {int(out['missed_lesions'].sum()) if len(out) else 0}",
    ]
    for key, value in counts.items():
        md.append(f"- {key}: {value}")

    md.extend(
        [
            "",
            "## Interpretation",
            "- A_no_gt_overlapping_prompt: no retained coarse prompt overlaps GT, so Stage 2 has no correct spatial prior.",
            "- B_gt_prompt_filtered_by_objectness: at least one GT-overlapping prompt exists, but objectness filtering/fallback did not keep it.",
            "- C_mask_threshold_removed_prediction: a GT prompt survived, but the final mask threshold left no component.",
            "- D_refined_mask_missed_gt_despite_kept_prompt: a GT prompt survived and prediction exists, but the refined mask missed GT.",
            "- E_multi_lesion_partial_hit: at least one lesion was hit but another lesion in the same case was missed.",
            "",
            "## Cases",
            "| case_id | reason | GT lesions | final hit | missed | GT voxels | GT prompts | kept GT prompts | max GT obj | pred comp | FP comp | Dice |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in out.iterrows():
        md.append(
            f"| {row.case_id} | {row.reason} | {row.gt_lesions} | {row.hit_lesions_final} | "
            f"{row.missed_lesions} | {row.gt_voxels} | {row.gt_prompts} | {row.kept_gt_prompts} | "
            f"{row.max_objectness_gt_prompt:.3f} | {row.final_pred_components} | "
            f"{row.final_fp_components} | {row.final_dice:.4f} |"
        )

    md.extend(["", f"Detailed CSV: `{csv_path}`"])
    md_path = output_dir / f"{args.tag}_miss_case_analysis.md"
    md_path.write_text("\n".join(md))

    print(md_path)
    print(csv_path)
    if len(out):
        print(
            out[
                [
                    "case_id",
                    "reason",
                    "gt_lesions",
                    "hit_lesions_final",
                    "gt_prompts",
                    "kept_gt_prompts",
                    "max_objectness_gt_prompt",
                    "final_pred_components",
                    "final_fp_components",
                    "final_dice",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
