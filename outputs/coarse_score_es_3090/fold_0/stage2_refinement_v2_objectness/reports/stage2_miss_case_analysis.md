# Stage 2 Miss Positive Case Analysis

- Miss / partial-hit positive cases: 16
- A_stage1_or_postprocess_no_gt_prompt: 7
- B_objectness_filtered_gt_prompt: 6
- E_multi_lesion_partial_hit: 2
- D_refined_mask_missed_gt_despite_kept_prompt: 1

## Interpretation

- `A_stage1_or_postprocess_no_gt_prompt`: Stage 2 cannot recover because no retained coarse prompt overlaps GT.
- `B_objectness_filtered_gt_prompt`: Stage 1 had a GT-overlapping prompt, but the selected objectness threshold removed it.
- `C_mask_threshold_removed_prediction`: a GT prompt survived, but final mask threshold produced no component.
- `D_refined_mask_missed_gt_despite_kept_prompt`: prompt survived, prediction exists, but it does not overlap GT.
- `E_multi_lesion_partial_hit`: at least one lesion was hit, but another GT lesion was missed.

## Cases

| case_id | reason | GT lesions | final hit | GT voxels | stage1 all hit | post GT prompts | kept GT prompts | max GT obj | final pred comp | final FP | final Dice |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10106_1000106 | E_multi_lesion_partial_hit | 2 | 1 | 393 | 2 | 2 | 2 | 0.930 | 3 | 2 | 0.0029 |
| 10334_1000340 | A_stage1_or_postprocess_no_gt_prompt | 1 | 0 | 87 | 1 | 0 | 0 | -1.000 | 0 | 0 | 0.0000 |
| 10433_1000441 | A_stage1_or_postprocess_no_gt_prompt | 1 | 0 | 264 | 1 | 0 | 0 | -1.000 | 1 | 1 | 0.0000 |
| 10605_1000619 | B_objectness_filtered_gt_prompt | 1 | 0 | 555 | 1 | 1 | 0 | 0.179 | 0 | 0 | 0.0000 |
| 10651_1000667 | B_objectness_filtered_gt_prompt | 1 | 0 | 485 | 1 | 1 | 0 | 0.105 | 0 | 0 | 0.0000 |
| 10700_1000716 | B_objectness_filtered_gt_prompt | 1 | 0 | 276 | 1 | 1 | 0 | 0.099 | 0 | 0 | 0.0000 |
| 10728_1000744 | B_objectness_filtered_gt_prompt | 1 | 0 | 1935 | 1 | 1 | 0 | 0.126 | 0 | 0 | 0.0000 |
| 10915_1000932 | A_stage1_or_postprocess_no_gt_prompt | 1 | 0 | 66 | 0 | 0 | 0 | -1.000 | 1 | 1 | 0.0000 |
| 11009_1001029 | A_stage1_or_postprocess_no_gt_prompt | 1 | 0 | 400 | 0 | 0 | 0 | -1.000 | 1 | 1 | 0.0000 |
| 11143_1001166 | A_stage1_or_postprocess_no_gt_prompt | 1 | 0 | 96 | 0 | 0 | 0 | -1.000 | 0 | 0 | 0.0000 |
| 11231_1001254 | A_stage1_or_postprocess_no_gt_prompt | 1 | 0 | 84 | 0 | 0 | 0 | -1.000 | 0 | 0 | 0.0000 |
| 11258_1001281 | D_refined_mask_missed_gt_despite_kept_prompt | 1 | 0 | 444 | 1 | 1 | 1 | 0.910 | 2 | 2 | 0.0000 |
| 11280_1001303 | B_objectness_filtered_gt_prompt | 1 | 0 | 1120 | 1 | 1 | 0 | 0.114 | 0 | 0 | 0.0000 |
| 11284_1001307 | B_objectness_filtered_gt_prompt | 1 | 0 | 2708 | 1 | 1 | 0 | 0.164 | 0 | 0 | 0.0000 |
| 11357_1001380 | A_stage1_or_postprocess_no_gt_prompt | 1 | 0 | 415 | 0 | 0 | 0 | -1.000 | 0 | 0 | 0.0000 |
| 11425_1001449 | E_multi_lesion_partial_hit | 2 | 1 | 20625 | 2 | 1 | 1 | 0.947 | 1 | 0 | 0.8166 |

Detailed CSV: `outputs/coarse_score_es_3090/fold_0/stage2_refinement_v2_objectness/reports/stage2_miss_case_analysis.csv`
