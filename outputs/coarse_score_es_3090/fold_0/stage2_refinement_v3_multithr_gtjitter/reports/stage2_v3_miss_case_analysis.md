# stage2_v3 Miss Positive Case Analysis

- Miss / partial-hit positive cases: 12
- Missed lesions: 12
- A_no_gt_overlapping_prompt: 5
- D_refined_mask_missed_gt_despite_kept_prompt: 3
- B_gt_prompt_filtered_by_objectness: 3
- E_multi_lesion_partial_hit: 1

## Interpretation
- A_no_gt_overlapping_prompt: no retained coarse prompt overlaps GT, so Stage 2 has no correct spatial prior.
- B_gt_prompt_filtered_by_objectness: at least one GT-overlapping prompt exists, but objectness filtering/fallback did not keep it.
- C_mask_threshold_removed_prediction: a GT prompt survived, but the final mask threshold left no component.
- D_refined_mask_missed_gt_despite_kept_prompt: a GT prompt survived and prediction exists, but the refined mask missed GT.
- E_multi_lesion_partial_hit: at least one lesion was hit but another lesion in the same case was missed.

## Cases
| case_id | reason | GT lesions | final hit | missed | GT voxels | GT prompts | kept GT prompts | max GT obj | pred comp | FP comp | Dice |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10182_1000185 | D_refined_mask_missed_gt_despite_kept_prompt | 1 | 0 | 1 | 747 | 2 | 1 | 0.643 | 4 | 4 | 0.0000 |
| 10334_1000340 | B_gt_prompt_filtered_by_objectness | 1 | 0 | 1 | 87 | 2 | 0 | 0.012 | 0 | 0 | 0.0000 |
| 10433_1000441 | D_refined_mask_missed_gt_despite_kept_prompt | 1 | 0 | 1 | 264 | 2 | 2 | 0.904 | 9 | 9 | 0.0000 |
| 10605_1000619 | B_gt_prompt_filtered_by_objectness | 1 | 0 | 1 | 555 | 1 | 0 | 0.030 | 2 | 2 | 0.0000 |
| 10872_1000888 | B_gt_prompt_filtered_by_objectness | 1 | 0 | 1 | 3381 | 2 | 0 | 0.063 | 0 | 0 | 0.0000 |
| 10915_1000932 | A_no_gt_overlapping_prompt | 1 | 0 | 1 | 66 | 0 | 0 | -1.000 | 1 | 1 | 0.0000 |
| 11009_1001029 | A_no_gt_overlapping_prompt | 1 | 0 | 1 | 400 | 0 | 0 | -1.000 | 1 | 1 | 0.0000 |
| 11143_1001166 | A_no_gt_overlapping_prompt | 1 | 0 | 1 | 96 | 0 | 0 | -1.000 | 1 | 1 | 0.0000 |
| 11231_1001254 | A_no_gt_overlapping_prompt | 1 | 0 | 1 | 84 | 0 | 0 | -1.000 | 0 | 0 | 0.0000 |
| 11258_1001281 | D_refined_mask_missed_gt_despite_kept_prompt | 1 | 0 | 1 | 444 | 1 | 1 | 0.996 | 2 | 2 | 0.0000 |
| 11357_1001380 | A_no_gt_overlapping_prompt | 1 | 0 | 1 | 415 | 0 | 0 | -1.000 | 0 | 0 | 0.0000 |
| 11425_1001449 | E_multi_lesion_partial_hit | 2 | 1 | 1 | 20625 | 3 | 2 | 0.947 | 1 | 0 | 0.8083 |

Detailed CSV: `outputs/coarse_score_es_3090/fold_0/stage2_refinement_v3_multithr_gtjitter/reports/stage2_v3_miss_case_analysis.csv`