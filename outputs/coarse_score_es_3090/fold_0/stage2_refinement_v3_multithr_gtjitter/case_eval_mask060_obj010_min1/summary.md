# Stage-2 Case-Level Evaluation

- checkpoint: outputs/coarse_score_es_3090/fold_0/stage2_refinement_v3_multithr_gtjitter/checkpoints/best_by_val_recall_safe_dice.pth
- prompt_csv: outputs/coarse_score_es_3090/fold_0/stage2_data_v3/val/prompts/coarse_prompts_multithreshold.csv
- mask_threshold: 0.6
- use_objectness_filter: True
- weight_by_objectness: False
- objectness_threshold: 0.1
- min_prompts_per_case: 1
- cases: 296
- dice: 0.4595168698060003
- precision: 0.3403424250788314
- recall: 0.7071235428642926
- lesion_recall: 0.8620689655172413
- hit_lesions: 75.0
- total_gt_lesions: 87.0
- pred_components_per_case: 1.7297297297297298
- fp_components_per_case: 1.4695945945945945
- component_precision: 0.150390625
