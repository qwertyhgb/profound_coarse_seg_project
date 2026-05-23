# Stage-2 Case-Level Evaluation

- checkpoint: outputs/coarse_score_es_3090/fold_0/stage2_refinement_v2_objectness/checkpoints/best_by_val_recall_safe_dice.pth
- prompt_csv: outputs/coarse_score_es_3090/fold_0/stage2_data/val/postprocess_sweep/coarse_prompts.csv
- mask_threshold: 0.6
- use_objectness_filter: True
- weight_by_objectness: False
- objectness_threshold: 0.2
- cases: 289
- dice: 0.43215940861289515
- precision: 0.30985290923570724
- recall: 0.7139878076754645
- lesion_recall: 0.8160919540229885
- hit_lesions: 71.0
- total_gt_lesions: 87.0
- pred_components_per_case: 1.4429065743944636
- fp_components_per_case: 1.193771626297578
- component_precision: 0.17266187050359713
