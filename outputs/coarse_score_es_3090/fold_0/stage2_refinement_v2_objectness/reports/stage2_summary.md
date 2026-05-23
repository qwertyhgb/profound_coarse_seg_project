# Stage 2 V2 Objectness Summary

- Run: `outputs/coarse_score_es_3090/fold_0/stage2_refinement_v2_objectness`
- Epochs: 46
- Early stopping: epoch 46, counter 12, monitor `val_recall_safe_dice`
- Best val_dice: epoch 34, value 0.6627, dice 0.6627, precision 0.6080, recall 0.7283, posDice 0.5228
- Best val_positive_case_dice: epoch 32, value 0.5305, dice 0.6599, precision 0.5954, recall 0.7401, posDice 0.5305
- Best val_recall: epoch 27, value 0.7651, dice 0.6463, precision 0.5594, recall 0.7651, posDice 0.5257
- Best val_recall_safe_dice: epoch 34, value 0.6627, dice 0.6627, precision 0.6080, recall 0.7283, posDice 0.5228

## Case-Level Evaluation

### No objectness filter
- dice: 0.3635
- precision: 0.2423
- recall: 0.7275
- lesion_recall: 0.8851
- pred_components_per_case: 2.7197
- fp_components_per_case: 2.4394
- component_precision: 0.1031

### Objectness filter 0.50
- dice: 0.4745
- precision: 0.3572
- recall: 0.7067
- lesion_recall: 0.6897
- pred_components_per_case: 0.8097
- fp_components_per_case: 0.5917
- component_precision: 0.2692

## Recommendation

- Use `checkpoints/best_by_val_recall_safe_dice.pth` as the main Stage-2 checkpoint for now.
- For Stage-2 as a high-recall refinement/candidate stage, do not use objectness filter 0.50 by default because lesion recall drops strongly.
- For a more final segmentation-style output, objectness filter 0.50 improves Dice/precision and reduces FP components, but it is too aggressive for coarse-to-fine prompt generation.
- Next experiment: sweep objectness thresholds around 0.20-0.50 and mask thresholds around 0.35-0.60, selecting by lesion recall plus FP/case rather than Dice alone.
## Final Selected Strategy

- Checkpoint: `checkpoints/best_by_val_recall_safe_dice.pth`
- Output: `final_case_eval_thr060_obj020`
- mask_threshold: 0.60
- use_objectness_filter: True
- objectness_threshold: 0.20
- Dice: 0.4322
- Precision: 0.3099
- Voxel recall: 0.7140
- Lesion recall: 0.8161 (71/87)
- Pred components/case: 1.4429
- FP components/case: 1.1938
- Component precision: 0.1727

This is the current fold-0 Stage-2 strategy to carry forward before expanding to 5-fold experiments.

