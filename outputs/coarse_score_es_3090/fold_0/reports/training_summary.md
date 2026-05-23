# Stage-1 Training Summary

- Log: `outputs/coarse_score_es_3090/fold_0/logs/train_log.csv`
- Total epochs in log: 18
- Best coarse_score epoch: 8
- Best Dice epoch: 17
- Best lesion_recall epoch: 1
- Best threshold-sweep coarse_score epoch: 8
- Lesion recall late drop: 0.1609
- Coarse score late drop: 0.0446
- Recommended Stage-2 checkpoint: `best_by_val_threshold_sweep_coarse_score.pth` from epoch 8

| checkpoint | epoch | selection metric | val_dice | positive_case_dice | lesion_recall | fp_per_case |
|---|---:|---:|---:|---:|---:|---:|
| best_by_val_coarse_score.pth | 8 | 0.7162 | 0.3044 | 0.3960 | 0.9080 | 2.2736 |
| best_by_val_threshold_sweep_coarse_score.pth | 8 | 0.7331 | 0.3044 | 0.3960 | 0.9080 | 2.2736 |
| best_by_val_lesion_recall.pth | 1 | 1.0000 | 0.0119 | 0.0453 | 1.0000 | 8.2196 |
| best_by_val_positive_case_dice.pth | 9 | 0.4002 | 0.4123 | 0.4002 | 0.8161 | 1.6993 |
| best_by_val_dice.pth | 17 | 0.5250 | 0.5250 | 0.3855 | 0.8161 | 1.1385 |

For Stage 2 prompt generation, prefer the threshold-sweep coarse checkpoint when its lesion recall is high and fp_per_case remains usable.
