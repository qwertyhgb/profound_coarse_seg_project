# coarse_score_es_3090 阶段进展汇报报告

## 1. 阶段目标

本阶段围绕 ProFound 粗分割模型 coarse_score_es_3090 进行实验。目标不是单纯最大化体素 Dice，而是为后续 Stage-2 精细分割提供高召回、数量可控的候选病灶提示。因此主要关注 lesion-level recall、positive-case Dice、FP components/case 与综合 coarse score。

验证集为 fold_0，共 296 个病例，其中阳性病例 85 例，GT lesion 总数 87 个。

## 2. 粗分割训练设置

- 配置文件：configs/train_profound_coarse_recall_3090.yaml
- 输出目录：outputs/coarse_score_es_3090/fold_0
- 模型：ProFound-Conv encoder + coarse lesion decoder
- 策略：lesion-aware sampling，强调前景召回
- 损失：Dice + Focal-Tversky + BCE，Tversky 对 FN 更敏感
- checkpoint 选择：同时保存 coarse score、threshold sweep score、lesion recall、positive-case Dice、Dice 等 best checkpoint

## 3. 粗分割训练结果

粗分割训练共运行 18 个 epoch，early stopping 在第 18 轮触发。

| 选择标准 | epoch | Dice | lesion recall | positive-case Dice | FP/case | 推荐阈值 | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| best val_threshold_sweep_best_coarse_score | 8 | 0.3044 | 0.9080 | 0.3960 | 2.2736 | 0.25 | 当前主推粗分割 checkpoint |
| best val_coarse_score | 8 | 0.3044 | 0.9080 | 0.3960 | 2.2736 | 0.25 | 与 threshold sweep 最优一致 |
| best val_lesion_recall | 1 | 0.0119 | 1.0000 | 0.0453 | 8.2196 | 0.25 | 召回最高但假阳性过多 |
| best val_positive_case_dice | 9 | 0.4123 | 0.8161 | 0.4002 | 1.6993 | 0.10 | 分割质量更好但 lesion recall 下降 |
| best val_dice | 17 | 0.5250 | 0.8161 | 0.3855 | 1.1385 | 0.25 | 体素 Dice 最高但不符合高召回提示源目标 |

结论：第 8 轮 checkpoint 在 lesion recall 与 FP/case 之间最均衡，选择 best_by_val_threshold_sweep_coarse_score.pth 作为粗分割提示源。

## 4. 粗分割 proposal 质量评估

使用第 8 轮 best threshold-sweep checkpoint 后，对粗分割候选连通域进行评估：

- Prompt threshold：0.25
- Lesion recall：0.9425，即 82 / 87 个 GT lesions 被候选覆盖
- Candidate components/case：2.7264
- FP components/case：2.4493
- Component precision：0.1016
- Mean prompt voxels/case：4472.71

这说明粗分割已基本达到 Stage-2 的核心要求：大多数真实病灶被保留，但候选中假阳性仍然偏多。

## 5. 后处理策略筛选

为了减少 Stage-2 输入候选数量，对粗分割候选做 post-processing sweep。以 lesion recall >= 0.90 为约束，最优策略为：

- min_component_size = 50
- min_max_probability = 0.50
- top_k = 5
- rank_by = max_probability

| 指标 | 数值 |
|---|---:|
| Lesion recall | 0.9080，79 / 87 |
| Candidates/case | 1.9493 |
| FP/case | 1.6824 |
| Component precision | 0.1369 |

后处理相比原始 proposal 将候选数量从约 2.73/case 降到约 1.95/case，同时仍维持 90% 以上 lesion recall。

## 6. Stage-2 refinement 进展

在粗分割候选基础上，进一步训练了多个 Stage-2 refinement 版本。训练日志中的最佳验证表现如下：

| 版本 | epoch 数 | best val Dice | best positive-case Dice | best voxel recall | 主要改动 |
|---|---:|---:|---:|---:|---|
| v1 | 10 | 0.4403 | 0.3888 | 0.6498 | 初始 refinement baseline |
| v1 balanced | 40 | 0.6465 | 0.5042 | 0.7408 | 平衡采样后显著提升 |
| v2 objectness | 46 | 0.6627 | 0.5305 | 0.7651 | 加入 objectness head 与筛选 |
| v3 multithreshold + GT jitter | 37 | 0.6641 | 0.5097 | 0.7465 | 多阈值 prompts 与 GT jitter |

训练日志显示，v2/v3 相比 v1 已明显改善，其中 v2 的 positive-case Dice 最好，v3 的最终 case-level lesion recall 更高。

## 7. 当前最佳 Stage-2 case-level 结果

### v2 objectness 最终策略

- Checkpoint：stage2_refinement_v2_objectness/checkpoints/best_by_val_recall_safe_dice.pth
- mask threshold：0.60
- objectness threshold：0.20
- Dice：0.4322
- Precision：0.3099
- Voxel recall：0.7140
- Lesion recall：0.8161，71 / 87
- Pred components/case：1.4429
- FP components/case：1.1938
- Component precision：0.1727

### v3 multithreshold + GT jitter 最终策略

- Checkpoint：stage2_refinement_v3_multithr_gtjitter/checkpoints/best_by_val_recall_safe_dice.pth
- prompt CSV：stage2_data_v3/val/prompts/coarse_prompts_multithreshold.csv
- mask threshold：0.60
- objectness threshold：0.10
- Dice：0.4595
- Precision：0.3403
- Voxel recall：0.7071
- Lesion recall：0.8621，75 / 87
- Pred components/case：1.7297
- FP components/case：1.4696
- Component precision：0.1504

相比 v2，v3 在 Dice、precision 和 lesion recall 上均提升，尤其 lesion recall 从 81.61% 提高到 86.21%。代价是 FP/case 从 1.19 增加到 1.47。

## 8. Stage-2 阈值筛选结论

v3 阶段 threshold sweep 给出的推荐策略为：

- mask threshold：0.60
- 使用 objectness filter：是
- objectness threshold：0.10
- selection score：0.8594
- Dice：0.4595
- Precision：0.3403
- Recall：0.7071
- Lesion recall：0.8621
- FP/case：1.4696

这与 case-level evaluation 的最佳结果一致，因此当前建议以后续实验采用 v3 + mask 0.60 + objectness 0.10 作为 fold_0 的主策略。

## 9. 错误案例分析

v3 当前仍有 12 个阳性病例存在漏检或部分命中，总共漏掉 12 个 lesions。失败原因分布：

| 原因 | 病例数 | 含义 |
|---|---:|---|
| A_no_gt_overlapping_prompt | 5 | 粗分割/多阈值 prompt 没有覆盖 GT，Stage-2 无正确空间先验 |
| B_gt_prompt_filtered_by_objectness | 3 | 有 GT-overlap prompt，但 objectness 过滤掉了 |
| D_refined_mask_missed_gt_despite_kept_prompt | 3 | prompt 保留了，但 refinement decoder 未分出病灶 |
| E_multi_lesion_partial_hit | 1 | 多病灶病例中只命中部分病灶 |

关键观察：约 5 / 12 的失败来自 Stage-1 prompt 覆盖不足，说明进一步提升粗分割 proposal recall 仍然是主要突破口；3 / 12 来自 objectness 误杀，提示 objectness 阈值需要保守使用。

## 10. 阶段性结论

1. coarse_score_es_3090 粗分割阶段已经形成可用的高召回候选生成器。
2. 最优粗分割 checkpoint 在 threshold sweep 下达到 lesion recall 94.25%，能够覆盖 82 / 87 个 GT lesions。
3. 后处理后可以在 lesion recall 90.80% 条件下将候选数量控制到约 1.95 个/case。
4. Stage-2 refinement 从 v1 到 v3 持续提升，当前 v3 case-level Dice 0.4595，lesion recall 86.21%。
5. 当前主要瓶颈不是训练是否收敛，而是粗 prompt 覆盖不足、objectness 误过滤和 refinement 对部分小病灶/困难病例仍不稳。

## 11. 下一步计划

- 继续优化粗分割 proposal：保持 lesion recall >= 0.90，同时进一步降低 FP/case。
- 对 objectness threshold 做更细粒度 sweep，重点避免误杀 GT-overlap prompt。
- 对 v3 漏检病例进行可视化复核，区分粗分割漏检、prompt 过滤错误和 refinement 失败。
- 将当前 fold_0 策略扩展到 5-fold 验证，检查稳定性。
- 在新的 PCaSAM3D-ProFound 端到端模型中吸收本阶段经验：以 lesion-level recall 为 coarse 阶段主目标，并使用多阈值/多点 prompt 缓解多病灶漏检。

## 12. 可引用文件

- 粗分割训练日志：outputs/coarse_score_es_3090/fold_0/logs/train_log.csv
- 粗分割阈值 sweep：outputs/coarse_score_es_3090/fold_0/logs/threshold_sweep_log.csv
- proposal 评估：outputs/coarse_score_es_3090/fold_0/inference/proposal_reports/proposal_summary.md
- postprocess sweep：outputs/coarse_score_es_3090/fold_0/inference/postprocess_sweep/postprocess_summary.md
- Stage-2 v2 总结：outputs/coarse_score_es_3090/fold_0/stage2_refinement_v2_objectness/reports/stage2_summary.md
- Stage-2 v3 case-level：outputs/coarse_score_es_3090/fold_0/stage2_refinement_v3_multithr_gtjitter/case_eval_mask060_obj010_min1/summary.md
- v3 漏检分析：outputs/coarse_score_es_3090/fold_0/stage2_refinement_v3_multithr_gtjitter/reports/stage2_v3_miss_case_analysis.md
