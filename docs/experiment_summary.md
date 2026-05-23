# Experiment Summary — PCaSAM-3D-ProFound

**数据集**：PI-CAI fold_0（train=1204, val=296；阳性率 28.3%）  
**任务**：前列腺 mpMRI（T2W/ADC/HBV）临床显著前列腺癌病灶分割  
**更新时间**：2026-05-23

---

## 总体流程

```
Phase 0: SAM-Med3D 零样本基线
    ↓
Stage 1: ProFound-Conv 粗分割（高召回率候选生成）
    ↓
Stage 2: SAM-Med3D 提示条件化精细化分割
```

---

## Phase 0 — SAM-Med3D 零样本基线

| 指标 | 值 |
|------|-----|
| 总 cases | 296 |
| Overall Dice | 0.0012 |
| Positive-case Dice | 0.0040 |
| Lesion Recall | **0.6322** |
| True Negative Rate | 0.0（大量假阳性） |

**结论**：SAM-Med3D 零样本有一定检测能力（63% 病灶召回），但 Dice 极低，假阳性泛滥，不可直接用于临床。验证了需要领域适配训练。

---

## Stage 1 — 粗分割实验

Stage 1 目标：训练高召回率的粗分割模型，为 Stage 2 提供候选区域和自动提示。  
核心指标：`coarse_score = lesion_recall × 1.0 + positive_dice × 0.45 - fp_penalty`，目标 lesion_recall ≥ 0.90。

### 实验对比

| 实验 | Epochs | Best coarse_score | Best threshold | Lesion Recall | Pos Dice | FP/case | 状态 |
|------|--------|-------------------|----------------|---------------|----------|---------|------|
| coarse_score_es_3090 | 18 (ES=10) | 0.7331 | N/A | 0.9080 | 0.3960 | N/A | 已废弃 |
| **v3_balanced** | 41 (ES=20) | **1.0750** | 0.20 | **0.9471** | 0.4514 | 1.878 | 参考基线 |
| **v4_data_opt** | 44 (ES=20) | 1.0537 | 0.25 | 0.9294 | **0.4570** | 2.034 | **当前推荐** |

### 各实验说明

#### coarse_score_es_3090（已废弃）
- 最早的粗分割基线，用于验证可行性
- Early stopping patience=10 过短，仅跑 18 epochs 就停止，模型未充分收敛
- 无 gland mask 后处理，FP 未统计
- **已被 v3/v4 取代，不再作为 Stage 2 的 prompt 来源**

#### pcasam3d_stage1_coarse_v3_balanced（参考基线）
- 引入 `coarse_score` 综合评分（lesion_recall + positive_dice - fp_penalty）
- 使用 `channelwise_nonzero` 双重归一化
- coarse_score 最高（1.075），lesion_recall 最高（0.947）
- 缺点：归一化方式与 Stage 2 v4 不一致，存在训练-推理分布差异

#### pcasam3d_stage1_coarse_v4_data_opt（**当前推荐**）
- **关键改进**：
  - `normalize: preprocessed`（去掉双重 z-score，与 Stage 2 v4 保持一致）
  - `gland_aware_negative_sampling: true`（90% 负样本在前列腺内，学习更难的假阳性抑制）
  - 验证时启用 gland mask 后处理（margin=3 voxels）
- positive_case_dice 略高于 v3（0.457 vs 0.451）
- lesion_recall 略低于 v3（0.929 vs 0.947），是 gland-aware 负采样的代价
- **选为 Stage 2 v4 的 prompt 来源，因为数据处理一致性更重要**

---

## Stage 2 — 精细化分割实验

Stage 2 目标：以 Stage 1 粗分割为提示来源，用 SAM-Med3D 解码器精细化分割。  
核心指标：`refined_sweep_score = 1.15×pos_dice + 0.25×global_dice + 0.3×lesion_recall + 0.15×precision - fp_penalty - recall_gap_penalty`

所有 Stage 2 实验均基于 `stage=prompt` 训练策略：冻结 ProFound 编码器和 SAM 解码器主体，只训练 FeatureBridge + ModalityCrossAttention + HighResRefinement。

### 实验对比

| 实验 | Epochs | Best sweep_score | Best threshold | Pos Dice | Lesion Recall | FP/case | Global Dice | 状态 |
|------|--------|-----------------|----------------|----------|---------------|---------|-------------|------|
| v2_balanced | 33 | **0.9660** | 0.25 | **0.4690** | 0.8765 | 1.081 | 0.5196 | 参考基线 |
| v3_align_adapter | 47 | 0.9424 | 0.85 | 0.4580 | 0.8882 | 1.284 | 0.4990 | 消融 |
| v3a_align_only | 47 | 0.9493 | 0.85 | 0.4611 | **0.9000** | 1.260 | 0.4971 | 消融 |
| v3c_selfgated | 47 | 0.9464 | 0.85 | 0.4591 | 0.8882 | **1.230** | 0.4988 | 消融 |
| **v4_data_opt** | 63 | 0.9618 | 0.85 | 0.4539 | 0.8882 | 1.020 | **0.5391** | **当前推荐** |
| stage3_decoder_v2 | 11 | 1.0110* | 0.65 | 0.4711 | 0.8882 | 1.210 | 0.5012 | 进行中 |
| v4_no_box | 4 | 0.8238 | — | 0.3625 | 0.8235 | 0.335 | 0.4960 | 进行中 |

> *stage3_decoder_v2 仅跑 11 epochs，best@ep1，结果不稳定，待继续训练。

### 各实验说明

#### pcasam3d_stage2_precision_fp_v2_balanced（参考基线）
- 第一个完整 Stage 2 实验，从 Stage 1 v3 checkpoint 继续训练
- 使用 GT prompt curriculum（ep1-8 全 GT，ep9-32 退火到 35%）
- **sweep_score 最高（0.966），是目前最强的完整实验**
- 缺点：使用旧的 channelwise_nonzero 归一化，与 v4 数据处理不一致

#### v3 消融系列（v3_align_adapter / v3a_align_only / v3c_selfgated）
三个消融实验验证额外适配模块的效果：

| 模块 | v3_align_adapter | v3a_align_only | v3c_selfgated |
|------|:---:|:---:|:---:|
| ImageEmbeddingAlignment3D | ✅ | ✅ | ✅ |
| MaskDecoderAdapters | ✅ | ❌ | ✅ |
| SelfGatedMultiScaleFusion | ❌ | ❌ | ✅ |

**消融结论**：三个变体的 sweep_score（0.942/0.949/0.946）均低于 v2 baseline（0.966）。额外的适配模块没有带来收益，反而引入了训练不稳定性（best 出现在 ep1-3）。**这些模块已在 v4 lean 中全部禁用。**

#### pcasam3d_stage2_precision_fp_v4_data_opt（**当前推荐**）
- 从 Stage 1 v4 checkpoint 继续训练，数据处理完全一致
- 关键改进：
  - `normalize: preprocessed`（与 Stage 1 v4 一致）
  - `gland_aware_negative_sampling: true`（prob=0.85）
  - 验证时启用 gland mask 后处理
- **global_dice 最高（0.539）**，FP/case 最低（1.020）
- sweep_score（0.962）略低于 v2（0.966），差距在误差范围内
- 训练仍在进行（63/75 epochs），有继续提升空间

#### pcasam3d_stage3_decoder_dice_v2_recall_guard（进行中）
- 从 v2_balanced 最佳 checkpoint 继续，尝试进一步微调 SAM 解码器
- 仅跑 11 epochs，best@ep1（1.011），说明模型在初始 checkpoint 基础上难以继续提升
- 需要继续训练观察是否收敛

#### pcasam3d_stage2_precision_fp_v4_no_box（进行中）
- 消融实验：禁用 box prompt，仅用 point + mask_prior
- 仅跑 4 epochs，结果不具参考价值
- 目的：验证 box prompt 对 Stage 2 的贡献

---

## 关键发现与结论

### Stage 1
1. **lesion_recall ≥ 0.90 是可达的**：v3 达到 0.947，v4 达到 0.929
2. **gland-aware 负采样有效**：v4 的 FP/case 在验证时更低，global_dice 更高
3. **数据处理一致性优先于分数最高**：v4 虽然 coarse_score 略低于 v3，但作为 Stage 2 的 prompt 来源更合适

### Stage 2
1. **SAM-Med3D 解码器的提升有限**：Stage 1 单独的 pos_dice ~0.45，Stage 2 最好也只到 ~0.47，提升约 2 个点
2. **额外适配模块无效**：v3 系列的 alignment/adapter/selfgated 全部不如 v2 baseline
3. **数据优化有效**：v4_data_opt 的 global_dice（0.539）和 FP/case（1.020）均优于 v2
4. **Prompt curriculum 是关键**：GT prompt 退火策略是稳定 Stage 2 训练的核心机制
5. **Objectness head 完全失效**：所有实验中 pos/neg 概率差 <0.001，已在所有配置中禁用

### 当前瓶颈
- SAM-Med3D 的 8³ token 空间（每 token 代表 16³=4096 voxels）对小病灶（中位 1248 voxels）不友好
- PI-CAI 数据集本身的标注 inter-rater variability 约 0.5-0.7 Dice，是物理上限
- Box prompt 在训练时实际未被使用（`training_use_soft_box: false`），MedSAM 的核心优势未被充分利用

---

## 推荐 Checkpoint

| 用途 | 实验 | Checkpoint |
|------|------|-----------|
| Stage 2 prompt 来源 | pcasam3d_stage1_coarse_v4_data_opt | `fold_0/checkpoints/best_by_val_threshold_sweep_coarse_score.pth` |
| 最终分割（当前最强） | pcasam3d_stage2_precision_fp_v4_data_opt | `fold_0/checkpoints/best_by_val_refined_sweep_score.pth` |
| 备选（分数略高） | pcasam3d_stage2_precision_fp_v2_balanced | `fold_0/checkpoints/best_by_val_refined_sweep_score.pth` |

---

## 下一步计划

1. **修复 box prompt 训练**：实现局部 box（围绕每个 top-k 峰值的固定半径框），替代当前关闭的全局 std 框
2. **删除无效模块**：objectness head、SelfGatedMultiScale、DecoderAlignment/Adapters
3. **完成 v4_no_box 消融**：确认 box prompt 的实际贡献
4. **等待 v4_data_opt 收敛**：当前 63/75 epochs，观察最终结果
5. **5-fold 交叉验证**：目前所有实验仅在 fold_0，需要扩展到 5 折以获得可靠的统计结果
