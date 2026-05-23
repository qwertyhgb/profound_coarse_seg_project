# PCaSAM-3D-ProFound

前列腺 mpMRI 临床显著前列腺癌（csPCa）病灶分割项目。将 ProFound-Conv 领域编码器、PCaSAM 风格自动提示生成与 SAM-Med3D 提示条件化解码器统一为一个端到端可训练模型。

**数据集**：PI-CAI（1500 cases，T2W/ADC/HBV 三模态 bpMRI）  
**当前状态**：fold_0 实验完成，Stage 1 + Stage 2 均有可用 checkpoint

---

## 目录

- [项目概述](#项目概述)
- [架构](#架构)
- [环境依赖](#环境依赖)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
  - [Stage 1：粗分割预训练](#stage-1粗分割预训练)
  - [Stage 2：提示条件化精细化](#stage-2提示条件化精细化)
- [评估](#评估)
- [调试](#调试)
- [实验结果](#实验结果)
- [输出目录规范](#输出目录规范)
- [参考文献](#参考文献)

---

## 项目概述

### 核心思路

PI-CAI 前列腺癌病灶极小（中位 ~1248 voxels）、低对比度、稀疏分布，直接端到端分割困难。本项目采用两阶段策略：

```
Stage 1: 高召回率粗分割  →  Stage 2: 提示条件化精细化
  "不漏检"                    "减假阳性，精边界"
```

**Stage 1** 训练一个高召回率的粗分割模型（目标 lesion_recall ≥ 0.90），输出全分辨率概率图。漏检的病灶在下游永远无法恢复，因此召回优先于精度。

**Stage 2** 从粗概率图自动生成 3D 点/框/mask 提示，送入 SAM-Med3D 解码器做精细化分割，抑制假阳性并改善边界质量。

### 与相关工作的关系

| 工作 | 本项目的借鉴 |
|------|------------|
| PCaSAM (Nature Digital Medicine 2025) | 粗分割 → 自动提示生成的两阶段框架 |
| MedSAM (Nature Communications 2024) | 边界框提示提供强空间上下文 |
| SAM-Med3D (arXiv 2310.15161) | 体积 3D 提示条件化分割 |
| ProFound | 前列腺 mpMRI 基础模型，ConvNeXtV2 骨干 |

---

## 架构

```
Input [B, 3, D, H, W]  (T2W / ADC / HBV)
    │
    ▼
ModalityAwareFusion3D          ← 可学习模态门控，支持训练时随机 dropout
    │
    ▼
ProFound-Conv Encoder          ← ConvNeXtV2-Tiny，前列腺 mpMRI 预训练，冻结
  stage1: [B,  96, D/4,  H/4,  W/4]
  stage2: [B, 192, D/8,  H/8,  W/8]
  stage3: [B, 384, D/16, H/16, W/16]
  stage4: [B, 768, D/32, H/32, W/32]
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
CoarseBranch (FPN)                    ProFoundToSAMBridge
  → coarse_logits [B,1,D,H,W]          → image_embedding [B,384,8,8,8]
    │                                          │
    ▼                                          │
AutoPrompt3DFromCoarse                         │
  → point_coords, box_coords                  │
  → mask_prior [B,1,8,8,8]                    │
    │                                          │
    ▼                                          ▼
SAM-Med3D PromptEncoder3D + MaskDecoder3D
  → refined_logits [B,1,D,H,W]
```

**训练阶段划分**：

| 阶段 | 训练模块 | 冻结模块 | 配置关键字 |
|------|---------|---------|-----------|
| `stage=coarse` | ModalityFusion, CoarseBranch | ProFound, SAM | Stage 1 |
| `stage=prompt` | FeatureBridge, CrossAttention, HighResRefinement | ProFound, SAM 主体 | Stage 2 |
| `stage=joint` | 全部非冻结模块 | ProFound | 端到端微调 |

---

## 环境依赖

```bash
pip install -r requirements.txt
```

额外依赖（需手动配置路径）：

- **ProFound**：`../ProFound/`，提供 ConvNeXtV2 编码器和预训练权重
- **SAM-Med3D**：`../SAM-Med3D/`，提供 PromptEncoder3D 和 MaskDecoder3D

在配置文件中指定路径：

```yaml
model:
  profound_checkpoint_path: ../ProFound/checkpoint/checkpoint-799 1.pth
  profound_repo_path: ../ProFound
  profound_model_import_path: "models.convnextv2:convnextv2_tiny"
  sam_checkpoint_path: /path/to/SAM-Med3D/ckpt/sam_med3d_turbo.pth
```

---

## 数据准备

数据由 `../picai_preprocessing_project/` 预处理生成，每个 `.npz` 包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `image` | `[3, D, H, W]` float16 | T2W / ADC / HBV，percentile-clip + z-score 归一化 |
| `label` | `[1, D, H, W]` uint8 | 二值病灶 mask |
| `gland_mask` | `[1, D, H, W]` uint8 | AI 前列腺分割（Bosma22b） |
| `boundary_uncertainty_mask` | `[1, D, H, W]` uint8 | 病灶边界不确定性 mask |
| `metadata_json` | string | 临床信息、处理参数 |

预处理流程：T2W 空间对齐 → 1mm 等距重采样 → gland-bbox 裁剪 → 归一化。

**生成数据划分**（已完成，结果在 `data/splits/5fold/`）：

```bash
python scripts/create_folds.py \
  --processed-root ../picai_preprocessing_project/data/processed/picai_profound_prompt_v2 \
  --num-folds 5 \
  --output-dir data/splits/5fold
```

---

## 训练流程

### Stage 1：粗分割预训练

目标：训练 CoarseBranch 达到 lesion_recall ≥ 0.90，为 Stage 2 提供高质量候选区域。

**当前推荐配置**：`configs/train_pcasam3d_stage1_coarse_v4_data_opt.yaml`

关键设计：
- `normalize: preprocessed`（避免双重 z-score）
- `gland_aware_negative_sampling: true`（90% 负样本在前列腺内，学习更难的假阳性抑制）
- 验证时启用 gland mask 后处理（抑制解剖学不合理的预测）
- 损失：Dice + Focal-Tversky（FN=0.7, FP=0.3）+ BCE，深度监督

```bash
python scripts/train_pcasam3d_profound.py \
  --config configs/train_pcasam3d_stage1_coarse_v4_data_opt.yaml \
  --fold 0 \
  --train-split data/splits/5fold/fold_0/train.txt \
  --val-split data/splits/5fold/fold_0/val.txt \
  --exp-name pcasam3d_stage1_coarse_v4_data_opt
```

**Checkpoint 选择**：用 `best_by_val_threshold_sweep_coarse_score.pth` 作为 Stage 2 的 prompt 来源（多阈值扫描下的最优操作点）。

---

### Stage 2：提示条件化精细化

从 Stage 1 checkpoint 继续训练，解冻 FeatureBridge + ModalityCrossAttention + HighResRefinement，SAM 解码器主体保持冻结。

**当前推荐配置**：`configs/train_pcasam3d_stage2_precision_fp_v4_data_opt.yaml`

关键设计：
- **Prompt Curriculum**：前 8 epoch 使用 GT 提示（100%），之后退火到 30%，让模型逐步适应自动提示的噪声
- **GT Prompt 增强**：GT 边界框 + 随机抖动（jitter_std=0.06），提升对不完美提示的鲁棒性
- **Prompt Dropout**：随机丢弃点/框提示，防止模型依赖单一提示类型
- 损失：Dice + Focal-Tversky（FN=0.6, FP=0.4）+ BCE + 边界 BCE

```bash
python scripts/train_pcasam3d_profound.py \
  --config configs/train_pcasam3d_stage2_precision_fp_v4_data_opt.yaml \
  --fold 0 \
  --train-split data/splits/5fold/fold_0/train.txt \
  --val-split data/splits/5fold/fold_0/val.txt \
  --exp-name pcasam3d_stage2_precision_fp_v4_data_opt
```

**Checkpoint 选择**：用 `best_by_val_refined_sweep_score.pth`（多阈值扫描下综合 Dice/Recall/FP 最优）。

---

## 评估

### 验证集评估（训练中自动进行）

训练脚本每 epoch 自动在验证集上评估，关键指标：

| 指标 | 说明 |
|------|------|
| `val_lesion_recall` | 病灶级召回率（连通域级别） |
| `val_positive_case_dice` | 阳性病例的平均 Dice |
| `val_fp_components_per_case` | 每 case 假阳性连通域数 |
| `val_global_dice` | 全局体素级 Dice |
| `val_refined_sweep_best_score` | 多阈值扫描综合评分（Stage 2 主要选择指标） |

### 单 case 推理

```bash
python scripts/infer_single_case.py \
  --config configs/infer_single_case.yaml \
  --npz-path ../picai_preprocessing_project/data/processed/picai_profound_prompt_v2/all/10005_1000005.npz \
  --checkpoint outputs/pcasam3d_stage2_precision_fp_v4_data_opt/fold_0/checkpoints/best_by_val_refined_sweep_score.pth
```

---

## 调试

**Shape 测试**（不需要 ProFound/SAM 权重）：

```bash
python scripts/debug_pcasam3d_forward.py --mode shapes
```

**完整前向测试**（需要配置权重路径）：

```bash
python scripts/debug_pcasam3d_forward.py \
  --mode full \
  --config configs/train_pcasam3d_stage1_coarse_v4_data_opt.yaml
```

**训练摘要**：

```bash
python scripts/summarize_training_run.py \
  --run-dir outputs/pcasam3d_stage1_coarse_v4_data_opt/fold_0 \
  --output outputs/pcasam3d_stage1_coarse_v4_data_opt/fold_0/reports/training_summary.md
```

---

## 实验结果

详细实验记录见 `docs/experiment_summary.md` 和 `Experiment Log.xlsx`。

### Stage 1 fold_0 结果

| 实验 | coarse_score | Lesion Recall | Pos Dice | FP/case |
|------|-------------|---------------|----------|---------|
| v3_balanced | **1.075** | **0.947** | 0.451 | 1.878 |
| **v4_data_opt** ✓ | 1.054 | 0.929 | **0.457** | 2.034 |

v4 选为推荐：数据处理与 Stage 2 一致，gland-aware 负采样使 global_dice 更高。

### Stage 2 fold_0 结果

| 实验 | sweep_score | Pos Dice | Lesion Recall | FP/case | Global Dice |
|------|------------|----------|---------------|---------|-------------|
| v2_balanced | **0.966** | **0.469** | 0.877 | 1.081 | 0.520 |
| v3_align_adapter | 0.942 | 0.458 | 0.888 | 1.284 | 0.499 |
| v3a_align_only | 0.949 | 0.461 | **0.900** | 1.260 | 0.497 |
| v3c_selfgated | 0.946 | 0.459 | 0.888 | **1.230** | 0.499 |
| **v4_data_opt** ✓ | 0.962 | 0.454 | 0.888 | 1.020 | **0.539** |

v4 的 global_dice 和 FP/case 均最优，与 Stage 1 v4 数据处理完全一致。

### 消融结论

- **Alignment + Adapter 模块无效**：v3 系列全部低于 v2 baseline，已在 v4 中禁用
- **Gland-aware 负采样有效**：显著降低 FP/case，提升 global_dice
- **Objectness head 失效**：所有实验 pos/neg 概率差 <0.001，已禁用（`objectness_weight: 0.0`）

---

## 输出目录规范

所有实验输出统一放在 `outputs/<experiment_name>/fold_<k>/`，详见 `docs/output_layout.md`。

```
outputs/
├── pcasam3d_stage1_<variant>/fold_<k>/
│   ├── checkpoints/   # *.pth（不上传 git）
│   └── logs/          # train_log.csv（不上传 git）
└── pcasam3d_stage2_<variant>/fold_<k>/
    ├── checkpoints/
    └── logs/
```

命名规则：
- `pcasam3d_stage1_<variant>` — Stage 1 粗分割实验
- `pcasam3d_stage2_<variant>` — Stage 2 精细化实验
- 不含硬件信息（不写 `_3090`、`_a100`）

---

## 参考文献

- **ProFound**: Prostate mpMRI foundation model with ConvNeXtV2 backbone.
- **PI-CAI**: Prostate Imaging: Cancer AI challenge dataset. https://pi-cai.grand-challenge.org
- **PCaSAM**: Automatic prompt generation from coarse prostate cancer masks. *Nature Digital Medicine* 2025. https://www.nature.com/articles/s41746-025-01756-2
- **MedSAM**: Box-prompt medical image segmentation. *Nature Communications* 2024. https://www.nature.com/articles/s41467-024-44824-z
- **SAM-Med3D**: Volumetric 3D promptable medical segmentation. arXiv 2310.15161. https://arxiv.org/abs/2310.15161
- **Tversky Loss**: arXiv 1706.05721 — β=0.7 optimal for small lesion recall.
- **Focal Tversky**: arXiv 1810.07842 — γ=4/3 focuses on hard examples.
