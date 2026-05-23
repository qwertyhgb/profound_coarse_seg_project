# ProFound-Conv Coarse Lesion Segmentation Project

This is an independent Stage-1 training project for PI-CAI prostate cancer lesion segmentation. It trains a ProFound-Conv driven coarse lesion segmentation model that will later feed an automatic 3D promptable refinement stage.

## Research Rationale

- ProFound is a prostate mpMRI foundation model. This project uses ProFound-Conv, not ProFound-ViT, because dense 3D lesion segmentation benefits from convolutional spatial feature maps, local context, and small-lesion sensitivity.
- PI-CAI bpMRI uses T2W, ADC, and high-b-value DWI/HBV as the core input channels.
- PCaSAM motivates automatic prompt generation from a coarse segmentation mask; MedSAM shows box prompts provide strong spatial context; SAM-Med3D motivates volumetric promptable segmentation with 3D spatial prompts.
- PI-CAI lesions are extremely sparse, so training uses lesion-aware volumetric patch sampling and validation reports both voxel-level metrics and lesion-level recall.

## Stage Scope

Implemented now:

`T2W/ADC/HBV -> ProFound-Conv encoder -> LesionAwareEnhancement3D -> UNetR3D-style decoder -> coarse logits`

Not implemented in Stage 1:

- automatic 3D box prompt generation
- 3D prompt encoder
- prompt-conditioned mask decoder
- refinement branch

Placeholder files exist so Stage 2 can be added without restructuring.

## Data

Default config points to:

```bash
../picai_preprocessing_project/data/processed/picai_profound_prompt_v2
```

Each `.npz` should contain:

- `image`: `[3, D, H, W]`, channel 0 T2W, channel 1 ADC, channel 2 HBV
- `label`: `[1, D, H, W]`
- `case_id`
- optional `metadata_json`, `spacing_after_resample`, `processed_shape`

The project does not copy the 1500 NPZ files.

## ProFound Dependency

This project does not fake ProFound with a generic CNN. To train the real model, configure:

```yaml
model:
  checkpoint_path: "/path/to/profound_conv_checkpoint.pth"
  profound_repo_path: "/path/to/ProFound"
  profound_model_import_path: "module.submodule:build_profound_conv"
```

If these are missing, real model construction raises a clear error. You can still run decoder-only smoke tests.

## Create Splits

```bash
cd profound_coarse_seg_project
/root/anaconda3/envs/lm/bin/python scripts/create_splits.py   --processed-root ../picai_preprocessing_project/data/processed/picai_profound_prompt_v2
```

The split script groups by `patient_id` when available, otherwise by the prefix of `case_id` before `_`.

## Debug Forward

Decoder-only shape test, no ProFound required:

```bash
/root/anaconda3/envs/lm/bin/python scripts/debug_forward.py --mode decoder_only --shape 1 3 64 128 128
```

Real model test, requires configured ProFound:

```bash
/root/anaconda3/envs/lm/bin/python scripts/debug_forward.py --mode model --config configs/train_profound_coarse.yaml
```

## Overfit 8 Cases

```bash
/root/anaconda3/envs/lm/bin/python scripts/overfit_8cases.py --config configs/overfit_8cases.yaml
```

This is a sanity check only. It should reduce train loss and improve Dice when a valid ProFound encoder is configured.

## Train

```bash
/root/anaconda3/envs/lm/bin/python scripts/train.py --config configs/train_profound_coarse.yaml
```

Checkpoints:

- `outputs/checkpoints/best_by_val_dice.pth`
- `outputs/checkpoints/last.pth`

Logs:

- `outputs/logs/train_log.csv`
- `outputs/tensorboard/`

## Validate and Test

```bash
/root/anaconda3/envs/lm/bin/python scripts/validate.py --config configs/train_profound_coarse.yaml --checkpoint outputs/checkpoints/best_by_val_dice.pth
/root/anaconda3/envs/lm/bin/python scripts/test.py --config configs/train_profound_coarse.yaml --checkpoint outputs/checkpoints/best_by_val_dice.pth
```

## Single-Case Inference

```bash
/root/anaconda3/envs/lm/bin/python scripts/infer_single_case.py   --config configs/infer_single_case.yaml   --npz-path ../picai_preprocessing_project/data/processed/picai_profound_prompt_v2/all/10005_1000005.npz   --checkpoint outputs/checkpoints/best_by_val_dice.pth
```

Outputs are saved under `outputs/predictions/` as compressed NPZ containing logits, probability, and binary mask. These coarse probability maps are intended to drive later automatic prompt generation.

## Freeze Encoder

```yaml
model:
  freeze_encoder: true
```

When frozen, encoder parameters are excluded from the optimizer. Enhancement and decoder use `head_lr`; encoder uses `encoder_lr` only when trainable.

## References

- ProFound: prostate mpMRI foundation model and ProFound-Conv backbone.
- PI-CAI: prostate bpMRI cancer detection/localization dataset.
- PCaSAM: automatic prompt generation from coarse prostate cancer masks.
- MedSAM: box-prompt medical image segmentation.
- SAM-Med3D: volumetric 3D promptable medical segmentation.


## 5-Fold Cross-Validation

Stage 1 should be evaluated with the same cross-validation discipline as the later prompt/refinement stages, because the coarse probability map determines automatic 3D box, centroid, and uncertainty prompts. Generate patient-level stratified folds with:

```bash
python scripts/create_folds.py \
  --processed-root ../picai_preprocessing_project/data/processed/picai_profound_prompt_v2 \
  --num-folds 5 \
  --output-dir data/splits/5fold
```

Train one fold with separated outputs:

```bash
python scripts/train.py \
  --config configs/train_profound_coarse.yaml \
  --fold 0 \
  --train-split data/splits/5fold/fold_0/train.txt \
  --val-split data/splits/5fold/fold_0/val.txt
```

Evaluate the held-out fold:

```bash
python scripts/test.py \
  --config configs/train_profound_coarse.yaml \
  --checkpoint outputs/fold_0/checkpoints/best_by_val_dice.pth \
  --split data/splits/5fold/fold_0/test.txt
```

Do not train a single Stage-1 model on all cases and use it to generate prompts for validation/test folds; that leaks information into the downstream promptable segmentation experiment.


## Recall-Oriented Coarse Optimization

For Stage 1, missing a lesion is more harmful than predicting a slightly larger coarse region because downstream automatic prompts are generated from the coarse probability map. The recall-oriented config uses:

- lesion-aware sampling with more positive exposure,
- Dice + Focal-Tversky + lightly weighted BCE,
- `fn_weight > fp_weight` to penalize false negatives more strongly,
- validation threshold `0.3` plus threshold sweep,
- additional best checkpoints by `lesion_recall`, `positive_case_dice`, and `coarse_score`.

Run one fold with:

```bash
python scripts/train.py \
  --config configs/train_profound_coarse_recall.yaml \
  --fold 0 \
  --train-split data/splits/5fold/fold_0/train.txt \
  --val-split data/splits/5fold/fold_0/val.txt
```

For downstream prompt generation, prefer `best_by_val_coarse_score.pth` or `best_by_val_lesion_recall.pth` over a checkpoint selected only by voxel Dice.


## Experiment Naming

All runs stay under `outputs/`. Use a name so experiments do not overwrite each other:

```bash
python scripts/train.py --config configs/train_profound_coarse_recall.yaml --exp-name recall_ftversky --fold 0 \
  --train-split data/splits/5fold/fold_0/train.txt \
  --val-split data/splits/5fold/fold_0/val.txt
```

This writes to `outputs/recall_ftversky/fold_0/`.


## Stage-1 Coarse Proposal Checkpointing

The current Stage-1 training policy treats the network as a high-recall coarse proposal model for Stage 2 prompts, not as the final segmentation model. By default, `configs/train_profound_coarse_recall_3090.yaml` monitors `val_coarse_score` for early stopping:

```yaml
early_stopping:
  enabled: true
  monitor: "val_coarse_score"
  mode: "max"
  patience: 10
  min_delta: 0.0005
```

`coarse_score` is computed from lesion recall, positive-case Dice, and a small false-positive component penalty. This favors checkpoints that still find lesions and can produce useful box/point/uncertainty prompts.

The trainer saves these checkpoints independently:

- `best_by_val_coarse_score.pth`
- `best_by_val_threshold_sweep_coarse_score.pth`
- `best_by_val_lesion_recall.pth`
- `best_by_val_positive_case_dice.pth`
- `best_by_val_dice.pth`
- `last.pth`

Each validation epoch also runs a threshold sweep, configured for example as:

```yaml
threshold_sweep:
  enabled: true
  thresholds: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
```

`logs/train_log.csv` stores compact epoch-level metrics, while `logs/threshold_sweep_log.csv` stores one row per threshold per epoch.

For single-case inference, keep segmentation reporting and prompt generation separate:

```yaml
inference:
  segmentation_threshold: 0.50
  prompt_generation_threshold: 0.15
```

The output `.npz` contains the coarse probability map plus both masks, so Stage 2 can use the lower prompt threshold without changing traditional segmentation reporting.

Recall-oriented loss ablations are provided in:

- `configs/recall_stronger_v1.yaml`: Tversky FN/FP = 0.8/0.2
- `configs/recall_stronger_v2.yaml`: Tversky FN/FP = 0.9/0.1

Summarize a completed run with:

```bash
python scripts/summarize_training_run.py   --run-dir outputs/coarse_score_es_3090/fold_0   --output outputs/reports/training_summary.md
```

For Stage 2 prompt generation, start from `best_by_val_threshold_sweep_coarse_score.pth` when it has high lesion recall and acceptable false positives per case; otherwise use `best_by_val_coarse_score.pth`.


## Stage 2 Prompt-Conditioned Refinement V2

Stage 2 is now a trainable coarse-to-fine refinement stage rather than a placeholder. It follows the direction used in promptable medical segmentation research:

- PCaSAM first produces a coarse mask and then uses morphological post-processing to generate automatic bounding boxes for a prompt-guided model: https://www.nature.com/articles/s41746-025-01756-2
- MedSAM shows that a bounding-box prompt gives strong spatial context for medical segmentation: https://www.nature.com/articles/s41467-024-44824-z
- SAM-Med3D motivates fully volumetric promptable segmentation with 3D prompts: https://arxiv.org/abs/2310.15161

Current Stage 2 design:

- Stage 1 produces coarse probability maps from the ProFound-Conv coarse model.
- `AutoPromptGenerator3D` / proposal post-processing converts coarse components into 3D boxes, center points, and optional uncertainty points.
- `Stage2PromptDataset` crops proposal-centered 3D patches and returns image, label, coarse probability, box prior, point Gaussian prior, and objectness label.
- `CoarsePromptRefinementModel` predicts a refined mask and an optional objectness score for proposal filtering/ranking.
- Case-level evaluation merges refined proposal patches back to full volumes and reports Dice, precision, recall, lesion recall, components/case, false-positive components/case, and component precision.

Key modules:

- `src/models/prompts/auto_prompt_generator.py`: reusable 3D prompt generation from coarse probability maps.
- `src/models/prompts/prompt_encoder_3d.py`: dense 3D prompt prior construction and prompt feature encoding.
- `src/datasets/stage2_prompt_dataset.py`: proposal patch dataset with GT-overlap balancing for supervised training.
- `src/models/refinement/coarse_prompt_refinement_model.py`: prompt-conditioned 3D mask refinement plus objectness head.
- `scripts/train_stage2_refinement.py`: Stage-2 multi-task training with mask loss and objectness loss.
- `scripts/evaluate_stage2_case_level.py`: merges proposal predictions back to case-level volumes and evaluates Stage-2 outputs.

Prepare Stage-2 data for fold 0 using the selected Stage-1 checkpoint and fixed post-processing strategy:

```bash
python scripts/precompute_stage1_coarse.py \
  --config configs/infer_single_case.yaml \
  --split data/splits/5fold/fold_0/train.txt \
  --output-dir outputs/coarse_score_es_3090/fold_0/stage2_data/train/coarse_predictions \
  --report-dir outputs/coarse_score_es_3090/fold_0/stage2_data/train/proposal_reports

python scripts/sweep_proposal_postprocess.py \
  --component-json outputs/coarse_score_es_3090/fold_0/stage2_data/train/proposal_reports/component_details.json \
  --output-dir outputs/coarse_score_es_3090/fold_0/stage2_data/train/postprocess_sweep \
  --min-component-sizes 50 \
  --min-max-probabilities 0.5 \
  --top-k 5 \
  --rank-by max_probability \
  --include-gt-hit-in-prompts

python scripts/precompute_stage1_coarse.py \
  --config configs/infer_single_case.yaml \
  --split data/splits/5fold/fold_0/val.txt \
  --output-dir outputs/coarse_score_es_3090/fold_0/stage2_data/val/coarse_predictions \
  --report-dir outputs/coarse_score_es_3090/fold_0/stage2_data/val/proposal_reports

python scripts/sweep_proposal_postprocess.py \
  --component-json outputs/coarse_score_es_3090/fold_0/stage2_data/val/proposal_reports/component_details.json \
  --output-dir outputs/coarse_score_es_3090/fold_0/stage2_data/val/postprocess_sweep \
  --min-component-sizes 50 \
  --min-max-probabilities 0.5 \
  --top-k 5 \
  --rank-by max_probability \
  --include-gt-hit-in-prompts
```

Train Stage 2 V2. The default output directory is `outputs/coarse_score_es_3090/fold_0/stage2_refinement_v2_objectness`, so it will not overwrite V1/V1-balanced runs:

```bash
python scripts/train_stage2_refinement.py --config configs/train_stage2_refinement.yaml
```

Evaluate the trained Stage-2 checkpoint at full-case level:

```bash
python scripts/evaluate_stage2_case_level.py \
  --config configs/train_stage2_refinement.yaml \
  --checkpoint outputs/coarse_score_es_3090/fold_0/stage2_refinement_v2_objectness/checkpoints/best_by_val_recall_safe_dice.pth \
  --output-dir outputs/coarse_score_es_3090/fold_0/stage2_refinement_v2_objectness/case_level_eval \
  --mask-threshold 0.5
```

For objectness experiments, add `--use-objectness-filter --objectness-threshold 0.5` to reject low-quality proposals, or `--weight-by-objectness` to softly weight merged patch probabilities. Do not enable these options for final reporting until a V2 checkpoint has trained the objectness head.



## Output Layout Discipline

All formal outputs are organized under `outputs/<experiment_name>/fold_<k>/`. See `docs/output_layout.md` before adding new experiments or debug outputs.


## PCaSAM-3D-ProFound: 端到端统一模型

PCaSAM-3D-ProFound 是本项目的核心创新方案，将三个关键技术统一到一个端到端可训练的模型中：

1. **ProFound-Conv**（前列腺 mpMRI 基础模型）作为领域特定图像编码器
2. **PCaSAM 风格的自动 3D 提示生成**：从粗分割分支自动产生 3D 框/点提示
3. **SAM-Med3D 的提示编码器 + 掩码解码器**：基于提示的精细化分割

### 架构

```
Input [B, 3, D, H, W] (T2W/ADC/HBV mpMRI)
    │
    ▼
┌─ ProFound-Conv Encoder (预训练，冻结) ──────────────────┐
│  stage1: [B, 96,  D/4,  H/4,  W/4]                     │
│  stage2: [B, 192, D/8,  H/8,  W/8]                     │
│  stage3: [B, 384, D/16, H/16, W/16]                    │
│  stage4: [B, 768, D/32, H/32, W/32]                    │
└──────────────────────────────────────────────────────────┘
    │                           │
    ▼                           ▼
┌─ Coarse Branch ─┐     ┌─ Feature Bridge ─────────────┐
│  FPN → logits   │     │  FPN → [B, 384, 8, 8, 8]    │
│  [B,1,D,H,W]   │     │  (SAM 嵌入空间)              │
└─────────────────┘     └──────────────────────────────┘
    │                           │
    ▼                           │
┌─ Auto Prompt 3D ─┐           │
│  point_coords     │           │
│  mask_prior       │           │
└───────────────────┘           │
    │                           │
    ▼                           ▼
┌─ SAM-Med3D Prompt Encoder ─┐  │
│  sparse_emb, dense_emb     │  │
└────────────────────────────┘  │
    │                           │
    ▼                           ▼
┌─ SAM-Med3D Mask Decoder ─────────────────────────────┐
│  TwoWayTransformer3D + hypernetwork MLP              │
│  → refined_logits [B, 1, D, H, W]                   │
│  → iou_pred [B, 1]                                   │
└──────────────────────────────────────────────────────┘
```

### 关键模块

- `src/models/pcasam3d_profound/pcasam3d_profound_model.py`: 统一模型主类
- `src/models/pcasam3d_profound/feature_bridge.py`: ProFound 多尺度特征 → SAM 384 维嵌入空间
- `src/models/pcasam3d_profound/coarse_branch.py`: 轻量级粗分割分支（辅助监督）
- `src/models/pcasam3d_profound/auto_prompt_3d.py`: PCaSAM 风格自动 3D 提示生成
- `src/models/pcasam3d_profound/prompt_adapter.py`: 提示格式适配（归一化坐标 → SAM 绝对坐标）
- `src/models/pcasam3d_profound/pcasam3d_loss.py`: 多任务损失（精细化 + 粗分割 + IoU 预测）
- `src/datasets/pcasam3d_dataset.py`: 端到端训练数据集（病灶感知采样 + 128³ resize）

### 训练

```bash
# 单折训练
python scripts/train_pcasam3d_profound.py \
  --config configs/train_pcasam3d_profound.yaml \
  --fold 0 \
  --train-split data/splits/5fold/fold_0/train.txt \
  --val-split data/splits/5fold/fold_0/val.txt

# 5 折交叉验证
bash scripts/train_pcasam3d_5fold.sh configs/train_pcasam3d_profound.yaml pcasam3d_v1
```

### 评估

```bash
python scripts/evaluate_pcasam3d_profound.py \
  --config configs/train_pcasam3d_profound.yaml \
  --checkpoint outputs/pcasam3d_profound/fold_0/checkpoints/best_by_val_recall_safe_dice.pth \
  --split data/splits/5fold/fold_0/val.txt \
  --output-dir outputs/pcasam3d_profound/fold_0/evaluation
```

### 调试

```bash
# Shape 测试（不需要 ProFound 权重）
python scripts/debug_pcasam3d_forward.py --mode shapes

# 完整模型测试（需要 ProFound + SAM-Med3D 权重）
python scripts/debug_pcasam3d_forward.py --mode full --config configs/train_pcasam3d_profound.yaml
```

### 训练策略

1. **阶段 1**：冻结 ProFound 编码器，训练 Feature Bridge + Coarse Branch + SAM Decoder（100 epochs）
2. **阶段 2**：解冻 ProFound 最后 2 个 stage，低学习率端到端微调（50 epochs）

```bash
# 阶段 2 微调
python scripts/train_pcasam3d_profound.py \
  --config configs/train_pcasam3d_profound_finetune.yaml \
  --fold 0 \
  --train-split data/splits/5fold/fold_0/train.txt \
  --val-split data/splits/5fold/fold_0/val.txt
```

### 模型参数

- 总参数: ~66.8M
- 可训练参数（冻结编码器）: ~37.5M
- ProFound-Conv 编码器: ~28.6M（冻结）
- Feature Bridge + Coarse Branch: ~5.2M
- SAM Prompt Encoder + Mask Decoder: ~32.3M

### 损失函数

多任务损失 = refined_weight × L_refined + coarse_weight × L_coarse + iou_weight × L_iou

- L_refined = Dice + Focal-Tversky (FN=0.7, FP=0.3) + BCE (pos_weight=3.0)
- L_coarse = Dice + BCE (pos_weight=2.0)
- L_iou = MSE(iou_pred, actual_dice)

### 与其他方案的对比

| 方案 | 编码器 | 提示方式 | 解码器 | 端到端 |
|------|--------|----------|--------|--------|
| Stage-1 ProFound-Conv | ProFound-Conv | 无 | UNetR3D | 否 |
| Stage-2 V2 Refinement | 无（用粗分割特征） | 手动后处理 | 自定义 CNN | 否 |
| Stage-2 SAM-Med3D V1 | SAM-Med3D ViT | 手动后处理 | SAM Mask Decoder | 否 |
| **PCaSAM-3D-ProFound** | **ProFound-Conv** | **自动可微分** | **SAM Mask Decoder** | **是** |

核心优势：
- 端到端训练，梯度从 SAM 解码器流回粗分割分支，自动优化提示质量
- ProFound-Conv 提供领域特定的前列腺 MRI 特征（优于通用 ViT）
- SAM-Med3D 的 Mask Decoder 提供强大的提示条件化分割能力
- 无需手动后处理步骤，推理时自动生成提示
