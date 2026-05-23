#!/usr/bin/env bash
set -euo pipefail

# Run Stage-2 v3 ablations in a fixed, interpretable order:
#   1) v3a: Alignment only
#   2) v3 : Alignment + decoder adapters
#   3) v3c: Alignment + decoder adapters + Self-Gated multiscale
#
# Usage:
#   bash scripts/run_stage2_v3_ablation_sequence.sh
#   FOLD=1 bash scripts/run_stage2_v3_ablation_sequence.sh
#   RUN_FROM_INDEX=2 bash scripts/run_stage2_v3_ablation_sequence.sh   # skip v3a
#
# Logs are saved under each experiment's fold directory as run_stdout.log.

PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/lm/bin/python}"
FOLD="${FOLD:-0}"
RUN_FROM_INDEX="${RUN_FROM_INDEX:-1}"
PROJECT_DIR="${PROJECT_DIR:-/opt/data/private/lm/project-segmentation-for-MIR/profound_coarse_seg_project}"
TRAIN_SPLIT="${TRAIN_SPLIT:-data/splits/5fold/fold_${FOLD}/train.txt}"
VAL_SPLIT="${VAL_SPLIT:-data/splits/5fold/fold_${FOLD}/val.txt}"

cd "${PROJECT_DIR}"

CONFIGS=(
  "configs/train_pcasam3d_stage2_precision_fp_v3a_align_only.yaml"
  "configs/train_pcasam3d_stage2_precision_fp_v3_align_adapter.yaml"
  "configs/train_pcasam3d_stage2_precision_fp_v3c_align_adapter_selfgated.yaml"
)

EXP_NAMES=(
  "pcasam3d_stage2_precision_fp_v3a_align_only"
  "pcasam3d_stage2_precision_fp_v3_align_adapter"
  "pcasam3d_stage2_precision_fp_v3c_align_adapter_selfgated"
)

LABELS=(
  "V3A alignment only"
  "V3 alignment + adapter"
  "V3C alignment + adapter + self-gated"
)

for i in "${!CONFIGS[@]}"; do
  idx=$((i + 1))
  if (( idx < RUN_FROM_INDEX )); then
    echo "[SKIP] ${idx}/3 ${LABELS[$i]}"
    continue
  fi

  cfg="${CONFIGS[$i]}"
  exp="${EXP_NAMES[$i]}"
  label="${LABELS[$i]}"
  out_dir="outputs/${exp}/fold_${FOLD}"
  log_file="${out_dir}/run_stdout.log"

  mkdir -p "${out_dir}"

  echo "======================================================================"
  echo "[RUN] ${idx}/3 ${label}"
  echo "  config: ${cfg}"
  echo "  exp:    ${exp}"
  echo "  fold:   ${FOLD}"
  echo "  train:  ${TRAIN_SPLIT}"
  echo "  val:    ${VAL_SPLIT}"
  echo "  log:    ${log_file}"
  echo "======================================================================"

  "${PYTHON_BIN}" scripts/train_pcasam3d_profound.py \
    --config "${cfg}" \
    --fold "${FOLD}" \
    --train-split "${TRAIN_SPLIT}" \
    --val-split "${VAL_SPLIT}" \
    --exp-name "${exp}" 2>&1 | tee "${log_file}"

  echo "[DONE] ${idx}/3 ${label}"
  echo
 done

echo "All Stage-2 v3 ablation runs finished."
