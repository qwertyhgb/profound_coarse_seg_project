#!/bin/bash
# PCaSAM-3D-ProFound 5-fold cross-validation training script.
#
# Usage:
#   bash scripts/train_pcasam3d_5fold.sh [CONFIG] [EXP_NAME]
#
# Example:
#   bash scripts/train_pcasam3d_5fold.sh configs/train_pcasam3d_profound.yaml pcasam3d_v1

set -e

CONFIG="${1:-configs/train_pcasam3d_profound.yaml}"
EXP_NAME="${2:-pcasam3d_v1}"
PYTHON="/root/anaconda3/envs/lm/bin/python"
SPLIT_DIR="data/splits/5fold"

echo "============================================================"
echo "PCaSAM-3D-ProFound 5-Fold Cross-Validation"
echo "Config: ${CONFIG}"
echo "Experiment: ${EXP_NAME}"
echo "============================================================"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Starting Fold ${FOLD} / 4"
    echo "------------------------------------------------------------"

    TRAIN_SPLIT="${SPLIT_DIR}/fold_${FOLD}/train.txt"
    VAL_SPLIT="${SPLIT_DIR}/fold_${FOLD}/val.txt"

    if [ ! -f "${TRAIN_SPLIT}" ]; then
        echo "ERROR: ${TRAIN_SPLIT} not found. Run scripts/create_folds.py first."
        exit 1
    fi

    ${PYTHON} scripts/train_pcasam3d_profound.py \
        --config "${CONFIG}" \
        --fold ${FOLD} \
        --train-split "${TRAIN_SPLIT}" \
        --val-split "${VAL_SPLIT}" \
        --exp-name "${EXP_NAME}"

    echo "Fold ${FOLD} complete."
done

echo ""
echo "============================================================"
echo "All 5 folds complete!"
echo "Results in: outputs/${EXP_NAME}/fold_*/checkpoints/"
echo "============================================================"
