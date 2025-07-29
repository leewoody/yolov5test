#!/bin/bash
# Medical Image Training Script for YOLOv5
# Optimized for heart ultrasound and medical imaging applications

echo "=== Medical Image YOLOv5 Training ==="
echo "Dataset: Heart Valve Regurgitation Detection"
echo "Classes: AR, MR, PR, TR"
echo "======================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training parameters
DATA_CONFIG="data_regurgitation.yaml"
HYP_CONFIG="hyp_medical.yaml"
WEIGHTS="yolov5s.pt"
EPOCHS=150
BATCH_SIZE=16
IMG_SIZE=640
PROJECT="runs/medical_train"
NAME="heart_valve_regurgitation"

# Create training command
TRAIN_CMD="python train_medical.py \
    --data ${DATA_CONFIG} \
    --hyp ${HYP_CONFIG} \
    --weights ${WEIGHTS} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --imgsz ${IMG_SIZE} \
    --project ${PROJECT} \
    --name ${NAME} \
    --medical-mode \
    --patience 50 \
    --save-period 25"

echo "Starting training with command:"
echo "${TRAIN_CMD}"
echo ""

# Run training
eval ${TRAIN_CMD}

echo ""
echo "=== Training Completed ==="
echo "Results saved to: ${PROJECT}/${NAME}"
echo "Best model: ${PROJECT}/${NAME}/weights/best.pt"
echo "Last model: ${PROJECT}/${NAME}/weights/last.pt" 