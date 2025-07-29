#!/bin/bash
# Medical Image Complete Training Script
# Combines YOLOv5 detection with ResNet classification for heart ultrasound analysis

echo "=== Medical Image Complete Training System ==="
echo "Detection: Heart Valve Regurgitation (AR, MR, PR, TR)"
echo "Classification: Severity Levels (Normal, Mild, Moderate, Severe)"
echo "================================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if required files exist
if [ ! -f "data_medical_complete.yaml" ]; then
    echo "Error: data_medical_complete.yaml not found!"
    echo "Please ensure the data configuration file exists."
    exit 1
fi

if [ ! -f "hyp_medical_complete.yaml" ]; then
    echo "Error: hyp_medical_complete.yaml not found!"
    echo "Please ensure the hyperparameter file exists."
    exit 1
fi

# Training parameters
DATA_CONFIG="data_medical_complete.yaml"
HYP_CONFIG="hyp_medical_complete.yaml"
EPOCHS=150
BATCH_SIZE=16
IMG_SIZE=640
PROJECT="runs/medical_complete"
NAME="heart_ultrasound_complete"

# Create training command
TRAIN_CMD="python train_medical_complete.py \
    --config config_medical_complete.yaml \
    --epochs ${EPOCHS} \
    --device cuda"

echo "Training Configuration:"
echo "  Data Config: ${DATA_CONFIG}"
echo "  Hyperparameters: ${HYP_CONFIG}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Image Size: ${IMG_SIZE}"
echo "  Project: ${PROJECT}"
echo "  Name: ${NAME}"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. Using CPU training."
    TRAIN_CMD="python train_medical_complete.py --config config_medical_complete.yaml --epochs ${EPOCHS} --device cpu"
fi

# Create config file if it doesn't exist
if [ ! -f "config_medical_complete.yaml" ]; then
    echo "Creating default configuration file..."
    python -c "
import yaml
config = {
    'data_yaml_path': 'data_medical_complete.yaml',
    'project': 'runs/medical_complete',
    'name': 'heart_ultrasound_complete',
    'epochs': ${EPOCHS},
    'nc': 4,
    'classification_classes': 4,
    'data': {
        'batch_size': ${BATCH_SIZE},
        'num_workers': 4,
        'img_size': ${IMG_SIZE}
    },
    'detection': {
        'cfg': 'models/yolov5s.yaml',
        'pretrained': True
    },
    'classification': {
        'type': 'resnet',
        'pretrained': True
    },
    'optimizer': {
        'detection_lr': 0.001,
        'classification_lr': 0.001,
        'weight_decay': 0.0005
    },
    'detection_weight': 1.0,
    'classification_weight': 0.1,
    'classification_weights': [1.0, 1.0, 1.0, 1.0]
}
with open('config_medical_complete.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Configuration file created: config_medical_complete.yaml')
"
fi

echo "Starting training with command:"
echo "${TRAIN_CMD}"
echo ""

# Run training
eval ${TRAIN_CMD}

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Training Completed Successfully ==="
    echo "Results saved to: ${PROJECT}/${NAME}"
    echo "Best model: ${PROJECT}/${NAME}/best.pt"
    echo "Latest model: ${PROJECT}/${NAME}/latest.pt"
    echo ""
    echo "To run inference on a new image:"
    echo "python inference_medical_complete.py --checkpoint ${PROJECT}/${NAME}/best.pt --image path/to/your/image.jpg"
    echo ""
    echo "To view training logs:"
    echo "tensorboard --logdir ${PROJECT}/${NAME}/logs"
else
    echo ""
    echo "=== Training Failed ==="
    echo "Please check the error messages above and try again."
    exit 1
fi 