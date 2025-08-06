#!/bin/bash

# Original Code Training Validation Script
# 原始代碼訓練驗證腳本

echo "=================================================="
echo "ORIGINAL CODE TRAINING VALIDATION"
echo "=================================================="

# Set working directory
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: train.py not found in current directory"
    exit 1
fi

if [ ! -f "val.py" ]; then
    echo "Error: val.py not found in current directory"
    exit 1
fi

# Check if data.yaml exists
DATA_YAML="../Regurgitation-YOLODataset-1new/data.yaml"
if [ ! -f "$DATA_YAML" ]; then
    echo "Error: $DATA_YAML not found"
    exit 1
fi

echo "✓ Found required files"

# Create timestamp for unique run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="original_validation_${TIMESTAMP}"

echo "Run name: $RUN_NAME"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU available"
    DEVICE="0"
else
    echo "⚠ No GPU detected, using CPU"
    DEVICE="cpu"
fi

# Function to run training
run_training() {
    echo "=================================================="
    echo "STARTING TRAINING"
    echo "=================================================="
    
    python train.py \
        --data "$DATA_YAML" \
        --weights yolov5s.pt \
        --epochs 50 \
        --batch-size 16 \
        --imgsz 640 \
        --device "$DEVICE" \
        --project runs/original_validation \
        --name "$RUN_NAME" \
        --exist-ok \
        --patience 50 \
        --save-period 10
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully"
        return 0
    else
        echo "✗ Training failed"
        return 1
    fi
}

# Function to run validation
run_validation() {
    echo "=================================================="
    echo "STARTING VALIDATION"
    echo "=================================================="
    
    # Check for best weights first, then last weights
    WEIGHTS_PATH="runs/original_validation/$RUN_NAME/weights/best.pt"
    if [ ! -f "$WEIGHTS_PATH" ]; then
        WEIGHTS_PATH="runs/original_validation/$RUN_NAME/weights/last.pt"
    fi
    
    if [ ! -f "$WEIGHTS_PATH" ]; then
        echo "✗ No weights file found for validation"
        return 1
    fi
    
    echo "Using weights: $WEIGHTS_PATH"
    
    python val.py \
        --data "$DATA_YAML" \
        --weights "$WEIGHTS_PATH" \
        --imgsz 640 \
        --device "$DEVICE" \
        --project runs/original_validation \
        --name "${RUN_NAME}_validation" \
        --exist-ok
    
    if [ $? -eq 0 ]; then
        echo "✓ Validation completed successfully"
        return 0
    else
        echo "✗ Validation failed"
        return 1
    fi
}

# Function to run quick test
run_quick_test() {
    echo "=================================================="
    echo "STARTING QUICK TEST"
    echo "=================================================="
    
    python quick_original_test.py
    
    if [ $? -eq 0 ]; then
        echo "✓ Quick test completed successfully"
        return 0
    else
        echo "✗ Quick test failed"
        return 1
    fi
}

# Main execution
echo "Choose an option:"
echo "1. Quick test (5 epochs, small batch size)"
echo "2. Full training (50 epochs, normal batch size)"
echo "3. Both quick test and full training"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "Running quick test..."
        run_quick_test
        ;;
    2)
        echo "Running full training..."
        if run_training; then
            run_validation
        fi
        ;;
    3)
        echo "Running both quick test and full training..."
        echo "First: Quick test"
        run_quick_test
        echo ""
        echo "Second: Full training"
        if run_training; then
            run_validation
        fi
        ;;
    *)
        echo "Invalid choice. Running quick test by default..."
        run_quick_test
        ;;
esac

echo "=================================================="
echo "VALIDATION COMPLETED"
echo "=================================================="
echo "Results saved in: runs/original_validation/"
echo "Run name: $RUN_NAME"
echo "==================================================" 