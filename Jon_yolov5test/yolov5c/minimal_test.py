#!/usr/bin/env python3
"""
Minimal Test for Original YOLOv5 Training
最小化原始YOLOv5訓練測試

This script provides a minimal test to validate the original YOLOv5 training
and get baseline performance metrics.
"""

import os
import sys
import subprocess
import torch
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_minimal_training():
    """運行最小化訓練"""
    logger.info("="*60)
    logger.info("MINIMAL TRAINING TEST")
    logger.info("="*60)
    
    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"minimal_test_{timestamp}"
    
    # Set device
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Prepare training command with minimal settings
    train_cmd = [
        'python3', 'train.py',
        '--data', '../detection_only_dataset/data.yaml',
        '--weights', 'yolov5s.pt',
        '--epochs', '1',  # Only 1 epoch for minimal test
        '--batch-size', '2',  # Very small batch size
        '--imgsz', '128',  # Very small image size
        '--device', device,
        '--project', 'runs/minimal_test',
        '--name', run_name,
        '--exist-ok',
        '--workers', '0',  # Disable multiprocessing
        '--noval',  # Skip validation during training
        '--nosave'  # Don't save plots
    ]
    
    logger.info(f"Training command: {' '.join(train_cmd)}")
    
    try:
        # Run training
        logger.info("Starting minimal training...")
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            logger.info("✓ Minimal training completed successfully")
            logger.info("Training output:")
            logger.info(result.stdout)
            return True, result.stdout, run_name
        else:
            logger.error("✗ Minimal training failed")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False, result.stderr, run_name
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Minimal training timed out (10 minutes)")
        return False, "Training timed out", run_name
    except Exception as e:
        logger.error(f"✗ Minimal training error: {e}")
        return False, str(e), run_name

def main():
    """主函數"""
    logger.info("="*60)
    logger.info("MINIMAL TRAINING TEST")
    logger.info("="*60)
    
    # Check environment
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Check dataset
    data_yaml_path = "../detection_only_dataset/data.yaml"
    if not os.path.exists(data_yaml_path):
        logger.error(f"Dataset not found: {data_yaml_path}")
        return
    
    logger.info(f"Dataset found: {data_yaml_path}")
    
    # Run minimal training
    training_success, training_output, run_name = run_minimal_training()
    
    if not training_success:
        logger.error("Minimal training failed")
        return
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_name': run_name,
        'training_success': training_success,
        'training_output': training_output,
        'gpu_available': torch.cuda.is_available(),
        'device_used': '0' if torch.cuda.is_available() else 'cpu'
    }
    
    # Save to JSON file
    results_file = f"runs/minimal_test/{run_name}/minimal_test_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        import json
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    logger.info("="*60)
    logger.info("MINIMAL TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Training success: {training_success}")
    
    logger.info("="*60)
    logger.info("MINIMAL TEST COMPLETED")
    logger.info("="*60)

if __name__ == "__main__":
    main() 