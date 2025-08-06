#!/usr/bin/env python3
"""
GradNorm Training Execution Script
GradNorm訓練執行腳本

This script runs the GradNorm solution to solve gradient interference
between detection and classification tasks.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path
import json
from datetime import datetime
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """檢查環境設置"""
    logger.info("="*60)
    logger.info("GRADNORM TRAINING ENVIRONMENT CHECK")
    logger.info("="*60)
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        logger.warning("CUDA not available, will use CPU")
    
    # Check current directory
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Check required files
    required_files = [
        'train_joint_gradnorm.py',
        'models/gradnorm.py',
        'models/common.py',
        '../detection_only_dataset/data.yaml'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✓ {file_path} found")
        else:
            logger.error(f"✗ {file_path} not found")
            return False
    
    return True

def run_gradnorm_training(epochs=10, batch_size=8, device='auto', use_adaptive=True):
    """運行GradNorm訓練"""
    logger.info("="*60)
    logger.info("STARTING GRADNORM TRAINING")
    logger.info("="*60)
    
    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gradnorm_training_{timestamp}"
    
    # Set device
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Prepare training command
    train_cmd = [
        'python3', 'train_joint_gradnorm.py',
        '--data', '../detection_only_dataset/data.yaml',
        '--weights', 'yolov5s.pt',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--imgsz', '320',  # Smaller image size for faster training
        '--device', device,
        '--project', 'runs/gradnorm_training',
        '--name', run_name,
        '--exist-ok',
        '--workers', '0',  # Disable multiprocessing for compatibility
        '--gradnorm-alpha', '1.5',
        '--initial-detection-weight', '1.0',
        '--initial-classification-weight', '0.01',
        '--log-interval', '10',
        '--save-interval', '5'
    ]
    
    # Add adaptive GradNorm if requested
    if use_adaptive:
        train_cmd.append('--use-adaptive-gradnorm')
    
    logger.info(f"Training command: {' '.join(train_cmd)}")
    
    try:
        # Run training
        logger.info("Starting GradNorm training...")
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info("✓ GradNorm training completed successfully")
            logger.info("Training output:")
            logger.info(result.stdout)
            return True, result.stdout, run_name
        else:
            logger.error("✗ GradNorm training failed")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False, result.stderr, run_name
            
    except subprocess.TimeoutExpired:
        logger.error("✗ GradNorm training timed out (1 hour)")
        return False, "Training timed out", run_name
    except Exception as e:
        logger.error(f"✗ GradNorm training error: {e}")
        return False, str(e), run_name

def run_gradnorm_validation(run_name):
    """運行GradNorm驗證"""
    logger.info("="*60)
    logger.info("GRADNORM VALIDATION")
    logger.info("="*60)
    
    # Set device
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Check for weights file
    weights_path = f"runs/gradnorm_training/{run_name}/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = f"runs/gradnorm_training/{run_name}/weights/last.pt"
    
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        return False, "Weights file not found"
    
    # Prepare validation command
    val_cmd = [
        'python3', 'val.py',
        '--data', '../detection_only_dataset/data.yaml',
        '--weights', weights_path,
        '--imgsz', '320',
        '--device', device,
        '--project', 'runs/gradnorm_training',
        '--name', f"{run_name}_validation",
        '--exist-ok'
    ]
    
    logger.info(f"Validation command: {' '.join(val_cmd)}")
    
    try:
        # Run validation
        logger.info("Starting GradNorm validation...")
        result = subprocess.run(val_cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            logger.info("✓ GradNorm validation completed successfully")
            logger.info("Validation output:")
            logger.info(result.stdout)
            return True, result.stdout
        else:
            logger.error("✗ GradNorm validation failed")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("✗ GradNorm validation timed out (5 minutes)")
        return False, "Validation timed out"
    except Exception as e:
        logger.error(f"✗ GradNorm validation error: {e}")
        return False, str(e)

def parse_results(output):
    """解析結果"""
    results = {}
    
    lines = output.split('\n')
    for line in lines:
        # Look for mAP values
        if 'mAP@0.5' in line and 'mAP@0.5:0.95' not in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'mAP@0.5:':
                        if i + 1 < len(parts):
                            results['mAP@0.5'] = float(parts[i + 1])
                        break
            except:
                pass
        
        if 'mAP@0.5:0.95' in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'mAP@0.5:0.95:':
                        if i + 1 < len(parts):
                            results['mAP@0.5:0.95'] = float(parts[i + 1])
                        break
            except:
                pass
    
    return results

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Run GradNorm training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive GradNorm')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("GRADNORM TRAINING EXECUTION")
    logger.info("="*60)
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        return
    
    # Run GradNorm training
    training_success, training_output, run_name = run_gradnorm_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_adaptive=args.adaptive
    )
    
    if not training_success:
        logger.error("GradNorm training failed")
        return
    
    # Parse training results
    training_results = parse_results(training_output)
    logger.info(f"Training results: {training_results}")
    
    # Run GradNorm validation
    validation_success, validation_output = run_gradnorm_validation(run_name)
    
    if validation_success:
        validation_results = parse_results(validation_output)
        logger.info(f"Validation results: {validation_results}")
    else:
        logger.error("GradNorm validation failed")
        validation_results = {}
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_name': run_name,
        'training_success': training_success,
        'training_results': training_results,
        'validation_success': validation_success,
        'validation_results': validation_results,
        'gpu_available': torch.cuda.is_available(),
        'device_used': args.device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'adaptive_gradnorm': args.adaptive
    }
    
    # Save to JSON file
    results_file = f"runs/gradnorm_training/{run_name}/gradnorm_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    logger.info("="*60)
    logger.info("GRADNORM TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Training success: {training_success}")
    if training_results:
        logger.info(f"Training mAP@0.5: {training_results.get('mAP@0.5', 'N/A')}")
        logger.info(f"Training mAP@0.5:0.95: {training_results.get('mAP@0.5:0.95', 'N/A')}")
    
    logger.info(f"Validation success: {validation_success}")
    if validation_results:
        logger.info(f"Validation mAP@0.5: {validation_results.get('mAP@0.5', 'N/A')}")
        logger.info(f"Validation mAP@0.5:0.95: {validation_results.get('mAP@0.5:0.95', 'N/A')}")
    
    logger.info(f"Adaptive GradNorm: {args.adaptive}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    logger.info("="*60)
    logger.info("GRADNORM TRAINING COMPLETED")
    logger.info("="*60)

if __name__ == "__main__":
    main() 