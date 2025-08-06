#!/usr/bin/env python3
"""
Simple Training Test
簡化訓練測試

This script provides a simple way to test the original YOLOv5 training
without complex data loading issues.
"""

import os
import sys
import subprocess
import yaml
import torch
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_simple_training():
    """運行簡化訓練"""
    logger.info("="*60)
    logger.info("SIMPLE TRAINING TEST")
    logger.info("="*60)
    
    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"simple_test_{timestamp}"
    
    # Set device
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Prepare training command with minimal settings
    train_cmd = [
        'python3', 'train.py',
        '--data', '../detection_only_dataset/data.yaml',
        '--weights', 'yolov5s.pt',
        '--epochs', '3',  # Very short training
        '--batch-size', '4',  # Small batch size
        '--imgsz', '256',  # Small image size
        '--device', device,
        '--project', 'runs/simple_test',
        '--name', run_name,
        '--exist-ok',
        '--workers', '0',  # Disable multiprocessing
        '--cache',  # Cache images
        '--patience', '5'  # Early stopping
    ]
    
    logger.info(f"Training command: {' '.join(train_cmd)}")
    
    try:
        # Run training
        logger.info("Starting simple training...")
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=900)  # 15 min timeout
        
        if result.returncode == 0:
            logger.info("✓ Simple training completed successfully")
            logger.info("Training output:")
            logger.info(result.stdout)
            return True, result.stdout, run_name
        else:
            logger.error("✗ Simple training failed")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False, result.stderr, run_name
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Simple training timed out (15 minutes)")
        return False, "Training timed out", run_name
    except Exception as e:
        logger.error(f"✗ Simple training error: {e}")
        return False, str(e), run_name

def run_simple_validation(run_name):
    """運行簡化驗證"""
    logger.info("="*60)
    logger.info("SIMPLE VALIDATION")
    logger.info("="*60)
    
    # Set device
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Check for weights file
    weights_path = f"runs/simple_test/{run_name}/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = f"runs/simple_test/{run_name}/weights/last.pt"
    
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        return False, "Weights file not found"
    
    # Prepare validation command
    val_cmd = [
        'python3', 'val.py',
        '--data', '../detection_only_dataset/data.yaml',
        '--weights', weights_path,
        '--imgsz', '256',
        '--device', device,
        '--project', 'runs/simple_test',
        '--name', f"{run_name}_validation",
        '--exist-ok'
    ]
    
    logger.info(f"Validation command: {' '.join(val_cmd)}")
    
    try:
        # Run validation
        logger.info("Starting simple validation...")
        result = subprocess.run(val_cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            logger.info("✓ Simple validation completed successfully")
            logger.info("Validation output:")
            logger.info(result.stdout)
            return True, result.stdout
        else:
            logger.error("✗ Simple validation failed")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Simple validation timed out (5 minutes)")
        return False, "Validation timed out"
    except Exception as e:
        logger.error(f"✗ Simple validation error: {e}")
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
    logger.info("="*60)
    logger.info("SIMPLE TRAINING TEST")
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
    
    # Run simple training
    training_success, training_output, run_name = run_simple_training()
    
    if not training_success:
        logger.error("Simple training failed")
        return
    
    # Parse training results
    training_results = parse_results(training_output)
    logger.info(f"Training results: {training_results}")
    
    # Run simple validation
    validation_success, validation_output = run_simple_validation(run_name)
    
    if validation_success:
        validation_results = parse_results(validation_output)
        logger.info(f"Validation results: {validation_results}")
    else:
        logger.error("Simple validation failed")
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
        'device_used': '0' if torch.cuda.is_available() else 'cpu'
    }
    
    # Save to JSON file
    results_file = f"runs/simple_test/{run_name}/simple_test_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    logger.info("="*60)
    logger.info("SIMPLE TEST SUMMARY")
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
    
    logger.info("="*60)
    logger.info("SIMPLE TEST COMPLETED")
    logger.info("="*60)

if __name__ == "__main__":
    main() 