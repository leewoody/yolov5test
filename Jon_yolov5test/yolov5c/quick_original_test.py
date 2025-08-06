#!/usr/bin/env python3
"""
Quick Original Code Test
快速測試原始代碼

This script provides a quick test of the original YOLOv5 training code
to validate the baseline performance before implementing solutions.
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

def check_environment():
    """檢查環境設置"""
    logger.info("="*50)
    logger.info("ENVIRONMENT CHECK")
    logger.info("="*50)
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA not available, will use CPU")
    
    # Check current directory
    logger.info(f"Current directory: {os.getcwd()}")
    
    # Check if train.py exists
    train_py_path = Path("train.py")
    if train_py_path.exists():
        logger.info("✓ train.py found")
    else:
        logger.error("✗ train.py not found")
        return False
    
    # Check if val.py exists
    val_py_path = Path("val.py")
    if val_py_path.exists():
        logger.info("✓ val.py found")
    else:
        logger.error("✗ val.py not found")
        return False
    
    return True

def check_dataset(data_yaml_path):
    """檢查數據集"""
    logger.info("="*50)
    logger.info("DATASET CHECK")
    logger.info("="*50)
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        logger.info(f"Dataset config: {data_yaml_path}")
        logger.info(f"Classes: {data_config.get('names', [])}")
        logger.info(f"Number of classes: {data_config.get('nc', 0)}")
        
        # Check paths
        train_path = Path(data_config.get('train', ''))
        val_path = Path(data_config.get('val', ''))
        test_path = Path(data_config.get('test', ''))
        
        logger.info(f"Train path: {train_path}")
        logger.info(f"  Exists: {train_path.exists()}")
        if train_path.exists():
            train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
            logger.info(f"  Images: {len(train_images)}")
        
        logger.info(f"Val path: {val_path}")
        logger.info(f"  Exists: {val_path.exists()}")
        if val_path.exists():
            val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
            logger.info(f"  Images: {len(val_images)}")
        
        logger.info(f"Test path: {test_path}")
        logger.info(f"  Exists: {test_path.exists()}")
        if test_path.exists():
            test_images = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
            logger.info(f"  Images: {len(test_images)}")
        
        return data_config
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def run_quick_training(data_yaml_path, epochs=5, batch_size=8):
    """運行快速訓練"""
    logger.info("="*50)
    logger.info("QUICK TRAINING")
    logger.info("="*50)
    
    # Create unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"quick_test_{timestamp}"
    
    # Set device
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Prepare training command
    train_cmd = [
        'python3', 'train.py',
        '--data', data_yaml_path,
        '--weights', 'yolov5s.pt',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--imgsz', '320',  # Smaller image size for quick test
        '--device', device,
        '--project', 'runs/quick_test',
        '--name', run_name,
        '--exist-ok',
        '--patience', '10'  # Early stopping for quick test
    ]
    
    logger.info(f"Training command: {' '.join(train_cmd)}")
    
    try:
        # Run training
        logger.info("Starting training...")
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            logger.info("✓ Training completed successfully")
            logger.info("Training output:")
            logger.info(result.stdout)
            return True, result.stdout, run_name
        else:
            logger.error("✗ Training failed")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False, result.stderr, run_name
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Training timed out (30 minutes)")
        return False, "Training timed out", run_name
    except Exception as e:
        logger.error(f"✗ Training error: {e}")
        return False, str(e), run_name

def run_validation(data_yaml_path, weights_path, run_name):
    """運行驗證"""
    logger.info("="*50)
    logger.info("VALIDATION")
    logger.info("="*50)
    
    # Set device
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Prepare validation command
    val_cmd = [
        'python3', 'val.py',
        '--data', data_yaml_path,
        '--weights', weights_path,
        '--imgsz', '320',
        '--device', device,
        '--project', 'runs/quick_test',
        '--name', f"{run_name}_validation",
        '--exist-ok'
    ]
    
    logger.info(f"Validation command: {' '.join(val_cmd)}")
    
    try:
        # Run validation
        logger.info("Starting validation...")
        result = subprocess.run(val_cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            logger.info("✓ Validation completed successfully")
            logger.info("Validation output:")
            logger.info(result.stdout)
            return True, result.stdout
        else:
            logger.error("✗ Validation failed")
            logger.error("Error output:")
            logger.error(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Validation timed out (5 minutes)")
        return False, "Validation timed out"
    except Exception as e:
        logger.error(f"✗ Validation error: {e}")
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
        
        # Look for precision and recall
        if 'precision' in line.lower():
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'precision' in part.lower():
                        if i + 1 < len(parts):
                            results['precision'] = float(parts[i + 1])
                        break
            except:
                pass
        
        if 'recall' in line.lower():
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'recall' in part.lower():
                        if i + 1 < len(parts):
                            results['recall'] = float(parts[i + 1])
                        break
            except:
                pass
    
    return results

def main():
    """主函數"""
    logger.info("="*60)
    logger.info("QUICK ORIGINAL CODE TEST")
    logger.info("="*60)
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        return
    
    # Check dataset
    data_yaml_path = "../detection_only_dataset/data.yaml"
    data_config = check_dataset(data_yaml_path)
    if not data_config:
        logger.error("Dataset check failed")
        return
    
    # Run quick training
    training_success, training_output, run_name = run_quick_training(
        data_yaml_path, epochs=5, batch_size=8
    )
    
    if not training_success:
        logger.error("Training failed, cannot proceed with validation")
        return
    
    # Parse training results
    training_results = parse_results(training_output)
    logger.info(f"Training results: {training_results}")
    
    # Run validation
    weights_path = f"runs/quick_test/{run_name}/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = f"runs/quick_test/{run_name}/weights/last.pt"
    
    if os.path.exists(weights_path):
        validation_success, validation_output = run_validation(
            data_yaml_path, weights_path, run_name
        )
        
        if validation_success:
            validation_results = parse_results(validation_output)
            logger.info(f"Validation results: {validation_results}")
        else:
            logger.error("Validation failed")
            validation_results = {}
    else:
        logger.error(f"Weights file not found: {weights_path}")
        validation_results = {}
    
    # Save results
    results_summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_name': run_name,
        'data_config': data_config,
        'training_success': training_success,
        'training_results': training_results,
        'validation_success': validation_success if 'validation_success' in locals() else False,
        'validation_results': validation_results,
        'gpu_available': torch.cuda.is_available(),
        'device_used': '0' if torch.cuda.is_available() else 'cpu'
    }
    
    # Save to JSON file
    results_file = f"runs/quick_test/{run_name}/quick_test_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    logger.info("="*60)
    logger.info("QUICK TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Training success: {training_success}")
    if training_results:
        logger.info(f"Training mAP@0.5: {training_results.get('mAP@0.5', 'N/A')}")
        logger.info(f"Training mAP@0.5:0.95: {training_results.get('mAP@0.5:0.95', 'N/A')}")
    
    if 'validation_success' in locals():
        logger.info(f"Validation success: {validation_success}")
        if validation_results:
            logger.info(f"Validation mAP@0.5: {validation_results.get('mAP@0.5', 'N/A')}")
            logger.info(f"Validation mAP@0.5:0.95: {validation_results.get('mAP@0.5:0.95', 'N/A')}")
    
    logger.info("="*60)
    logger.info("QUICK TEST COMPLETED")
    logger.info("="*60)

if __name__ == "__main__":
    main() 