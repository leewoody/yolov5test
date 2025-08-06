#!/usr/bin/env python3
"""
Original Code Training Validation Script
使用原始代碼進行訓練驗證

This script uses the original YOLOv5 code to train and validate the model
on the regurgitation dataset with classification enabled.
"""

import os
import sys
import argparse
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

def check_dataset_structure(data_yaml_path):
    """檢查數據集結構"""
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        logger.info(f"Dataset configuration loaded from: {data_yaml_path}")
        logger.info(f"Classes: {data_config.get('names', [])}")
        logger.info(f"Number of classes: {data_config.get('nc', 0)}")
        
        # Check if paths exist
        train_path = Path(data_config.get('train', ''))
        val_path = Path(data_config.get('val', ''))
        test_path = Path(data_config.get('test', ''))
        
        logger.info(f"Train path: {train_path} - Exists: {train_path.exists()}")
        logger.info(f"Val path: {val_path} - Exists: {val_path.exists()}")
        logger.info(f"Test path: {test_path} - Exists: {test_path.exists()}")
        
        return data_config
    except Exception as e:
        logger.error(f"Error loading dataset configuration: {e}")
        return None

def check_gpu_availability():
    """檢查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU available: {gpu_count} devices")
        logger.info(f"Primary GPU: {gpu_name}")
        return True
    else:
        logger.warning("No GPU available, using CPU")
        return False

def run_training_command(cmd, description):
    """執行訓練命令"""
    logger.info(f"Starting {description}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"{description} completed successfully")
        logger.info(f"Output: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with error: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr

def run_validation_command(cmd, description):
    """執行驗證命令"""
    logger.info(f"Starting {description}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"{description} completed successfully")
        logger.info(f"Output: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with error: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr

def parse_training_results(output):
    """解析訓練結果"""
    results = {}
    
    # Look for mAP values in the output
    lines = output.split('\n')
    for line in lines:
        if 'mAP@0.5' in line:
            try:
                # Extract mAP@0.5 value
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
                # Extract mAP@0.5:0.95 value
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
    parser = argparse.ArgumentParser(description='Original Code Training Validation')
    parser.add_argument('--data', type=str, default='../Regurgitation-YOLODataset-1new/data.yaml', 
                       help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', 
                       help='Initial weights path')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--device', default='auto', 
                       help='Device to use (auto, cpu, 0, 1, etc.)')
    parser.add_argument('--project', default='runs/train', 
                       help='Save to project/name')
    parser.add_argument('--name', default='original_training', 
                       help='Save to project/name')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Number of workers')
    parser.add_argument('--save-period', type=int, default=10, 
                       help='Save checkpoint every x epochs')
    parser.add_argument('--patience', type=int, default=50, 
                       help='Early stopping patience')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], 
                       help='Freeze layers')
    parser.add_argument('--cache', action='store_true', 
                       help='Cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', 
                       help='Use weighted image selection for training')
    parser.add_argument('--multi-scale', action='store_true', 
                       help='Vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', 
                       help='Train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, default='SGD', 
                       help='Optimizer (SGD, Adam, etc.)')
    parser.add_argument('--sync-bn', action='store_true', 
                       help='Use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default='runs/train', 
                       help='Save to project/name')
    parser.add_argument('--name', default='exp', 
                       help='Save to project/name')
    parser.add_argument('--exist-ok', action='store_true', 
                       help='Existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', 
                       help='Quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', 
                       help='Cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, 
                       help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, 
                       help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], 
                       help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, 
                       help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, 
                       help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, 
                       help='Automatic DDP Multi-GPU argument, do not modify')
    
    args = parser.parse_args()
    
    # Create timestamp for unique run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{timestamp}"
    
    logger.info("="*60)
    logger.info("ORIGINAL CODE TRAINING VALIDATION")
    logger.info("="*60)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Check dataset structure
    data_config = check_dataset_structure(args.data)
    if not data_config:
        logger.error("Failed to load dataset configuration")
        return
    
    # Set device
    if args.device == 'auto':
        if gpu_available:
            device = '0'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Prepare training command
    train_cmd = [
        'python', 'train.py',
        '--data', args.data,
        '--weights', args.weights,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--imgsz', str(args.img_size),
        '--device', device,
        '--project', args.project,
        '--name', run_name,
        '--workers', str(args.workers),
        '--save-period', str(args.save_period),
        '--patience', str(args.patience),
        '--exist-ok'
    ]
    
    # Add optional arguments
    if args.cache:
        train_cmd.append('--cache')
    if args.image_weights:
        train_cmd.append('--image-weights')
    if args.multi_scale:
        train_cmd.append('--multi-scale')
    if args.single_cls:
        train_cmd.append('--single-cls')
    if args.sync_bn:
        train_cmd.append('--sync-bn')
    if args.quad:
        train_cmd.append('--quad')
    if args.cos_lr:
        train_cmd.append('--cos-lr')
    if args.label_smoothing > 0:
        train_cmd.extend(['--label-smoothing', str(args.label_smoothing)])
    
    # Run training
    success, output = run_training_command(train_cmd, "Training")
    
    if not success:
        logger.error("Training failed, stopping validation")
        return
    
    # Parse training results
    training_results = parse_training_results(output)
    logger.info(f"Training results: {training_results}")
    
    # Prepare validation command
    weights_path = f"{args.project}/{run_name}/weights/best.pt"
    if not os.path.exists(weights_path):
        weights_path = f"{args.project}/{run_name}/weights/last.pt"
    
    if os.path.exists(weights_path):
        val_cmd = [
            'python', 'val.py',
            '--data', args.data,
            '--weights', weights_path,
            '--imgsz', str(args.img_size),
            '--device', device,
            '--project', args.project,
            '--name', f"{run_name}_validation",
            '--exist-ok'
        ]
        
        # Run validation
        val_success, val_output = run_validation_command(val_cmd, "Validation")
        
        if val_success:
            validation_results = parse_training_results(val_output)
            logger.info(f"Validation results: {validation_results}")
        else:
            logger.error("Validation failed")
    else:
        logger.error(f"Weights file not found: {weights_path}")
    
    # Save results summary
    results_summary = {
        'timestamp': timestamp,
        'run_name': run_name,
        'data_config': data_config,
        'training_args': vars(args),
        'training_results': training_results,
        'gpu_available': gpu_available,
        'device_used': device
    }
    
    if 'validation_results' in locals():
        results_summary['validation_results'] = validation_results
    
    # Save to JSON file
    results_file = f"{args.project}/{run_name}/training_validation_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info("="*60)
    logger.info("TRAINING VALIDATION COMPLETED")
    logger.info("="*60)

if __name__ == "__main__":
    main() 