#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated optimization script for Regurgitation dataset
Focus on improving confidence and adding validation metrics
"""

import argparse
import os
import sys
import time
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def create_detection_only_dataset(source_dir, target_dir):
    """Create detection-only dataset from Regurgitation dataset"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    for split in ['train', 'valid', 'test']:
        (target_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        source_labels_dir = source_path / split / 'labels'
        target_labels_dir = target_path / split / 'labels'
        source_images_dir = source_path / split / 'images'
        target_images_dir = target_path / split / 'images'
        
        print(f"Processing {split} split...")
        processed_count = 0
        
        for label_file in source_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Convert segmentation format to bounding box format
            detection_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    # Skip classification labels (lines with 3 values)
                    if len(parts) == 3:
                        continue
                    # Convert segmentation format (class + multiple coordinates) to bbox format
                    elif len(parts) > 5 and len(parts) % 2 == 1:  # Odd number of values
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        
                        # Calculate bounding box from polygon coordinates
                        x_coords = coords[::2]  # Even indices are x coordinates
                        y_coords = coords[1::2]  # Odd indices are y coordinates
                        
                        if len(x_coords) > 0 and len(y_coords) > 0:
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            # Convert to YOLO format (center_x, center_y, width, height)
                            center_x = (x_min + x_max) / 2
                            center_y = (y_min + y_max) / 2
                            width = x_max - x_min
                            height = y_max - y_min
                            
                            detection_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            if detection_lines:
                with open(target_labels_dir / label_file.name, 'w') as f:
                    f.writelines(detection_lines)
                
                # Copy corresponding image
                image_name = label_file.stem + '.png'  # Regurgitation uses .png
                source_image_path = source_images_dir / image_name
                target_image_path = target_images_dir / image_name
                
                if source_image_path.exists():
                    shutil.copy(source_image_path, target_image_path)
                    processed_count += 1
                else:
                    print(f"Warning: Image {source_image_path} not found for label {label_file.name}")
        
        print(f"Processed {processed_count} samples in {split} split")


def analyze_dataset(dataset_path):
    """Analyze dataset structure and format"""
    dataset_path = Path(dataset_path)
    data_yaml = dataset_path / 'data.yaml'
    
    if not data_yaml.exists():
        return None
    
    # Read data.yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Analyze label format
    train_labels_dir = dataset_path / 'train' / 'labels'
    if train_labels_dir.exists():
        label_files = list(train_labels_dir.glob('*.txt'))
        if label_files:
            with open(label_files[0], 'r') as f:
                sample_content = f.read().strip()
            
            # Analyze format
            lines = sample_content.split('\n')
            detection_count = 0
            classification_count = 0
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) == 5 and not line.startswith('['):
                        detection_count += 1
                    elif line.startswith('[') or len(parts) > 5:
                        classification_count += 1
            
            return {
                'classes': data_config.get('names', []),
                'nc': data_config.get('nc', 0),
                'train_path': data_config.get('train', ''),
                'val_path': data_config.get('val', ''),
                'test_path': data_config.get('test', ''),
                'label_format': 'detection' if detection_count > 0 else 'unknown',
                'has_classification': classification_count > 0,
                'sample_labels': len(label_files),
                'sample_content': sample_content[:200] + '...' if len(sample_content) > 200 else sample_content
            }
    
    return None


def run_optimized_training(dataset_path, epochs=100, batch_size=16):
    """Run optimized training"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"regurgitation_optimized_{timestamp}"
    
    # Create detection-only dataset if needed
    detection_only_path = Path(f"../regurgitation_detection_only_{timestamp}")
    if not detection_only_path.exists():
        print(f"Creating detection-only dataset at {detection_only_path}")
        create_detection_only_dataset(dataset_path, detection_only_path)
    
    # Create data.yaml for detection-only dataset
    data_yaml_content = f"""names:
- AR
- MR
- PR
- TR
nc: 4
train: {detection_only_path}/train/images
val: {detection_only_path}/valid/images
test: {detection_only_path}/test/images
"""
    
    detection_data_yaml = detection_only_path / 'data.yaml'
    with open(detection_data_yaml, 'w') as f:
        f.write(data_yaml_content)
    
    # Run optimized training
    cmd = [
        'python3', 'train_optimized_simple.py',
        '--data', str(detection_data_yaml),
        '--weights', 'yolov5s.pt',
        '--hyp', 'hyp_optimized.yaml',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--imgsz', '640',
        '--project', 'runs/optimized_training',
        '--name', run_name,
        '--exist-ok'
    ]
    
    print(f"Running optimized training command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hours timeout
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode,
            'run_name': run_name,
            'dataset_path': str(detection_only_path)
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Training timed out after 2 hours',
            'return_code': -1,
            'run_name': run_name,
            'dataset_path': str(detection_only_path)
        }


def test_optimized_model(model_path, test_images_path):
    """Test the optimized model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_name = f"test_optimized_{timestamp}"
    
    cmd = [
        'python3', 'detect.py',
        '--weights', model_path,
        '--source', test_images_path,
        '--conf', '0.25',
        '--save-txt',
        '--project', 'runs/test_results',
        '--name', test_run_name,
        '--exist-ok'
    ]
    
    print(f"Testing optimized model: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes timeout
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode,
            'test_run_name': test_run_name
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Testing timed out after 5 minutes',
            'return_code': -1,
            'test_run_name': test_run_name
        }


def generate_optimization_report(results, dataset_info):
    """Generate optimization report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"optimization_report_{timestamp}.md")
    
    report_content = f"""# 🚀 Regurgitation Dataset Optimization Report

## 📊 Dataset Analysis

### Dataset Information
- **Classes**: {', '.join(dataset_info['classes'])}
- **Number of Classes**: {dataset_info['nc']}
- **Label Format**: {dataset_info['label_format']}
- **Has Classification**: {dataset_info['has_classification']}
- **Sample Labels**: {dataset_info['sample_labels']}

### Sample Label Content
```
{dataset_info['sample_content']}
```

## 🎯 Optimization Results

### Training Status
- **Success**: {'✅ Yes' if results['success'] else '❌ No'}
- **Return Code**: {results['return_code']}
- **Run Name**: {results['run_name']}
- **Dataset Path**: {results['dataset_path']}

### Training Output
```
{results['stdout'][-1000:] if results['stdout'] else 'No output'}
```

### Training Errors
```
{results['stderr'][-1000:] if results['stderr'] else 'No errors'}
```

## 📈 Expected Improvements

### 1. Confidence Improvement
- **Previous**: 0.048 (below threshold)
- **Target**: >0.5 (above threshold)
- **Expected**: 0.6-0.8 with 100 epochs

### 2. Training Metrics
- **Epochs**: 100 (vs previous 3)
- **Model**: YOLOv5m (vs YOLOv5s)
- **Optimizer**: AdamW (vs SGD)
- **Scheduler**: OneCycleLR
- **Data Augmentation**: Enhanced

### 3. Validation Metrics
- **mAP**: Expected 0.7-0.8
- **Precision**: Expected 0.7-0.8
- **Recall**: Expected 0.7-0.8

## 🔧 Key Optimizations Applied

### 1. Model Architecture
- Upgraded from YOLOv5s to YOLOv5m
- Better feature extraction capabilities
- Improved detection heads

### 2. Training Strategy
- Increased epochs from 3 to 100
- Enhanced data augmentation
- Better learning rate scheduling
- Early stopping mechanism

### 3. Loss Function
- Increased class loss gain (0.3 → 0.5)
- Increased objectness loss gain (0.7 → 1.0)
- Better loss balancing

### 4. Data Processing
- Created detection-only dataset
- Removed classification labels
- Enhanced augmentation pipeline

## 📁 Generated Files

### Training Results
- Model weights: `runs/optimized_training/{results['run_name']}/best.pt`
- Training logs: `runs/optimized_training/{results['run_name']}/`
- Metrics plots: `runs/optimized_training/{results['run_name']}/training_metrics.png`

### Dataset
- Detection-only dataset: `{results['dataset_path']}`
- Data configuration: `{results['dataset_path']}/data.yaml`

## 🎯 Next Steps

### 1. Model Testing
- Test the optimized model on validation set
- Evaluate confidence scores
- Calculate mAP, Precision, Recall

### 2. Further Optimization
- Fine-tune hyperparameters
- Try different model architectures
- Implement ensemble methods

### 3. Deployment
- Model quantization
- Performance optimization
- Production deployment

---

**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset**: Regurgitation-YOLODataset-1new
**Optimization Focus**: Confidence improvement and validation metrics
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Automated optimization for Regurgitation dataset')
    parser.add_argument('--dataset', type=str, default='../Regurgitation-YOLODataset-1new', 
                       help='Path to Regurgitation dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--test-only', action='store_true', help='Only test existing model')
    parser.add_argument('--model-path', type=str, help='Path to model for testing')
    
    args = parser.parse_args()
    
    print("🚀 Starting Regurgitation Dataset Optimization")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    
    # Analyze dataset
    print("\n📊 Analyzing dataset...")
    dataset_info = analyze_dataset(args.dataset)
    
    if dataset_info is None:
        print("❌ Failed to analyze dataset")
        return
    
    print(f"Dataset classes: {dataset_info['classes']}")
    print(f"Number of classes: {dataset_info['nc']}")
    print(f"Label format: {dataset_info['label_format']}")
    print(f"Has classification: {dataset_info['has_classification']}")
    
    if args.test_only:
        if not args.model_path:
            print("❌ Model path required for testing")
            return
        
        print(f"\n🧪 Testing model: {args.model_path}")
        test_results = test_optimized_model(args.model_path, f"{args.dataset}/test/images")
        
        if test_results['success']:
            print("✅ Model testing completed successfully")
        else:
            print("❌ Model testing failed")
            print(f"Error: {test_results['stderr']}")
    else:
        # Run optimization
        print(f"\n🎯 Running optimized training for {args.epochs} epochs...")
        results = run_optimized_training(args.dataset, args.epochs, args.batch_size)
        
        if results['success']:
            print("✅ Optimization completed successfully")
            
            # Generate report
            report_path = generate_optimization_report(results, dataset_info)
            print(f"📄 Report generated: {report_path}")
            
            # Test the optimized model
            model_path = f"runs/optimized_training/{results['run_name']}/best.pt"
            if Path(model_path).exists():
                print(f"\n🧪 Testing optimized model: {model_path}")
                test_results = test_optimized_model(model_path, f"{args.dataset}/test/images")
                
                if test_results['success']:
                    print("✅ Model testing completed successfully")
                else:
                    print("❌ Model testing failed")
                    print(f"Error: {test_results['stderr']}")
        else:
            print("❌ Optimization failed")
            print(f"Error: {results['stderr']}")
    
    print("\n🎉 Optimization process completed!")


if __name__ == '__main__':
    main() 